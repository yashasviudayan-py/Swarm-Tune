"""
SwarmNode — the top-level orchestrator for a single Swarm-Tune peer.

This module wires together all subsystems:
  P2P (discovery + gossip + heartbeat) ←→ Trainer (model + gradients) ←→ Aggregator

Entrypoint for both direct Python invocation and Docker containers.

Usage:
    # Direct
    python -m swarm_tune.node.main

    # Via CLI
    swarm-tune
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import anyio
import click
import structlog
import torch
from libp2p.tools.async_service import background_trio_service  # type: ignore[attr-defined]

from swarm_tune.config.settings import NodeSettings
from swarm_tune.node.aggregator.averaging import PeerGradient
from swarm_tune.node.aggregator.timeout import TimeoutAggregator
from swarm_tune.node.p2p.discovery import PeerDiscovery
from swarm_tune.node.p2p.gossip import GossipProtocol, GradientMessage
from swarm_tune.node.p2p.heartbeat import Heartbeat
from swarm_tune.node.trainer.compressor import (
    Compressor,
    IdentityCompressor,
    TopKCompressor,
)
from swarm_tune.node.trainer.data import DataShardLoader
from swarm_tune.node.trainer.gradient import GradientExtractor
from swarm_tune.node.trainer.model import ModelShard
from swarm_tune.node.trainer.serializer import GradientSerializer

if TYPE_CHECKING:
    pass


def _configure_logging(settings: NodeSettings) -> None:
    """Set up structlog with the configured format and level."""
    log_level = getattr(logging, settings.log_level)
    processors: list[structlog.types.Processor]

    if settings.log_format == "json":
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def _build_compressor(settings: NodeSettings) -> Compressor:
    """Instantiate the configured gradient compressor."""
    if settings.compression == "topk":
        return TopKCompressor(k=settings.topk_ratio)
    return IdentityCompressor()


log: structlog.BoundLogger = structlog.get_logger(__name__)


class SwarmNode:
    """
    A single peer in the Swarm-Tune network.

    Orchestrates the full training loop:
      1. Discover peers via libp2p.
      2. Load local model shard and data shard.
      3. For each round:
         a. Run local forward + backward pass.
         b. Extract and compress gradients.
         c. Submit local gradient to aggregator.
         d. Broadcast gradients to peers via gossip.
         e. Wait for peer gradients (with timeout).
         f. Average and apply gradients (fall back to local if solo).
         g. Optionally checkpoint.
    """

    def __init__(self, settings: NodeSettings) -> None:
        self._settings = settings

        # Subsystems
        self._discovery = PeerDiscovery(settings)
        self._gossip = GossipProtocol(settings)
        self._heartbeat = Heartbeat(settings, self._discovery)
        self._model = ModelShard(settings)
        self._data_loader = DataShardLoader(settings.data_shard_path)
        self._extractor = GradientExtractor()
        self._serializer = GradientSerializer()
        self._compressor: Compressor = _build_compressor(settings)
        self._aggregator = TimeoutAggregator(settings)

        # Register gossip handler for incoming peer gradients
        self._gossip.on_gradient(self._on_peer_gradient)

        structlog.contextvars.bind_contextvars(node_id=settings.node_id)

    async def start(self) -> None:
        """Bootstrap the node: bind TCP, start mDNS, load model and data."""
        log.info("swarm node starting", node_id=self._settings.node_id)

        await self._discovery.start()
        await self._gossip.start()
        self._model.load()
        self._data_loader.load()

        log.info(
            "swarm node ready",
            device=self._settings.device,
            compression=self._settings.compression,
            data_shard=str(self._settings.data_shard_path),
            dataset_size=self._data_loader.dataset_size,
        )

    async def stop(self) -> None:
        """Graceful shutdown of all subsystems."""
        await self._heartbeat.stop()
        await self._gossip.stop()
        await self._discovery.stop()
        log.info("swarm node stopped")

    async def run(self) -> None:
        """
        Main entry point.  Starts all subsystems, runs background tasks
        (pubsub, heartbeat, heartbeat-receiver) in an anyio TaskGroup
        alongside the training loop, then shuts down cleanly.

        Ordering matters:
          1. start() — binds TCP, creates pubsub objects (no bootstrap yet)
          2. background_trio_service(pubsub) — starts handle_peer_queue so
             the zero-capacity internal channel has a consumer
          3. connect_bootstrap() — safe to connect now; notify_connected can
             enqueue without deadlocking
          4. training loop with heartbeat tasks
        """
        await self.start()

        try:
            assert self._discovery.pubsub is not None
            async with background_trio_service(self._discovery.pubsub):
                # Bootstrap connection is safe once pubsub is consuming events
                await self._discovery.connect_bootstrap()

                async with anyio.create_task_group() as tg:
                    # Heartbeat: publish liveness + evict stale peers
                    await self._heartbeat.start(tg)

                    # Heartbeat receiver: translate control-topic messages
                    # into peer-table updates
                    tg.start_soon(self._heartbeat_receiver)

                    for round_num in range(self._settings.num_rounds):
                        await self._training_round(round_num)

                    # All training rounds done; cancel background tasks cleanly
                    tg.cancel_scope.cancel()
        finally:
            await self.stop()

    async def _heartbeat_receiver(self) -> None:
        """Process incoming heartbeat messages from the control topic."""
        sub = self._discovery.control_subscription
        if sub is None:
            return
        while True:
            msg = await sub.get()
            try:
                data = msg.data.decode()
                node_id, addr = data.split("|", 1)
                if node_id != self._settings.node_id:
                    self._heartbeat.record_peer_seen(node_id, addr)
                    log.debug("heartbeat received", from_node=node_id, addr=addr)
            except (ValueError, UnicodeDecodeError):
                log.warning("malformed heartbeat message", raw=msg.data[:64])

    async def _training_round(self, round_num: int) -> None:
        """Execute one full round of decentralised training."""
        log.info("training round started", round=round_num)
        self._aggregator.open_round(round_num)

        # --- Step 1: local forward + backward ---
        inputs, targets = self._load_local_batch()
        output = self._model.forward(inputs)
        loss = self._compute_loss(output, targets)
        self._model.backward(loss)
        log.info("local backward complete", round=round_num, loss=f"{loss.item():.4f}")

        # --- Step 2: extract, compress, serialize ---
        local_gradients = self._extractor.extract(self._model.model)
        compressed = self._compressor.compress(local_gradients)
        payload = self._serializer.serialize(compressed)

        log.info(
            "gradient payload ready",
            round=round_num,
            num_params=len(local_gradients),
            total_elements=sum(g.numel() for g in local_gradients.values()),
            payload_bytes=len(payload),
            compression=self._settings.compression,
        )

        # --- Step 3: submit own gradient to aggregator ---
        # This ensures the local contribution is included even when there are
        # no peers (Phase 2 solo) or if gossip delivery back to self is slow.
        # TimeoutAggregator is idempotent: a duplicate from gossip loopback is dropped.
        self._aggregator.submit(
            PeerGradient(
                peer_id=self._settings.node_id,
                gradients=local_gradients,
                dataset_size=self._data_loader.dataset_size,
            )
        )

        # --- Step 4: broadcast to peers ---
        message = GradientMessage(
            sender_id=self._settings.node_id,
            round_number=round_num,
            payload=payload,
            dataset_size=self._data_loader.dataset_size,
        )
        await self._gossip.broadcast_gradient(message)

        # --- Step 5: wait for peers, then apply ---
        await self._aggregator.wait()
        try:
            averaged = self._aggregator.get_averaged_gradients()
            self._model.apply_averaged_gradients(averaged)
            log.info(
                "round complete (averaged)",
                round=round_num,
                loss=f"{loss.item():.4f}",
            )
        except ValueError as exc:
            # Insufficient peer responses — apply local gradient directly.
            # In solo mode (Phase 2) this is the normal path.
            # In Phase 3 with real peers it means a deferred round.
            log.warning(
                "insufficient peers, applying local gradient",
                round=round_num,
                reason=str(exc),
            )
            self._model.apply_averaged_gradients(local_gradients)

    async def _on_peer_gradient(self, sender_id: str, raw: bytes) -> None:
        """Gossip handler: deserialize, decompress, validate, and submit a peer's gradient."""
        try:
            compressed = self._serializer.deserialize(raw)
            gradients = self._compressor.decompress(compressed)
            validated = self._extractor.validate(gradients)
            self._aggregator.submit(
                PeerGradient(
                    peer_id=sender_id,
                    gradients=validated,
                    dataset_size=self._settings.batch_size,  # TODO: received in message
                )
            )
        except (ValueError, RuntimeError):
            log.warning("rejected gradient from peer", peer_id=sender_id, exc_info=True)

    def _load_local_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a mini-batch (inputs, targets) from the local data shard."""
        return self._data_loader.get_batch(self._settings.batch_size)

    def _compute_loss(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between model output and shard targets."""
        return torch.nn.functional.mse_loss(output, targets.to(output.device))


# ============================================================
# CLI entrypoint
# ============================================================


@click.command()
@click.option("--node-id", envvar="SWARM_NODE_ID", default="", help="Node identifier.")
@click.option("--host", envvar="SWARM_HOST", default="0.0.0.0")
@click.option("--port", envvar="SWARM_PORT", default=9000, type=int)
@click.option("--log-level", envvar="SWARM_LOG_LEVEL", default="INFO")
@click.option(
    "--log-format",
    envvar="SWARM_LOG_FORMAT",
    default="console",
    type=click.Choice(["json", "console"]),
)
def cli(
    node_id: str,
    host: str,
    port: int,
    log_level: str,
    log_format: str,
) -> None:
    """Start a Swarm-Tune peer node."""
    settings = NodeSettings(
        node_id=node_id,
        host=host,
        port=port,
        log_level=log_level,  # type: ignore[arg-type]
        log_format=log_format,  # type: ignore[arg-type]
    )
    _configure_logging(settings)
    # libp2p uses trio internally; run the entire node under the trio backend
    anyio.run(SwarmNode(settings).run, backend="trio")


if __name__ == "__main__":
    cli()
