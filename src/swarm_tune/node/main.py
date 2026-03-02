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

import asyncio
import logging
from typing import TYPE_CHECKING

import click
import structlog
import torch

from swarm_tune.config.settings import NodeSettings
from swarm_tune.node.aggregator.averaging import PeerGradient
from swarm_tune.node.aggregator.timeout import TimeoutAggregator
from swarm_tune.node.p2p.discovery import PeerDiscovery
from swarm_tune.node.p2p.gossip import GossipProtocol, GradientMessage
from swarm_tune.node.p2p.heartbeat import Heartbeat
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


log: structlog.BoundLogger = structlog.get_logger(__name__)


class SwarmNode:
    """
    A single peer in the Swarm-Tune network.

    Orchestrates the full training loop:
      1. Discover peers via libp2p.
      2. Load local model shard and data.
      3. For each round:
         a. Run local forward + backward pass.
         b. Extract gradients.
         c. Broadcast gradients to peers via gossip.
         d. Wait for peer gradients (with timeout).
         e. Average and apply gradients.
         f. Optionally checkpoint.
    """

    def __init__(self, settings: NodeSettings) -> None:
        self._settings = settings

        # Subsystems
        self._discovery = PeerDiscovery(settings)
        self._gossip = GossipProtocol(settings)
        self._heartbeat = Heartbeat(settings, self._discovery)
        self._model = ModelShard(settings)
        self._extractor = GradientExtractor()
        self._serializer = GradientSerializer()
        self._aggregator = TimeoutAggregator(settings)

        # Register gossip handler for incoming peer gradients
        self._gossip.on_gradient(self._on_peer_gradient)

        structlog.contextvars.bind_contextvars(node_id=settings.node_id)

    async def start(self) -> None:
        """Bootstrap the node: connect to peers and start background tasks."""
        log.info("swarm node starting", node_id=self._settings.node_id)

        await self._discovery.start()
        await self._gossip.start()
        await self._heartbeat.start()
        self._model.load()

        log.info("swarm node ready", device=self._settings.device)

    async def stop(self) -> None:
        """Graceful shutdown of all subsystems."""
        await self._heartbeat.stop()
        await self._gossip.stop()
        await self._discovery.stop()
        log.info("swarm node stopped")

    async def run(self) -> None:
        """Main training loop. Runs for num_rounds rounds."""
        await self.start()

        try:
            for round_num in range(self._settings.num_rounds):
                await self._training_round(round_num)
        finally:
            await self.stop()

    async def _training_round(self, round_num: int) -> None:
        """Execute one full round of decentralised training."""
        log.info("training round started", round=round_num)
        self._aggregator.open_round(round_num)

        # --- Step 1: local forward + backward ---
        batch = self._load_local_batch()
        output = self._model.forward(batch)
        loss = self._compute_loss(output, batch)
        self._model.backward(loss)
        log.info("local backward complete", round=round_num, loss=f"{loss.item():.4f}")

        # --- Step 2: extract and broadcast gradients ---
        local_gradients = self._extractor.extract(self._model.model)
        payload = self._serializer.serialize(local_gradients)

        message = GradientMessage(
            sender_id=self._settings.node_id,
            round_number=round_num,
            payload=payload,
            dataset_size=self._settings.batch_size,  # TODO: use actual shard size
        )
        await self._gossip.broadcast_gradient(message)

        # --- Step 3: wait for peers, then average ---
        await self._aggregator.wait()
        try:
            averaged = self._aggregator.get_averaged_gradients()
            self._model.apply_averaged_gradients(averaged)
            log.info("round complete", round=round_num, loss=f"{loss.item():.4f}")
        except ValueError as exc:
            log.warning("round deferred — insufficient peer responses", reason=str(exc))

    async def _on_peer_gradient(self, sender_id: str, raw: bytes) -> None:
        """Gossip handler: deserialize and submit a peer's gradient."""
        try:
            gradients = self._serializer.deserialize(raw)
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

    def _load_local_batch(self) -> torch.Tensor:
        """Load a mini-batch from the local data shard. Placeholder for Phase 2."""
        # TODO(phase-2): load real data from self._settings.data_shard_path
        return torch.randn(self._settings.batch_size, 128)

    def _compute_loss(self, output: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Compute training loss. Placeholder for Phase 2."""
        # TODO(phase-2): use real labels and a proper loss function
        return torch.nn.functional.mse_loss(output, batch)


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
    asyncio.run(SwarmNode(settings).run())


if __name__ == "__main__":
    cli()
