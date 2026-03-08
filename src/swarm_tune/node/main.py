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
from swarm_tune.node.metrics import MetricsStore, run_metrics_server
from swarm_tune.node.p2p.discovery import PeerDiscovery
from swarm_tune.node.p2p.gossip import GossipProtocol, GradientMessage
from swarm_tune.node.p2p.heartbeat import Heartbeat
from swarm_tune.node.p2p.peer_selector import BanList
from swarm_tune.node.trainer.compressor import (
    Compressor,
    IdentityCompressor,
    TopKCompressor,
)
from swarm_tune.node.trainer.data import DataShardLoader, HFDataShardLoader, create_data_loader
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
            structlog.processors.format_exc_info,
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
        self._gossip = GossipProtocol(settings, self._discovery)
        self._heartbeat = Heartbeat(settings, self._discovery)
        self._model = ModelShard(settings)
        self._data_loader: DataShardLoader | HFDataShardLoader = create_data_loader(settings)
        self._extractor = GradientExtractor()
        self._serializer = GradientSerializer()
        self._compressor: Compressor = _build_compressor(settings)
        self._aggregator = TimeoutAggregator(settings)
        self._ban_list = BanList(ban_duration_secs=settings.rejection_ban_duration_secs)
        self._metrics = MetricsStore(
            node_id=settings.node_id,
            total_rounds=settings.num_rounds,
        )

        # Register gossip handler for incoming peer gradients
        self._gossip.on_gradient(self._on_peer_gradient)

        structlog.contextvars.bind_contextvars(node_id=settings.node_id)

    async def start(self) -> None:
        """Bootstrap the node: bind TCP, start mDNS, load model and data."""
        log.info("swarm node starting", node_id=self._settings.node_id)

        await self._discovery.start()
        await self._gossip.start()
        # Model and data loading are blocking (disk I/O + HuggingFace downloads).
        # Run in a worker thread so the async event loop stays responsive.
        await anyio.to_thread.run_sync(self._model.load)
        await anyio.to_thread.run_sync(self._data_loader.load)

        metrics_port = self._settings.port + 100
        log.info(
            "swarm node ready",
            device=self._settings.device,
            compression=self._settings.compression,
            dataset_size=self._data_loader.dataset_size,
            metrics_port=metrics_port,
        )
        # Human-readable startup summary so non-developer participants can
        # confirm the node is live without parsing JSON logs.
        print(
            f"\n  Swarm-Tune node '{self._settings.node_id}' is READY\n"
            f"  Device: {self._settings.device} | "
            f"Model: {self._settings.model_name} | "
            f"Rounds: {self._settings.num_rounds}\n"
            f"  Metrics: http://0.0.0.0:{metrics_port}/metrics\n"
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
            if self._discovery.pubsub is None:
                raise RuntimeError(
                    "Pubsub not initialized — discovery.start() must be called first."
                )
            async with background_trio_service(self._discovery.pubsub):
                # Bootstrap connection is safe once pubsub is consuming events
                await self._discovery.connect_bootstrap()

                async with anyio.create_task_group() as tg:
                    # Metrics sidecar: lightweight HTTP /metrics endpoint
                    metrics_port = self._settings.port + 100
                    tg.start_soon(run_metrics_server, self._metrics, metrics_port)

                    # Heartbeat: publish liveness + evict stale peers
                    await self._heartbeat.start(tg)

                    # Heartbeat receiver: translate control-topic messages
                    # into peer-table updates
                    tg.start_soon(self._heartbeat_receiver)

                    # Gradient receiver: poll gradient topic and dispatch to
                    # _on_peer_gradient for each incoming peer gradient
                    await self._gossip.run_receiver(tg)

                    try:
                        for round_num in range(self._settings.num_rounds):
                            await self._training_round(round_num)
                            # Periodic checkpoint save.
                            n = self._settings.checkpoint_every_n_rounds
                            if n > 0 and (round_num + 1) % n == 0:
                                ckpt = (
                                    self._settings.checkpoint_dir
                                    / f"{self._settings.node_id}_round_{round_num + 1}.pt"
                                )
                                try:
                                    self._model.save_checkpoint(ckpt)
                                except OSError as exc:
                                    log.error(
                                        "checkpoint save failed",
                                        path=str(ckpt),
                                        error=str(exc),
                                    )
                        log.info("training complete", total_rounds=self._settings.num_rounds)
                    finally:
                        # Final checkpoint on clean completion, error, or SIGTERM.
                        # Shield from cancellation so a Docker stop / SIGTERM does not
                        # abort the write mid-flight and corrupt the checkpoint file.
                        final_ckpt = (
                            self._settings.checkpoint_dir / f"{self._settings.node_id}_final.pt"
                        )
                        with anyio.CancelScope(shield=True):
                            try:
                                self._model.save_checkpoint(final_ckpt)
                            except OSError as exc:
                                log.error(
                                    "final checkpoint save failed",
                                    path=str(final_ckpt),
                                    error=str(exc),
                                )

                    # All training rounds done; cancel background tasks cleanly
                    tg.cancel_scope.cancel()
        finally:
            await self.stop()

    async def _heartbeat_receiver(self) -> None:
        """Process incoming heartbeat messages from the control topic.

        Wire format (v2): "node_id|multiaddr|libp2p_peer_id"
        Wire format (v1): "node_id|multiaddr"  (backward compat — older nodes)
        """
        sub = self._discovery.control_subscription
        if sub is None:
            return
        while True:
            msg = await sub.get()
            try:
                data = msg.data.decode()
                parts = data.split("|", 2)
                if len(parts) < 2:
                    raise ValueError("too few fields")
                node_id = parts[0]
                addr = parts[1]
                libp2p_peer_id = parts[2] if len(parts) == 3 else ""
                if node_id != self._settings.node_id:
                    self._heartbeat.record_peer_seen(node_id, addr, libp2p_peer_id)
                    log.debug("heartbeat received", from_node=node_id, addr=addr)
            except (ValueError, UnicodeDecodeError):
                log.warning("malformed heartbeat message", raw=msg.data[:64])

    async def _training_round(self, round_num: int) -> None:
        """Execute one full round of decentralised training."""
        log.info("training round started", round=round_num)
        self._aggregator.open_round(round_num)
        # Keep metrics store current so the sidecar serves live data.
        live_peers = self._discovery.get_live_peers()
        self._metrics.update_peers([p.peer_id for p in live_peers])

        # --- Step 1: local forward + backward ---
        inputs, targets = self._load_local_batch()
        try:
            loss = self._model.compute_loss(inputs, targets)
            self._model.backward(loss)
        except RuntimeError as exc:
            # Catch GPU out-of-memory: log and defer the round rather than crash.
            if "out of memory" in str(exc).lower():
                log.error(
                    "OOM during forward/backward — round skipped",
                    round=round_num,
                    error=str(exc),
                )
                self._metrics.record_deferred()
                return
            raise
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
        #
        # IMPORTANT: submit the compress→decompress version, not the raw gradient.
        # Peer gradients always arrive after compress→serialize→deserialize→decompress,
        # so with TopKCompressor the non-top-K elements are zeroed out. Submitting the
        # raw local_gradients would mix representations in the FedAvg pool and produce
        # biased averages for non-top-K elements.  IdentityCompressor makes this a no-op.
        local_for_averaging = self._compressor.decompress(compressed)
        # Validate own gradient before submitting — a NaN from an exploding loss
        # would silently corrupt every peer's average if not caught here.
        try:
            self._extractor.validate(local_for_averaging)
        except ValueError as exc:
            log.error(
                "local gradient failed validation — round skipped",
                round=round_num,
                reason=str(exc),
            )
            return
        self._aggregator.submit(
            PeerGradient(
                peer_id=self._settings.node_id,
                gradients=local_for_averaging,
                dataset_size=self._data_loader.dataset_size,
            )
        )

        # --- Step 4: broadcast to peers ---
        # In adversarial mode, replace the payload with NaN-filled tensors.
        # The local gradient submitted in step 3 is still real — this node's
        # own model continues to train correctly. Only the broadcast is poisoned.
        # Receiving nodes must detect this via GradientExtractor.validate() and
        # reject it; the swarm must continue without this node's contribution.
        broadcast_payload = payload
        if self._settings.adversarial:
            poisoned = {
                name: torch.full_like(g, float("nan")) for name, g in local_for_averaging.items()
            }
            broadcast_payload = self._serializer.serialize(poisoned)
            log.warning(
                "adversarial mode: broadcasting NaN gradient payload",
                round=round_num,
            )

        message = GradientMessage(
            sender_id=self._settings.node_id,
            round_number=round_num,
            payload=broadcast_payload,
            dataset_size=self._data_loader.dataset_size,
        )
        await self._gossip.broadcast_gradient(message)
        self._metrics.bytes_sent += len(broadcast_payload)

        # --- Step 5: wait for peers, then apply ---
        await self._aggregator.wait()
        try:
            averaged = self._aggregator.get_averaged_gradients()
            self._model.apply_averaged_gradients(averaged)
            self._metrics.record_round(round_num, loss.item())
            log.info(
                "round complete (averaged)",
                round=round_num,
                loss=f"{loss.item():.4f}",
            )
        except ValueError as exc:
            # Insufficient peer responses — apply local gradient directly.
            # In solo mode (Phase 2) this is the normal path.
            # In Phase 3 with real peers it means a deferred round.
            self._metrics.record_deferred()
            self._metrics.record_round(round_num, loss.item())
            log.warning(
                "insufficient peers, applying local gradient",
                round=round_num,
                reason=str(exc),
            )
            self._model.apply_averaged_gradients(local_for_averaging)

    async def _on_peer_gradient(
        self,
        sender_id: str,
        authenticated_libp2p_id: str,
        raw: bytes,
        dataset_size: int,
        round_number: int,
    ) -> None:
        """Gossip handler: deserialize, decompress, validate, and submit a peer's gradient.

        sender_id            -- application-level claim from the wire payload (UNVERIFIED)
        authenticated_libp2p_id -- libp2p Noise-verified peer ID (used for ban list)
        """
        self._metrics.bytes_received += len(raw)

        # Ban-list lookup uses the authenticated ID — an attacker claiming a different
        # sender_id cannot escape a ban by spoofing the application-level field.
        ban_key = authenticated_libp2p_id or sender_id
        if self._ban_list.is_banned(ban_key):
            log.debug(
                "gradient dropped from banned peer",
                peer_id=sender_id,
                authenticated_id=authenticated_libp2p_id,
            )
            return

        # Drop gradients from a previous round — late arrivals must not contaminate
        # the current round's FedAvg pool.
        current = self._aggregator.current_round
        if round_number != current:
            log.debug(
                "stale gradient dropped",
                peer_id=sender_id,
                msg_round=round_number,
                current_round=current,
            )
            return

        rejected = False
        try:
            compressed = self._serializer.deserialize(raw)
            gradients = self._compressor.decompress(compressed)
            validated = self._extractor.validate(
                gradients, max_norm_rms=self._settings.gradient_max_norm_rms
            )
            peer_ip = self._discovery.get_peer_ip(sender_id)
            self._aggregator.submit(
                PeerGradient(
                    peer_id=sender_id,
                    gradients=validated,
                    dataset_size=dataset_size,
                    peer_ip=peer_ip,
                )
            )
        except (ValueError, RuntimeError):
            # Expected operational failures: bad SWRM magic, NaN/Inf gradient,
            # norm threshold exceeded, duplicate round submission.
            rejected = True
            self._metrics.record_rejection()
            log.warning("rejected gradient from peer", peer_id=sender_id, exc_info=True)
        except Exception:
            # Unexpected failure — likely a programming bug or memory pressure.
            # Log at ERROR so it is not confused with a routine network rejection.
            rejected = True
            self._metrics.record_rejection()
            log.error(
                "unexpected error processing peer gradient",
                peer_id=sender_id,
                exc_info=True,
            )

        # Update ban list tracking using authenticated ID so spoofed sender_id
        # cannot bypass rejection rate tracking.
        self._ban_list.record_round(ban_key, rejected)
        self._ban_list.check_and_ban(ban_key, self._settings.rejection_ban_threshold)

    def _load_local_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a mini-batch (inputs, targets) from the local data shard."""
        return self._data_loader.get_batch(self._settings.batch_size)


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
