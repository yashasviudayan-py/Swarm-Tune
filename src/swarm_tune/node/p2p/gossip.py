"""
Gossip protocol for gradient exchange.

Swarm-Tune uses libp2p's FloodSub pubsub protocol to broadcast
serialized gradient tensors across the swarm. Every node publishes
its gradients to a shared topic and subscribes to receive all peers'.

Topics:
  /swarm-tune/gradients/1.0.0  — serialized gradient payloads
  /swarm-tune/control/1.0.0    — heartbeats (managed by Heartbeat / PeerDiscovery)

Chunking
--------
The Noise protocol (libp2p's encryption layer) has a hard maximum of 65,535
bytes per frame.  Gradient payloads for even small models easily exceed this.
GossipProtocol handles chunking transparently:

  Sender: splits the encoded GradientMessage into ≤ MAX_CHUNK_SIZE byte
          chunks, each prefixed with a chunk header (transfer_id, index, total).

  Receiver: buffers incoming chunks by transfer_id until all have arrived,
            then reassembles and dispatches the full message.

Chunk pubsub frame (big-endian):
  [8B: transfer_id (uint64)]   — random, unique per broadcast
  [4B: chunk_index (uint32)]   — 0-based position in this transfer
  [4B: total_chunks (uint32)]  — total chunks in this transfer
  [N bytes: chunk data]

Inner GradientMessage frame (big-endian, carried inside reassembled chunks):
  [4B: sender_id length (uint32)]
  [4B: round_number (int32)]
  [8B: dataset_size (int64)]
  [N bytes: sender_id (UTF-8)]
  [M bytes: payload (serialized torch tensors)]

Why FloodSub instead of direct dial?
  - A direct dial to every peer scales as O(N²). FloodSub is O(log N).
  - Built-in message deduplication at the pubsub layer.
  - Works across NAT boundaries via relay peers.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anyio
import anyio.abc
import structlog

if TYPE_CHECKING:
    from libp2p.pubsub.subscription import ISubscriptionAPI  # type: ignore[attr-defined]

    from swarm_tune.config.settings import NodeSettings
    from swarm_tune.node.p2p.discovery import PeerDiscovery

log: structlog.BoundLogger = structlog.get_logger(__name__)

GRADIENT_TOPIC = "/swarm-tune/gradients/1.0.0"
CONTROL_TOPIC = "/swarm-tune/control/1.0.0"

# Noise protocol hard limit is 65,535 bytes; use 60,000 for headroom.
MAX_CHUNK_SIZE = 60_000

# Chunk frame: uint64 (transfer_id) | uint32 (chunk_index) | uint32 (total_chunks)
_CHUNK_HEADER = struct.Struct(">Q I I")

# Inner message frame: uint32 (sender_id_len) | int32 (round_number) | int64 (dataset_size)
_MSG_HEADER = struct.Struct(">I i q")

# Handler signature: (sender_id, serialized_payload, dataset_size) → coroutine
GradientHandler = Callable[[str, bytes, int], Coroutine[Any, Any, None]]


@dataclass
class GradientMessage:
    """Wire format for a gradient broadcast."""

    sender_id: str
    round_number: int
    payload: bytes  # serialized torch tensors — see trainer/serializer.py
    dataset_size: int  # used for weighted averaging in aggregator


@dataclass
class _PendingTransfer:
    """Accumulates chunks for a single in-flight transfer."""

    total_chunks: int
    chunks: dict[int, bytes] = field(default_factory=dict)


class GossipProtocol:
    """
    Wraps libp2p FloodSub to provide a clean gradient broadcast API.

    Handles message chunking transparently so callers never need to worry
    about the Noise protocol's 65,535-byte frame limit.

    Usage:
        gossip = GossipProtocol(settings, discovery)
        await gossip.start()                # subscribe to gradient topic
        gossip.on_gradient(my_handler)      # register incoming-gradient handler
        await gossip.run_receiver(tg)       # start background polling loop
        await gossip.broadcast_gradient(message)
    """

    def __init__(self, settings: NodeSettings, discovery: PeerDiscovery) -> None:
        self._settings = settings
        self._discovery = discovery
        self._handlers: list[GradientHandler] = []
        self._running = False
        self._gradient_sub: ISubscriptionAPI | None = None
        # Partial chunk buffers keyed by transfer_id.
        # Only the _receiver_loop accesses this — no locking needed.
        self._pending: dict[int, _PendingTransfer] = {}

    async def start(self) -> None:
        """
        Subscribe to the gradient topic.

        Must be called after PeerDiscovery.start() so the pubsub instance
        is available.  Subscribing just registers the queue — pubsub does
        not need to be fully running yet.
        """
        pubsub = self._discovery.pubsub
        if pubsub is None:
            raise RuntimeError("PeerDiscovery.start() must be called before GossipProtocol.start()")

        log.info("gossip protocol starting", node_id=self._settings.node_id)
        self._gradient_sub = await pubsub.subscribe(GRADIENT_TOPIC)
        self._running = True
        log.info("gossip subscribed", topics=[GRADIENT_TOPIC, CONTROL_TOPIC])

    async def stop(self) -> None:
        self._running = False
        log.info("gossip protocol stopped")

    def on_gradient(self, handler: GradientHandler) -> None:
        """Register a coroutine to be called on each fully reassembled gradient message."""
        self._handlers.append(handler)

    async def run_receiver(self, task_group: anyio.abc.TaskGroup) -> None:
        """
        Start the background task that polls the gradient subscription queue
        and dispatches incoming messages to registered handlers.

        Must be called inside an anyio TaskGroup that outlives all training
        rounds (e.g., the same group that runs the heartbeat).
        """
        task_group.start_soon(self._receiver_loop)

    async def _receiver_loop(self) -> None:
        """
        Poll the gradient subscription indefinitely.  For each incoming chunk,
        attempt reassembly; when a transfer is complete, dispatch to handlers.

        Runs until the enclosing TaskGroup is cancelled (end of training).
        Any error in deserialization or handler code is logged and skipped —
        the loop never crashes.
        """
        if self._gradient_sub is None:
            log.warning("gradient subscription not initialized; receiver not starting")
            return

        while True:
            msg = await self._gradient_sub.get()
            try:
                full_data = self._process_chunk(msg.data)
            except (struct.error, ValueError) as exc:
                log.warning("malformed chunk dropped", error=str(exc), raw_len=len(msg.data))
                continue

            if full_data is None:
                continue  # transfer still in progress

            try:
                await self._on_raw_message(full_data)
            except Exception:
                log.warning("unhandled error dispatching gradient message", exc_info=True)

    async def broadcast_gradient(self, message: GradientMessage) -> None:
        """
        Serialize and publish a gradient message to all subscribed peers.

        Large messages are split into ≤ MAX_CHUNK_SIZE chunks automatically
        to stay below the Noise protocol frame limit.  Each chunk is a
        separate pubsub message; the receiver reassembles them.

        This is a fire-and-forget call from the node's perspective.
        """
        if not self._running:
            raise RuntimeError("GossipProtocol is not running")

        pubsub = self._discovery.pubsub
        if pubsub is None:
            raise RuntimeError("PeerDiscovery pubsub is not available")

        full_data = self._encode_message(message)
        chunks = [
            full_data[i : i + MAX_CHUNK_SIZE]
            for i in range(0, len(full_data), MAX_CHUNK_SIZE)
        ]
        total = len(chunks)
        transfer_id = int.from_bytes(os.urandom(8), "big")

        log.debug(
            "broadcasting gradient",
            round=message.round_number,
            payload_bytes=len(message.payload),
            total_bytes=len(full_data),
            num_chunks=total,
        )

        for idx, chunk in enumerate(chunks):
            frame = _CHUNK_HEADER.pack(transfer_id, idx, total) + chunk
            await pubsub.publish(GRADIENT_TOPIC, frame)

    def _process_chunk(self, raw: bytes) -> bytes | None:
        """
        Accumulate an incoming chunk frame.

        Returns the fully reassembled message bytes when all chunks have
        arrived, or None if the transfer is still in progress.

        Raises ValueError / struct.error on malformed frames.
        """
        if len(raw) < _CHUNK_HEADER.size:
            raise ValueError(
                f"chunk frame too short: {len(raw)} < {_CHUNK_HEADER.size} bytes"
            )

        transfer_id, chunk_idx, total_chunks = _CHUNK_HEADER.unpack_from(raw)
        chunk_data = raw[_CHUNK_HEADER.size :]

        if transfer_id not in self._pending:
            self._pending[transfer_id] = _PendingTransfer(total_chunks=total_chunks)

        transfer = self._pending[transfer_id]

        if chunk_idx in transfer.chunks:
            log.debug(
                "duplicate chunk ignored",
                transfer_id=transfer_id,
                chunk_idx=chunk_idx,
            )
            return None

        transfer.chunks[chunk_idx] = chunk_data

        if len(transfer.chunks) < transfer.total_chunks:
            return None  # still waiting for more chunks

        # All chunks have arrived — reassemble in order
        del self._pending[transfer_id]
        return b"".join(transfer.chunks[i] for i in range(transfer.total_chunks))

    async def _on_raw_message(self, raw: bytes) -> None:
        """Decode a fully reassembled message and dispatch to handlers."""
        try:
            message = self._decode_message(raw)
        except (struct.error, UnicodeDecodeError, ValueError) as exc:
            log.warning(
                "malformed gradient message dropped",
                error=str(exc),
                raw_len=len(raw),
            )
            return

        log.debug(
            "gradient message received",
            sender_id=message.sender_id,
            round=message.round_number,
            dataset_size=message.dataset_size,
            payload_bytes=len(message.payload),
        )

        for handler in self._handlers:
            try:
                await handler(message.sender_id, message.payload, message.dataset_size)
            except Exception:
                log.warning(
                    "gradient handler raised",
                    sender_id=message.sender_id,
                    exc_info=True,
                )

    @staticmethod
    def _encode_message(message: GradientMessage) -> bytes:
        """Pack a GradientMessage into bytes (inner wire format, before chunking)."""
        sender_bytes = message.sender_id.encode("utf-8")
        header = _MSG_HEADER.pack(
            len(sender_bytes),
            message.round_number,
            message.dataset_size,
        )
        return header + sender_bytes + message.payload

    @staticmethod
    def _decode_message(data: bytes) -> GradientMessage:
        """Unpack bytes (inner wire format, after reassembly) into a GradientMessage."""
        if len(data) < _MSG_HEADER.size:
            raise ValueError(
                f"gradient message too short: {len(data)} < {_MSG_HEADER.size} bytes"
            )

        sender_len, round_number, dataset_size = _MSG_HEADER.unpack_from(data)
        offset = _MSG_HEADER.size

        if len(data) < offset + sender_len:
            raise ValueError(
                f"gradient message header claims sender_id extends beyond packet "
                f"({sender_len} bytes at offset {offset}, total={len(data)})"
            )

        sender_id = data[offset : offset + sender_len].decode("utf-8")
        payload = data[offset + sender_len :]

        return GradientMessage(
            sender_id=sender_id,
            round_number=round_number,
            payload=payload,
            dataset_size=dataset_size,
        )
