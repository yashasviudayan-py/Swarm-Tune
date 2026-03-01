"""
Gossip protocol for gradient exchange.

Swarm-Tune uses libp2p's GossipSub pubsub protocol to broadcast
serialized gradient tensors across the swarm. Every node publishes
its gradients to a shared topic and subscribes to receive all peers'.

Topics:
  /swarm-tune/gradients/1.0.0  — serialized gradient payloads
  /swarm-tune/control/1.0.0    — round coordination messages

Why GossipSub instead of direct dial?
  - A direct dial to every peer scales as O(N²). GossipSub is O(log N).
  - Built-in message deduplication.
  - Works across NAT boundaries via relay peers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from swarm_tune.config.settings import NodeSettings

log: structlog.BoundLogger = structlog.get_logger(__name__)

GRADIENT_TOPIC = "/swarm-tune/gradients/1.0.0"
CONTROL_TOPIC = "/swarm-tune/control/1.0.0"

GradientHandler = Callable[[str, bytes], Coroutine[Any, Any, None]]


@dataclass
class GradientMessage:
    """Wire format for a gradient broadcast."""

    sender_id: str
    round_number: int
    payload: bytes  # serialized torch tensors — see trainer/serializer.py
    dataset_size: int  # used for weighted averaging in aggregator


class GossipProtocol:
    """
    Wraps libp2p GossipSub to provide a clean gradient broadcast API.

    Usage:
        gossip = GossipProtocol(settings)
        await gossip.start()
        gossip.on_gradient(my_handler)
        await gossip.broadcast_gradient(message)
    """

    def __init__(self, settings: NodeSettings) -> None:
        self._settings = settings
        self._handlers: list[GradientHandler] = []
        self._running = False
        # TODO(phase-3): store libp2p pubsub instance here

    async def start(self) -> None:
        """Subscribe to gradient and control topics."""
        log.info("gossip protocol starting", node_id=self._settings.node_id)
        self._running = True
        # TODO(phase-3): pubsub.subscribe(GRADIENT_TOPIC, self._on_raw_message)
        log.info("gossip subscribed", topics=[GRADIENT_TOPIC, CONTROL_TOPIC])

    async def stop(self) -> None:
        self._running = False
        log.info("gossip protocol stopped")

    def on_gradient(self, handler: GradientHandler) -> None:
        """Register a coroutine to be called on each incoming gradient message."""
        self._handlers.append(handler)

    async def broadcast_gradient(self, message: GradientMessage) -> None:
        """
        Serialize and publish a gradient message to all subscribed peers.

        This is a fire-and-forget call from the node's perspective.
        The aggregator collects responses via registered handlers.
        """
        if not self._running:
            raise RuntimeError("GossipProtocol is not running")

        log.debug(
            "broadcasting gradient",
            round=message.round_number,
            payload_bytes=len(message.payload),
        )
        # TODO(phase-3): serialise GradientMessage to bytes and publish to pubsub
        await asyncio.sleep(0)  # placeholder yield

    async def _on_raw_message(self, raw: bytes) -> None:
        """Deserialize an incoming pubsub message and dispatch to handlers."""
        # TODO(phase-3): deserialise bytes -> GradientMessage
        # TODO(phase-3): call each handler(sender_id, payload)
        pass
