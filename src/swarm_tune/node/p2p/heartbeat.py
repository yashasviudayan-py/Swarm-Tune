"""
Heartbeat: liveness tracking for swarm peers.

Every node broadcasts a lightweight heartbeat message at a fixed interval.
Any peer that misses EVICTION_THRESHOLD consecutive heartbeats is considered
dead and removed from the active peer table.

This is the mechanism that enables straggler tolerance: the aggregator
only waits for peers that are in the live peer table.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from swarm_tune.config.settings import NodeSettings
    from swarm_tune.node.p2p.discovery import PeerDiscovery

log: structlog.BoundLogger = structlog.get_logger(__name__)

HEARTBEAT_INTERVAL_SECS: float = 5.0
EVICTION_THRESHOLD_SECS: float = 20.0  # 4 missed heartbeats


class Heartbeat:
    """
    Publishes periodic liveness pings and evicts silent peers.

    Runs as a background async task alongside the main training loop.
    """

    def __init__(self, settings: NodeSettings, discovery: PeerDiscovery) -> None:
        self._settings = settings
        self._discovery = discovery
        self._task: asyncio.Task[None] | None = None
        self._peer_last_seen: dict[str, float] = {}

    async def start(self) -> None:
        """Launch the heartbeat loop as a background task."""
        log.info("heartbeat starting", interval_secs=HEARTBEAT_INTERVAL_SECS)
        self._task = asyncio.create_task(self._loop(), name="heartbeat")

    async def stop(self) -> None:
        """Cancel the heartbeat loop and wait for it to finish."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("heartbeat stopped")

    def record_peer_seen(self, peer_id: str, multiaddr: str) -> None:
        """Called by the gossip layer when a heartbeat arrives from a peer."""
        now = time.monotonic()
        self._peer_last_seen[peer_id] = now
        self._discovery.register_peer(peer_id, multiaddr, now)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECS)
            await self._publish_heartbeat()
            self._evict_stale_peers()

    async def _publish_heartbeat(self) -> None:
        log.debug("publishing heartbeat", node_id=self._settings.node_id)
        # TODO(phase-1): publish heartbeat via gossip control topic

    def _evict_stale_peers(self) -> None:
        now = time.monotonic()
        stale = [
            peer_id
            for peer_id, last_seen in self._peer_last_seen.items()
            if (now - last_seen) > EVICTION_THRESHOLD_SECS
        ]
        for peer_id in stale:
            del self._peer_last_seen[peer_id]
            self._discovery.evict_peer(peer_id)
