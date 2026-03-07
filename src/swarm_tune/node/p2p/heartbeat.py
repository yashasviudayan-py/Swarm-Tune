"""
Heartbeat: liveness tracking for swarm peers.

Every node broadcasts a lightweight heartbeat message at a fixed interval.
Any peer that misses EVICTION_THRESHOLD consecutive heartbeats is considered
dead and removed from the active peer table.

This is the mechanism that enables straggler tolerance: the aggregator
only waits for peers that are in the live peer table.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import anyio
import structlog

from swarm_tune.node.p2p.gossip import CONTROL_TOPIC

if TYPE_CHECKING:
    import anyio.abc

    from swarm_tune.config.settings import NodeSettings
    from swarm_tune.node.p2p.discovery import PeerDiscovery

log: structlog.BoundLogger = structlog.get_logger(__name__)

HEARTBEAT_INTERVAL_SECS: float = 5.0
EVICTION_THRESHOLD_SECS: float = 20.0  # 4 missed heartbeats


class Heartbeat:
    """
    Publishes periodic liveness pings and evicts silent peers.

    Runs as a background task alongside the main training loop.
    Started via start(task_group) which registers _loop() in the
    caller's anyio TaskGroup.
    """

    def __init__(self, settings: NodeSettings, discovery: PeerDiscovery) -> None:
        self._settings = settings
        self._discovery = discovery
        self._peer_last_seen: dict[str, float] = {}

    async def start(self, task_group: anyio.abc.TaskGroup) -> None:
        """Register the heartbeat loop as a background task in the given group."""
        log.info("heartbeat starting", interval_secs=HEARTBEAT_INTERVAL_SECS)
        task_group.start_soon(self._loop)

    async def stop(self) -> None:
        """No-op: lifecycle is managed by the task group passed to start()."""
        log.info("heartbeat stopped")

    def record_peer_seen(self, peer_id: str, multiaddr: str, libp2p_peer_id: str = "") -> None:
        """Called by the gossip layer when a heartbeat arrives from a peer."""
        now = time.monotonic()
        self._peer_last_seen[peer_id] = now
        self._discovery.register_peer(peer_id, multiaddr, now, libp2p_peer_id)

    async def _loop(self) -> None:
        while True:
            await anyio.sleep(HEARTBEAT_INTERVAL_SECS)
            await self._publish_heartbeat()
            self._evict_stale_peers()

    async def _publish_heartbeat(self) -> None:
        pubsub = self._discovery.pubsub
        own_multiaddr = self._discovery.own_multiaddr
        if pubsub is None or own_multiaddr is None:
            return

        # Wire format: "node_id|multiaddr|libp2p_peer_id"  (plain bytes, no pickle)
        # The libp2p_peer_id lets receivers cross-check that the node_id in the
        # heartbeat payload matches the cryptographic identity of the libp2p connection.
        libp2p_id = self._discovery.own_libp2p_id
        payload = f"{self._settings.node_id}|{own_multiaddr}|{libp2p_id}".encode()
        await pubsub.publish(CONTROL_TOPIC, payload)
        log.debug("heartbeat published", node_id=self._settings.node_id)

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
