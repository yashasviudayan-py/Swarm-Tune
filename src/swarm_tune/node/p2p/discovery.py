"""
Peer discovery for the Swarm-Tune P2P network.

Strategy:
  - Local simulation (Docker): mDNS-based discovery within the Docker bridge network.
  - Internet deployment: libp2p Kademlia DHT seeded from bootstrap peers.

There is NO central tracker. A bootstrap peer is simply the first known
address — once connected, the DHT takes over and the node is fully
decentralised.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from swarm_tune.config.settings import NodeSettings

log: structlog.BoundLogger = structlog.get_logger(__name__)


@dataclass
class PeerInfo:
    """Immutable snapshot of a discovered peer."""

    peer_id: str
    multiaddr: str
    last_seen: float = field(default=0.0)


class PeerDiscovery:
    """
    Manages peer discovery and maintains the live peer table.

    The peer table is a dict[peer_id -> PeerInfo]. Peers are evicted
    when the Heartbeat component stops receiving their liveness signals.

    Phase 1 deliverable: this class connects to bootstrap peers and
    populates the peer table so gradient sync can begin.
    """

    def __init__(self, settings: NodeSettings) -> None:
        self._settings = settings
        self._peers: dict[str, PeerInfo] = {}
        self._running = False
        # TODO(phase-1): initialise libp2p host here

    async def start(self) -> None:
        """
        Start the libp2p host and connect to bootstrap peers.

        Raises:
            RuntimeError: if the host cannot bind to the configured port.
        """
        log.info(
            "starting peer discovery",
            node_id=self._settings.node_id,
            host=self._settings.host,
            port=self._settings.port,
        )
        self._running = True
        # TODO(phase-1): create libp2p.new_node(), start mDNS / DHT
        await self._connect_to_bootstrap_peers()

    async def stop(self) -> None:
        """Gracefully shut down the libp2p host."""
        self._running = False
        # TODO(phase-1): host.close()
        log.info("peer discovery stopped", node_id=self._settings.node_id)

    async def _connect_to_bootstrap_peers(self) -> None:
        for addr in self._settings.bootstrap_peers:
            try:
                log.info("connecting to bootstrap peer", addr=addr)
                # TODO(phase-1): await host.connect(parse_multiaddr(addr))
                await asyncio.sleep(0)  # placeholder yield
            except Exception:
                log.warning("failed to connect to bootstrap peer", addr=addr, exc_info=True)

    def register_peer(self, peer_id: str, multiaddr: str, timestamp: float) -> None:
        """Called by the Heartbeat when a liveness signal arrives."""
        self._peers[peer_id] = PeerInfo(peer_id=peer_id, multiaddr=multiaddr, last_seen=timestamp)

    def evict_peer(self, peer_id: str) -> None:
        """Called by the Heartbeat when a peer times out."""
        self._peers.pop(peer_id, None)
        log.info("peer evicted", peer_id=peer_id)

    def get_live_peers(self) -> list[PeerInfo]:
        """Return a snapshot of the current live peer table."""
        return list(self._peers.values())

    @property
    def peer_count(self) -> int:
        return len(self._peers)
