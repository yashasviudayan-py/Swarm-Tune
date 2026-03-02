"""
Peer selection abstraction.

Controls which peers a node broadcasts its gradients to.

At 20 nodes: AllPeersSelector returns every live peer. Simple, correct.
At 100 nodes: ClusterPeerSelector returns only the node's cluster members
(~10-20 peers). This keeps per-node connection count manageable on home
internet while hierarchical aggregation handles cross-cluster averaging.

The GossipProtocol never knows which selector is in use. It just asks
"who should I send to?" and broadcasts accordingly.

Adding a new selection strategy (e.g. random subset, reputation-weighted)
requires only a new class implementing the PeerSelector Protocol — zero
changes to the gossip or training loop code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    from swarm_tune.node.p2p.discovery import PeerInfo

log: structlog.BoundLogger = structlog.get_logger(__name__)


@runtime_checkable
class PeerSelector(Protocol):
    """Strategy interface for choosing gradient broadcast targets."""

    def select(self, live_peers: list[PeerInfo]) -> list[PeerInfo]:
        """
        Choose which peers to send this node's gradients to.

        Args:
            live_peers: the current live peer table snapshot.

        Returns:
            Subset of live_peers to broadcast to.
        """
        ...


class AllPeersSelector:
    """
    Broadcast to every live peer.

    Correct for ≤ ~30 nodes. Above that, consider ClusterPeerSelector
    to avoid O(N) connections per node.
    """

    def select(self, live_peers: list[PeerInfo]) -> list[PeerInfo]:
        log.debug("peer selection: all peers", count=len(live_peers))
        return live_peers


class ClusterPeerSelector:
    """
    Broadcast only to peers in this node's cluster.

    Used when the swarm is organized into clusters for hierarchical
    aggregation. Each node knows its cluster_id via NodeSettings.
    Only peers with a matching cluster_id are selected.

    Requires peers to advertise their cluster_id in their heartbeat
    payload (implemented in Phase 5+).
    """

    def __init__(self, cluster_id: int) -> None:
        self.cluster_id = cluster_id

    def select(self, live_peers: list[PeerInfo]) -> list[PeerInfo]:
        # TODO(phase-5): filter peers by cluster_id once heartbeat carries it
        # For now, fall back to all peers (safe default)
        log.debug(
            "peer selection: cluster peers (fallback to all, cluster metadata not yet implemented)",
            cluster_id=self.cluster_id,
            total_peers=len(live_peers),
        )
        return live_peers
