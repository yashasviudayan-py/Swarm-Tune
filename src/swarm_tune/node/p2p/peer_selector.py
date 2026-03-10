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

import time
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


class BanList:
    """
    Tracks temporarily banned peers for Sybil resistance (Phase 5+).

    A peer is banned when its gradient rejection rate exceeds the configured
    threshold over a rolling window of rounds. Bans expire after a fixed
    duration.

    Thread-safety: all methods are called from the async training loop
    (single coroutine). No locking required.
    """

    def __init__(self, ban_duration_secs: float = 600.0) -> None:
        self._ban_duration = ban_duration_secs
        # peer_id -> (ban_expiry_timestamp, rejection_count, total_rounds)
        self._bans: dict[str, float] = {}
        self._rejections: dict[str, int] = {}
        self._rounds: dict[str, int] = {}

    def record_round(self, peer_id: str, rejected: bool) -> None:
        """Record one round outcome for a peer."""
        self._rounds[peer_id] = self._rounds.get(peer_id, 0) + 1
        if rejected:
            self._rejections[peer_id] = self._rejections.get(peer_id, 0) + 1

    def check_and_ban(self, peer_id: str, threshold: float) -> bool:
        """
        Check if the peer's rejection rate exceeds threshold and ban if so.

        Returns True if the peer was newly banned.
        """
        total = self._rounds.get(peer_id, 0)
        if total < 5:  # need at least 5 rounds of data before banning
            return False
        rate = self._rejections.get(peer_id, 0) / total
        if rate > threshold and peer_id not in self._bans:
            expiry = time.monotonic() + self._ban_duration
            self._bans[peer_id] = expiry
            log.warning(
                "peer temporarily banned (high rejection rate)",
                peer_id=peer_id,
                rejection_rate=round(rate, 3),
                ban_until=expiry,
            )
            return True
        return False

    def is_banned(self, peer_id: str) -> bool:
        """True if the peer is currently under a temporary ban."""
        expiry = self._bans.get(peer_id)
        if expiry is None:
            return False
        if time.monotonic() >= expiry:
            del self._bans[peer_id]
            # H1 fix: reset rejection counters on ban expiry so the peer gets
            # a clean slate. Without this, historical rejection rates cause
            # immediate re-banning after the ban expires.
            self._rejections.pop(peer_id, None)
            self._rounds.pop(peer_id, None)
            log.info("peer ban expired, counters reset", peer_id=peer_id)
            return False
        return True

    def banned_peers(self) -> set[str]:
        """Return the set of currently banned peer IDs (pruning expired bans)."""
        now = time.monotonic()
        expired = [p for p, exp in self._bans.items() if now >= exp]
        for p in expired:
            del self._bans[p]
        return set(self._bans.keys())


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
