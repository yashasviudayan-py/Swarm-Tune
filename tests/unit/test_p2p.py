"""Unit tests for the P2P subsystem: discovery, heartbeat."""

from __future__ import annotations

import time

import pytest

from swarm_tune.node.p2p.discovery import PeerDiscovery
from swarm_tune.node.p2p.heartbeat import EVICTION_THRESHOLD_SECS, Heartbeat


class TestPeerDiscovery:
    @pytest.mark.unit
    def test_starts_with_empty_peer_table(self, base_settings: object) -> None:
        discovery = PeerDiscovery(base_settings)  # type: ignore[arg-type]
        assert discovery.peer_count == 0
        assert discovery.get_live_peers() == []

    @pytest.mark.unit
    def test_register_peer(self, base_settings: object) -> None:
        discovery = PeerDiscovery(base_settings)  # type: ignore[arg-type]
        discovery.register_peer("peer_1", "/ip4/127.0.0.1/tcp/9001", time.monotonic())
        assert discovery.peer_count == 1
        peers = discovery.get_live_peers()
        assert peers[0].peer_id == "peer_1"

    @pytest.mark.unit
    def test_evict_peer(self, base_settings: object) -> None:
        discovery = PeerDiscovery(base_settings)  # type: ignore[arg-type]
        discovery.register_peer("peer_1", "/ip4/127.0.0.1/tcp/9001", time.monotonic())
        discovery.evict_peer("peer_1")
        assert discovery.peer_count == 0

    @pytest.mark.unit
    def test_evict_unknown_peer_is_noop(self, base_settings: object) -> None:
        discovery = PeerDiscovery(base_settings)  # type: ignore[arg-type]
        discovery.evict_peer("nonexistent")  # should not raise
        assert discovery.peer_count == 0

    @pytest.mark.unit
    def test_register_updates_existing_peer(self, base_settings: object) -> None:
        discovery = PeerDiscovery(base_settings)  # type: ignore[arg-type]
        t1 = time.monotonic()
        t2 = t1 + 5.0
        discovery.register_peer("peer_1", "/ip4/127.0.0.1/tcp/9001", t1)
        discovery.register_peer("peer_1", "/ip4/127.0.0.1/tcp/9001", t2)
        assert discovery.peer_count == 1
        assert discovery.get_live_peers()[0].last_seen == t2


class TestHeartbeat:
    @pytest.mark.unit
    def test_record_seen_registers_peer(self, base_settings: object) -> None:
        discovery = PeerDiscovery(base_settings)  # type: ignore[arg-type]
        hb = Heartbeat(base_settings, discovery)  # type: ignore[arg-type]
        hb.record_peer_seen("peer_1", "/ip4/127.0.0.1/tcp/9001")
        assert discovery.peer_count == 1

    @pytest.mark.unit
    def test_stale_peer_is_evicted(self, base_settings: object) -> None:
        from swarm_tune.config.settings import NodeSettings

        discovery = PeerDiscovery(base_settings)  # type: ignore[arg-type]
        hb = Heartbeat(base_settings, discovery)  # type: ignore[arg-type]
        hb.record_peer_seen("peer_1", "/ip4/127.0.0.1/tcp/9001")

        # Manually back-date the last_seen time beyond the configurable eviction threshold.
        # Using settings.heartbeat_eviction_secs (not the old hardcoded constant) so this
        # test stays correct when the threshold changes.
        eviction_secs = (
            base_settings.heartbeat_eviction_secs  # type: ignore[union-attr]
            if isinstance(base_settings, NodeSettings)
            else EVICTION_THRESHOLD_SECS
        )
        hb._peer_last_seen["peer_1"] -= eviction_secs + 1
        hb._evict_stale_peers()

        assert discovery.peer_count == 0

    @pytest.mark.unit
    def test_fresh_peer_is_not_evicted(self, base_settings: object) -> None:
        discovery = PeerDiscovery(base_settings)  # type: ignore[arg-type]
        hb = Heartbeat(base_settings, discovery)  # type: ignore[arg-type]
        hb.record_peer_seen("peer_1", "/ip4/127.0.0.1/tcp/9001")
        hb._evict_stale_peers()  # just registered, should not evict
        assert discovery.peer_count == 1
