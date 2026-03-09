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


class TestBanList:
    """Tests for BanList — Sybil resistance peer banning (Phase 5)."""

    @pytest.mark.unit
    def test_new_peer_is_not_banned(self) -> None:
        from swarm_tune.node.p2p.peer_selector import BanList

        bl = BanList()
        assert not bl.is_banned("peer_1")

    @pytest.mark.unit
    def test_ban_not_triggered_before_min_rounds(self) -> None:
        from swarm_tune.node.p2p.peer_selector import BanList

        bl = BanList()
        # Record 4 rejections (need >= 5 rounds before ban kicks in)
        for _ in range(4):
            bl.record_round("peer_1", rejected=True)
        banned = bl.check_and_ban("peer_1", threshold=0.5)
        assert not banned
        assert not bl.is_banned("peer_1")

    @pytest.mark.unit
    def test_ban_triggered_above_threshold(self) -> None:
        from swarm_tune.node.p2p.peer_selector import BanList

        bl = BanList(ban_duration_secs=600.0)
        # 6 rounds, all rejected → rate = 1.0 > threshold 0.5
        for _ in range(6):
            bl.record_round("peer_1", rejected=True)
        banned = bl.check_and_ban("peer_1", threshold=0.5)
        assert banned
        assert bl.is_banned("peer_1")

    @pytest.mark.unit
    def test_ban_not_triggered_below_threshold(self) -> None:
        from swarm_tune.node.p2p.peer_selector import BanList

        bl = BanList()
        # 10 rounds, 2 rejected → rate = 0.2 < threshold 0.5
        for i in range(10):
            bl.record_round("peer_1", rejected=(i < 2))
        banned = bl.check_and_ban("peer_1", threshold=0.5)
        assert not banned
        assert not bl.is_banned("peer_1")

    @pytest.mark.unit
    def test_ban_expires(self) -> None:
        from swarm_tune.node.p2p.peer_selector import BanList

        bl = BanList(ban_duration_secs=0.001)  # 1 ms ban
        for _ in range(6):
            bl.record_round("peer_1", rejected=True)
        bl.check_and_ban("peer_1", threshold=0.5)
        assert bl.is_banned("peer_1")

        time.sleep(0.01)  # wait for ban to expire
        assert not bl.is_banned("peer_1")

    @pytest.mark.unit
    def test_banned_peers_returns_active_set(self) -> None:
        from swarm_tune.node.p2p.peer_selector import BanList

        bl = BanList(ban_duration_secs=600.0)
        for _ in range(6):
            bl.record_round("peer_1", rejected=True)
            bl.record_round("peer_2", rejected=True)
        bl.check_and_ban("peer_1", threshold=0.5)
        bl.check_and_ban("peer_2", threshold=0.5)
        assert bl.banned_peers() == {"peer_1", "peer_2"}

    @pytest.mark.unit
    def test_banned_peers_excludes_expired(self) -> None:
        from swarm_tune.node.p2p.peer_selector import BanList

        bl = BanList(ban_duration_secs=0.001)
        for _ in range(6):
            bl.record_round("peer_1", rejected=True)
        bl.check_and_ban("peer_1", threshold=0.5)
        time.sleep(0.01)
        assert bl.banned_peers() == set()

    @pytest.mark.unit
    def test_already_banned_peer_not_double_banned(self) -> None:
        from swarm_tune.node.p2p.peer_selector import BanList

        bl = BanList(ban_duration_secs=600.0)
        for _ in range(6):
            bl.record_round("peer_1", rejected=True)
        first = bl.check_and_ban("peer_1", threshold=0.5)
        second = bl.check_and_ban("peer_1", threshold=0.5)
        assert first is True
        assert second is False  # already banned, not re-banned
