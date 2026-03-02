"""
Integration test: two libp2p nodes discover each other and exchange heartbeats.

This tests the full Phase 1 P2P stack without any mocking:
  - Real Ed25519 key pairs and libp2p hosts
  - Real TCP connections on loopback
  - Real FloodSub pubsub message delivery
  - PeerDiscovery.register_peer() called from received heartbeat

Node A starts first (bootstrap node, deterministic key so its peer ID is
predictable). Node B connects to A via bootstrap address, then both pubsubs
exchange a heartbeat on the control topic. We verify B's peer table now
contains A.
"""

from __future__ import annotations

import pytest

from swarm_tune.config.settings import NodeSettings
from swarm_tune.node.p2p.discovery import PeerDiscovery
from swarm_tune.node.p2p.gossip import CONTROL_TOPIC
from swarm_tune.node.p2p.heartbeat import Heartbeat

# Ports well outside the common test range to avoid clashes
_PORT_A = 19200
_PORT_B = 19201


def _make_settings(node_id: str, port: int, **kwargs: object) -> NodeSettings:
    defaults: dict[str, object] = {
        "node_id": node_id,
        "host": "127.0.0.1",
        "port": port,
        "bootstrap_peers": [],
        "num_rounds": 1,
        "aggregation_timeout_secs": 2.0,
        "min_peers_for_round": 1,
        "device": "cpu",
        "log_level": "WARNING",
        "log_format": "console",
    }
    defaults.update(kwargs)
    return NodeSettings(**defaults)  # type: ignore[arg-type]


@pytest.mark.anyio
@pytest.mark.integration
async def test_two_nodes_exchange_heartbeat_and_register_peer() -> None:
    """
    Node A publishes a heartbeat on the control topic.
    Node B receives it and registers A in its peer table.
    """
    import anyio
    from libp2p.tools.async_service import background_trio_service  # type: ignore[attr-defined]

    settings_a = _make_settings("node_a", _PORT_A, node_key_seed="test_bootstrap_a")
    discovery_a = PeerDiscovery(settings_a)
    await discovery_a.start()

    # Node A has a deterministic address — use it as bootstrap for B
    addr_a = discovery_a.own_multiaddr
    assert addr_a is not None, "node_a must advertise its multiaddr"

    settings_b = _make_settings("node_b", _PORT_B, bootstrap_peers=[addr_a])
    discovery_b = PeerDiscovery(settings_b)
    await discovery_b.start()

    try:
        assert discovery_a.pubsub is not None
        assert discovery_b.pubsub is not None

        async with background_trio_service(discovery_a.pubsub):
            async with background_trio_service(discovery_b.pubsub):
                # Bootstrap connection is safe now that both pubsubs are running
                await discovery_b.connect_bootstrap()

                # Allow the FloodSub stream between A and B to be negotiated
                await anyio.sleep(0.5)

                # Node A publishes its heartbeat payload
                payload = f"node_a|{addr_a}".encode()
                await discovery_a.pubsub.publish(CONTROL_TOPIC, payload)

                # Node B should receive the message within 3 seconds
                assert discovery_b.control_subscription is not None
                received: bytes | None = None
                with anyio.move_on_after(3.0):
                    msg = await discovery_b.control_subscription.get()
                    received = msg.data

                assert received is not None, "node_b timed out waiting for heartbeat from node_a"
                assert received == payload, f"unexpected payload: {received!r}"

                # Simulate what SwarmNode._heartbeat_receiver does:
                # parse the message and register A in B's peer table
                heartbeat_b = Heartbeat(settings_b, discovery_b)
                data = received.decode()
                sender_id, sender_addr = data.split("|", 1)
                heartbeat_b.record_peer_seen(sender_id, sender_addr)

                assert discovery_b.peer_count == 1, (
                    f"node_b should have 1 peer (node_a), got {discovery_b.peer_count}"
                )
                live = discovery_b.get_live_peers()
                assert live[0].peer_id == "node_a"
                assert live[0].multiaddr == addr_a
    finally:
        await discovery_b.stop()
        await discovery_a.stop()
