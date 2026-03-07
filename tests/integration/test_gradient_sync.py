"""
Integration test: two libp2p nodes exchange real gradient messages.

This tests the full Phase 3 gossip stack without any mocking:
  - Real Ed25519 key pairs and libp2p hosts
  - Real TCP connections on loopback
  - Real FloodSub pubsub gradient delivery
  - GradientMessage encode/decode round-trip
  - _on_peer_gradient handler dispatched with correct sender_id, payload, dataset_size

Node A starts first (bootstrap node, deterministic key). Node B connects to A,
both subscribe to the gradient topic, A broadcasts a GradientMessage, and B
receives it with the correct decoded fields.
"""

from __future__ import annotations

import pytest
import torch

from swarm_tune.config.settings import NodeSettings
from swarm_tune.node.p2p.discovery import PeerDiscovery
from swarm_tune.node.p2p.gossip import GossipProtocol, GradientMessage
from swarm_tune.node.trainer.gradient import GradientExtractor
from swarm_tune.node.trainer.model import ModelShard
from swarm_tune.node.trainer.serializer import GradientSerializer

_PORT_A = 19300
_PORT_B = 19301


def _make_settings(node_id: str, port: int, **kwargs: object) -> NodeSettings:
    defaults: dict[str, object] = {
        "node_id": node_id,
        "host": "127.0.0.1",
        "port": port,
        "bootstrap_peers": [],
        "model_name": "mlp",  # toy MLP: no HuggingFace download in integration tests
        "num_rounds": 1,
        "aggregation_timeout_secs": 5.0,
        "min_peers_for_round": 1,
        "device": "cpu",
        "log_level": "WARNING",
        "log_format": "console",
    }
    defaults.update(kwargs)
    return NodeSettings(**defaults)  # type: ignore[arg-type]


@pytest.mark.anyio
@pytest.mark.integration
async def test_gradient_message_encode_decode_round_trip() -> None:
    """
    GradientMessage must survive encode → decode without any field drift.

    This is a pure unit check on the wire format, run here to catch issues
    before the full two-node network test.
    """
    serializer = GradientSerializer()
    model = ModelShard(_make_settings("codec_test", 19399))
    model.load()

    batch = torch.randn(4, 128)
    output = model.forward(batch)
    loss = torch.nn.functional.mse_loss(output, batch)
    model.backward(loss)

    gradients = GradientExtractor().extract(model.model)
    payload = serializer.serialize(gradients)

    original = GradientMessage(
        sender_id="node_alpha",
        round_number=7,
        payload=payload,
        dataset_size=512,
    )
    encoded = GossipProtocol._encode_message(original)
    decoded = GossipProtocol._decode_message(encoded)

    assert decoded.sender_id == original.sender_id
    assert decoded.round_number == original.round_number
    assert decoded.dataset_size == original.dataset_size
    assert decoded.payload == original.payload

    # Verify the payload is still a valid gradient dict after decode
    recovered = serializer.deserialize(decoded.payload)
    for name in gradients:
        assert torch.allclose(recovered[name], gradients[name], atol=1e-7), (
            f"Gradient round-trip drifted for '{name}'"
        )


@pytest.mark.anyio
@pytest.mark.integration
async def test_two_nodes_exchange_gradient_over_pubsub() -> None:
    """
    Node A broadcasts a GradientMessage over libp2p FloodSub.
    Node B receives it via the gossip receiver and the handler fires
    with the correct sender_id, payload, and dataset_size.
    """
    import anyio
    from libp2p.tools.async_service import background_trio_service  # type: ignore[attr-defined]

    settings_a = _make_settings("node_a", _PORT_A, node_key_seed="phase3_test_a")
    discovery_a = PeerDiscovery(settings_a)
    await discovery_a.start()

    addr_a = discovery_a.own_multiaddr
    assert addr_a is not None, "node_a must advertise its multiaddr"

    settings_b = _make_settings("node_b", _PORT_B, bootstrap_peers=[addr_a])
    discovery_b = PeerDiscovery(settings_b)
    await discovery_b.start()

    gossip_a = GossipProtocol(settings_a, discovery_a)
    gossip_b = GossipProtocol(settings_b, discovery_b)

    await gossip_a.start()
    await gossip_b.start()

    # Collect what node B receives.
    # Signature must match GradientHandler:
    #   (sender_id, authenticated_libp2p_id, payload, dataset_size, round_number)
    received_messages: list[tuple[str, str, bytes, int, int]] = []

    async def capture_handler(
        sender_id: str,
        authenticated_libp2p_id: str,
        payload: bytes,
        dataset_size: int,
        round_number: int,
    ) -> None:
        received_messages.append(
            (sender_id, authenticated_libp2p_id, payload, dataset_size, round_number)
        )

    gossip_b.on_gradient(capture_handler)

    # Build a real gradient payload
    serializer = GradientSerializer()
    model = ModelShard(settings_a)
    model.load()
    batch = torch.randn(4, 128)
    output = model.forward(batch)
    loss = torch.nn.functional.mse_loss(output, batch)
    model.backward(loss)
    gradients = GradientExtractor().extract(model.model)
    payload = serializer.serialize(gradients)

    outgoing = GradientMessage(
        sender_id="node_a",
        round_number=0,
        payload=payload,
        dataset_size=256,
    )

    try:
        assert discovery_a.pubsub is not None
        assert discovery_b.pubsub is not None

        async with background_trio_service(discovery_a.pubsub):
            async with background_trio_service(discovery_b.pubsub):
                await discovery_b.connect_bootstrap()

                async with anyio.create_task_group() as tg:
                    # Start B's receiver loop so it processes incoming messages
                    await gossip_b.run_receiver(tg)

                    # Allow FloodSub stream negotiation between A and B
                    await anyio.sleep(0.5)

                    # Node A broadcasts its gradient
                    await gossip_a.broadcast_gradient(outgoing)

                    # Wait up to 3 seconds for B to receive it
                    with anyio.move_on_after(3.0):
                        while not received_messages:
                            await anyio.sleep(0.05)

                    tg.cancel_scope.cancel()

        assert received_messages, "node_b timed out waiting for gradient from node_a"

        sender_id, _auth_id, recv_payload, recv_dataset_size, recv_round = received_messages[0]
        assert sender_id == "node_a", f"unexpected sender_id: {sender_id!r}"
        assert recv_dataset_size == 256, f"unexpected dataset_size: {recv_dataset_size}"
        assert recv_round == 0, f"unexpected round_number: {recv_round}"
        assert recv_payload == payload, "payload bytes do not match"

        # Verify the recovered payload deserializes correctly
        recovered = serializer.deserialize(recv_payload)
        for name in gradients:
            assert torch.allclose(recovered[name], gradients[name], atol=1e-7), (
                f"Gradient drifted after gossip transport for '{name}'"
            )

    finally:
        await gossip_b.stop()
        await gossip_a.stop()
        await discovery_b.stop()
        await discovery_a.stop()
