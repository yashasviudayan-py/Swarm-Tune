"""
Integration tests: multiple nodes coordinating gradient averaging.

These tests wire together real subsystems (trainer + aggregator) without
a real network layer — the gossip transport is replaced by direct in-process
calls to simulate what would happen over libp2p in production.
"""

from __future__ import annotations

import pytest
import torch

from swarm_tune.node.aggregator.averaging import GradientAverager, PeerGradient
from swarm_tune.node.trainer.gradient import GradientExtractor
from swarm_tune.node.trainer.model import ModelShard
from swarm_tune.node.trainer.serializer import GradientSerializer


@pytest.mark.integration
class TestMultiNodeGradientSync:
    """Simulate 3 nodes computing local gradients and averaging them."""

    def test_all_nodes_converge_to_same_weights(self, multi_node_settings: list[object]) -> None:
        """
        After one round of gradient sync, all nodes should have identical weights.

        This validates the full pipeline:
          local backward → extract → serialize → deserialize → average → apply
        """
        shards = [ModelShard(s) for s in multi_node_settings]  # type: ignore[arg-type]
        for shard in shards:
            shard.load()

        # Sync all nodes to the same initial weights so applying the same
        # averaged gradients via AdamW produces identical final weights.
        initial_state = shards[0].model.state_dict()
        for shard in shards[1:]:
            shard.model.load_state_dict(initial_state)

        extractor = GradientExtractor()
        serializer = GradientSerializer()
        averager = GradientAverager()

        # Each node computes local gradients from random data
        all_contributions: list[PeerGradient] = []
        for i, shard in enumerate(shards):
            batch = torch.randn(4, 128)
            output = shard.forward(batch)
            loss = torch.nn.functional.mse_loss(output, batch)
            shard.backward(loss)

            raw_grads = extractor.extract(shard.model)
            # Simulate network round-trip: serialize → deserialize
            payload = serializer.serialize(raw_grads)
            recovered_grads = serializer.deserialize(payload)

            all_contributions.append(
                PeerGradient(peer_id=f"node_{i}", gradients=recovered_grads, dataset_size=100)
            )

        averaged = averager.average(all_contributions)

        # Apply the same averaged gradients to all nodes
        for shard in shards:
            shard.apply_averaged_gradients(averaged)

        # After applying the same averaged gradients, all nodes share the same weights
        # (They started from the same random init since they all called ModelShard().load())
        params_0 = dict(shards[0].model.named_parameters())
        for i, shard in enumerate(shards[1:], 1):
            for name, param in shard.model.named_parameters():
                assert torch.allclose(params_0[name].data, param.data, atol=1e-6), (
                    f"Node 0 and node {i} diverged on parameter '{name}'"
                )

    @pytest.mark.integration
    def test_serialization_preserves_gradient_values(
        self, simple_gradients: dict[str, torch.Tensor]
    ) -> None:
        """Gradients must survive serialize → deserialize without numerical drift."""
        s = GradientSerializer()
        recovered = s.deserialize(s.serialize(simple_gradients))
        for name in simple_gradients:
            assert torch.allclose(recovered[name], simple_gradients[name], atol=1e-7), (
                f"Numerical drift in parameter '{name}'"
            )
