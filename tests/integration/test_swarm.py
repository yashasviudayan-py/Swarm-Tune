"""
Integration tests: multiple nodes coordinating gradient averaging.

These tests wire together real subsystems (trainer + aggregator) without
a real network layer — the gossip transport is replaced by direct in-process
calls to simulate what would happen over libp2p in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from swarm_tune.node.aggregator.averaging import GradientAverager, PeerGradient
from swarm_tune.node.trainer.compressor import IdentityCompressor
from swarm_tune.node.trainer.data import DataShardLoader
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


@pytest.mark.integration
class TestSingleNodeGradientPipeline:
    """
    Phase 2 deliverable: a single node loads a real shard, computes a gradient,
    compresses, serializes, deserializes, decompresses, and validates it —
    end-to-end, with no peer network required.
    """

    @pytest.fixture()
    def shard_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "shard_0.pt"
        torch.save(
            {
                "inputs": torch.randn(64, 128),
                "targets": torch.randn(64, 128),
                "shard_index": 0,
                "num_shards": 1,
            },
            path,
        )
        return path

    def test_full_gradient_pipeline_with_real_shard(
        self, shard_path: Path, base_settings: object
    ) -> None:
        """
        Forward → backward → extract → compress → serialize → deserialize →
        decompress → validate.  All steps must complete without error and the
        recovered gradients must be numerically identical to the originals
        (IdentityCompressor is lossless).
        """
        data = DataShardLoader(shard_path)
        data.load()
        assert data.dataset_size == 64

        model = ModelShard(base_settings)  # type: ignore[arg-type]
        model.load()

        extractor = GradientExtractor()
        compressor = IdentityCompressor()
        serializer = GradientSerializer()

        for _ in range(3):
            inputs, targets = data.get_batch(8)
            output = model.forward(inputs)
            loss = torch.nn.functional.mse_loss(output, targets.to(output.device))
            assert torch.isfinite(loss), "loss must be finite"

            model.backward(loss)

            # Extract → compress → serialize
            gradients = extractor.extract(model.model)
            assert len(gradients) > 0
            compressed = compressor.compress(gradients)
            payload = serializer.serialize(compressed)
            assert isinstance(payload, bytes) and len(payload) > 0

            # Deserialize → decompress → validate (mimics the receiving side)
            recovered_compressed = serializer.deserialize(payload)
            recovered = compressor.decompress(recovered_compressed)
            validated = extractor.validate(recovered)
            assert set(validated.keys()) == set(gradients.keys())

            for name in gradients:
                assert torch.allclose(validated[name], gradients[name], atol=1e-7), (
                    f"Gradient round-trip drifted for parameter '{name}'"
                )

            # Apply gradient so model actually trains (loss should generally decrease)
            model.apply_averaged_gradients(validated)

    def test_loss_decreases_over_training_rounds(
        self, shard_path: Path, base_settings: object
    ) -> None:
        """
        After several rounds of training on fixed data, the model loss should
        decrease — this confirms gradient extraction + apply is working correctly.
        """
        data = DataShardLoader(shard_path)
        data.load()

        model = ModelShard(base_settings)  # type: ignore[arg-type]
        model.load()

        extractor = GradientExtractor()

        # Fix a single batch to measure convergence on the same data
        inputs, targets = data.get_batch(32)

        losses: list[float] = []
        for _ in range(10):
            output = model.forward(inputs)
            loss = torch.nn.functional.mse_loss(output, targets.to(output.device))
            losses.append(loss.item())
            model.backward(loss)
            gradients = extractor.extract(model.model)
            model.apply_averaged_gradients(gradients)

        # Loss after 10 rounds should be strictly lower than after round 1
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )
