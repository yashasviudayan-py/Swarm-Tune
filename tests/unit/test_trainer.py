"""Unit tests for the trainer subsystem: model, gradient extraction, serialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from swarm_tune.node.trainer.data import DataShardLoader
from swarm_tune.node.trainer.gradient import GradientExtractor
from swarm_tune.node.trainer.model import ModelShard
from swarm_tune.node.trainer.serializer import GradientSerializer


class TestGradientExtractor:
    """Tests for gradient extraction after backward pass."""

    @pytest.fixture()
    def trained_model(self) -> nn.Module:
        """A model that has already had backward called."""
        model = nn.Linear(4, 2)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        return model

    @pytest.mark.unit
    def test_extracts_all_parameters(self, trained_model: nn.Module) -> None:
        grads = GradientExtractor().extract(trained_model)
        assert "weight" in grads
        assert "bias" in grads

    @pytest.mark.unit
    def test_gradients_are_on_cpu(self, trained_model: nn.Module) -> None:
        grads = GradientExtractor().extract(trained_model)
        for tensor in grads.values():
            assert tensor.device.type == "cpu"

    @pytest.mark.unit
    def test_raises_if_no_gradients(self) -> None:
        model = nn.Linear(4, 2)  # backward not called
        with pytest.raises(ValueError, match="No gradients found"):
            GradientExtractor().extract(model)

    @pytest.mark.unit
    def test_validation_rejects_nan(self) -> None:
        grads = {"w": torch.tensor([float("nan"), 1.0])}
        with pytest.raises(ValueError, match="NaN or Inf"):
            GradientExtractor().validate(grads)

    @pytest.mark.unit
    def test_validation_rejects_large_norm(self) -> None:
        grads = {"w": torch.full((100,), 1e6)}
        with pytest.raises(ValueError, match="exceeds threshold"):
            GradientExtractor().validate(grads, max_norm_rms=1.0)

    @pytest.mark.unit
    def test_validation_passes_normal_gradients(
        self, simple_gradients: dict[str, torch.Tensor]
    ) -> None:
        result = GradientExtractor().validate(simple_gradients)
        assert result is simple_gradients


class TestGradientSerializer:
    """Tests for binary serialization round-trips."""

    @pytest.mark.unit
    def test_roundtrip(self, simple_gradients: dict[str, torch.Tensor]) -> None:
        s = GradientSerializer()
        payload = s.serialize(simple_gradients)
        recovered = s.deserialize(payload)

        assert set(recovered.keys()) == set(simple_gradients.keys())
        for name in simple_gradients:
            assert torch.allclose(recovered[name], simple_gradients[name])

    @pytest.mark.unit
    def test_payload_is_bytes(self, simple_gradients: dict[str, torch.Tensor]) -> None:
        payload = GradientSerializer().serialize(simple_gradients)
        assert isinstance(payload, bytes)

    @pytest.mark.unit
    def test_bad_magic_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid magic bytes"):
            GradientSerializer().deserialize(b"BAAD" + b"\x00" * 20)

    @pytest.mark.unit
    def test_truncated_payload_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            GradientSerializer().deserialize(b"\x00\x01")


class TestDataShardLoader:
    """Tests for DataShardLoader: shard file loading and mini-batch sampling."""

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

    @pytest.mark.unit
    def test_loads_shard(self, shard_path: Path) -> None:
        loader = DataShardLoader(shard_path)
        loader.load()
        assert loader.dataset_size == 64

    @pytest.mark.unit
    def test_get_batch_shape(self, shard_path: Path) -> None:
        loader = DataShardLoader(shard_path)
        loader.load()
        inputs, targets = loader.get_batch(8)
        assert inputs.shape == (8, 128)
        assert targets.shape == (8, 128)

    @pytest.mark.unit
    def test_batch_on_cpu(self, shard_path: Path) -> None:
        loader = DataShardLoader(shard_path)
        loader.load()
        inputs, targets = loader.get_batch(4)
        assert inputs.device.type == "cpu"
        assert targets.device.type == "cpu"

    @pytest.mark.unit
    def test_batch_size_capped_at_dataset_size(self, shard_path: Path) -> None:
        loader = DataShardLoader(shard_path)
        loader.load()
        inputs, _targets = loader.get_batch(9999)
        assert inputs.shape[0] == 64  # shard only has 64 samples

    @pytest.mark.unit
    def test_raises_before_load(self, shard_path: Path) -> None:
        loader = DataShardLoader(shard_path)
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            loader.get_batch(4)

    @pytest.mark.unit
    def test_dataset_size_raises_before_load(self, shard_path: Path) -> None:
        loader = DataShardLoader(shard_path)
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            _ = loader.dataset_size

    @pytest.mark.unit
    def test_missing_shard_raises(self, tmp_path: Path) -> None:
        loader = DataShardLoader(tmp_path / "nonexistent.pt")
        with pytest.raises(FileNotFoundError):
            loader.load()

    @pytest.mark.unit
    def test_missing_keys_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_shard.pt"
        torch.save({"wrong_key": torch.randn(10, 4)}, path)
        loader = DataShardLoader(path)
        with pytest.raises(KeyError, match="missing"):
            loader.load()


class TestModelShard:
    """Tests for model loading and gradient application."""

    @pytest.mark.unit
    def test_load_creates_model(self, base_settings: object) -> None:
        shard = ModelShard(base_settings)  # type: ignore[arg-type]
        shard.load()
        assert shard.model is not None

    @pytest.mark.unit
    def test_forward_returns_tensor(self, base_settings: object) -> None:
        shard = ModelShard(base_settings)  # type: ignore[arg-type]
        shard.load()
        output = shard.forward(torch.randn(4, 128))
        assert isinstance(output, torch.Tensor)

    @pytest.mark.unit
    def test_raises_before_load(self, base_settings: object) -> None:
        shard = ModelShard(base_settings)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="not loaded"):
            shard.forward(torch.randn(1, 128))


class TestTopKCompressor:
    """Tests for TopKCompressor — bandwidth-reducing gradient sparsification."""

    @pytest.fixture()
    def gradients(self) -> dict[str, torch.Tensor]:
        return {
            "0.weight": torch.randn(64, 32),
            "0.bias": torch.randn(64),
            "2.weight": torch.randn(32, 64),
        }

    @pytest.mark.unit
    def test_compress_reduces_element_count(self, gradients: dict[str, torch.Tensor]) -> None:
        from swarm_tune.node.trainer.compressor import TopKCompressor

        c = TopKCompressor(k=0.1)
        compressed = c.compress(gradients)
        for name, orig in gradients.items():
            assert compressed[name].numel() < orig.numel(), (
                f"Compressed tensor for '{name}' should have fewer elements than original"
            )

    @pytest.mark.unit
    def test_roundtrip_preserves_shape(self, gradients: dict[str, torch.Tensor]) -> None:
        from swarm_tune.node.trainer.compressor import TopKCompressor

        c = TopKCompressor(k=0.1)
        recovered = c.decompress(c.compress(gradients))
        for name, orig in gradients.items():
            assert recovered[name].shape == orig.shape

    @pytest.mark.unit
    def test_roundtrip_top_elements_preserved(self) -> None:
        """The top-K elements (by magnitude) should be losslessly recovered."""
        from swarm_tune.node.trainer.compressor import TopKCompressor

        # A 1-D tensor with a single dominant element
        g = {"w": torch.zeros(100)}
        g["w"][42] = 99.0  # single large value
        c = TopKCompressor(k=0.1)
        recovered = c.decompress(c.compress(g))
        # The dominant element should be recovered exactly
        assert recovered["w"][42].item() == pytest.approx(99.0)

    @pytest.mark.unit
    def test_compressed_is_still_float32(self, gradients: dict[str, torch.Tensor]) -> None:
        from swarm_tune.node.trainer.compressor import TopKCompressor

        c = TopKCompressor(k=0.05)
        compressed = c.compress(gradients)
        for name, t in compressed.items():
            assert t.dtype == torch.float32, f"'{name}' should be float32 after compression"

    @pytest.mark.unit
    def test_invalid_k_raises(self) -> None:
        from swarm_tune.node.trainer.compressor import TopKCompressor

        with pytest.raises(ValueError, match="k must be in"):
            TopKCompressor(k=0.0)
        with pytest.raises(ValueError, match="k must be in"):
            TopKCompressor(k=1.5)

    @pytest.mark.unit
    def test_k_equals_1_is_near_lossless(self, gradients: dict[str, torch.Tensor]) -> None:
        """k=1.0 keeps all elements — roundtrip should be nearly exact."""
        from swarm_tune.node.trainer.compressor import TopKCompressor

        c = TopKCompressor(k=1.0)
        recovered = c.decompress(c.compress(gradients))
        for name, orig in gradients.items():
            assert torch.allclose(recovered[name].float(), orig.float(), atol=1e-5), (
                f"k=1.0 roundtrip should be lossless for '{name}'"
            )

    @pytest.mark.unit
    def test_compressor_protocol_satisfied(self) -> None:
        """TopKCompressor and IdentityCompressor both satisfy the Compressor protocol."""
        from swarm_tune.node.trainer.compressor import (
            Compressor,
            IdentityCompressor,
            TopKCompressor,
        )

        assert isinstance(TopKCompressor(), Compressor)
        assert isinstance(IdentityCompressor(), Compressor)
