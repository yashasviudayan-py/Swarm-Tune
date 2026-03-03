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
            GradientExtractor().validate(grads, max_norm=1.0)

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
