"""Regression tests for audit findings C1, H1, H3, M2, M3, M4, M5.

Each test class maps to a specific audit finding. Tests verify the fix
and prevent regression.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from swarm_tune.node.aggregator.averaging import GradientAverager, PeerGradient
from swarm_tune.node.p2p.peer_selector import BanList
from swarm_tune.node.trainer.compressor import TopKCompressor, _decode_sparse, _encode_sparse

# ============================================================
# C1: TopK decode numel bounds check — prevents remote OOM
# ============================================================


class TestC1TopKNukelBoundsCheck:
    """A malicious peer can craft a TopK header with huge numel to OOM victims."""

    @pytest.mark.unit
    def test_decode_rejects_huge_numel(self) -> None:
        """numel > _MAX_DECODE_NUMEL should raise ValueError, not allocate."""
        # Craft a minimal valid-looking encoded tensor with absurd numel.
        # ndim=1, shape=[1], numel=2_000_000_000, k_count=0
        encoded = torch.tensor([1.0, 1.0, 2_000_000_000.0, 0.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="out of valid range"):
            _decode_sparse(encoded)

    @pytest.mark.unit
    def test_decode_rejects_negative_numel(self) -> None:
        encoded = torch.tensor([1.0, 5.0, -1.0, 0.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="out of valid range"):
            _decode_sparse(encoded)

    @pytest.mark.unit
    def test_decode_rejects_zero_numel(self) -> None:
        encoded = torch.tensor([1.0, 5.0, 0.0, 0.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="out of valid range"):
            _decode_sparse(encoded)

    @pytest.mark.unit
    def test_decode_rejects_shape_mismatch(self) -> None:
        """Shape product must match numel."""
        # ndim=2, shape=[3,4], numel=10 (should be 12), k_count=0
        encoded = torch.tensor([2.0, 3.0, 4.0, 10.0, 0.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="does not match numel"):
            _decode_sparse(encoded)

    @pytest.mark.unit
    def test_decode_rejects_negative_ndim(self) -> None:
        encoded = torch.tensor([-1.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="ndim="):
            _decode_sparse(encoded)

    @pytest.mark.unit
    def test_decode_rejects_huge_ndim(self) -> None:
        encoded = torch.tensor([100.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="ndim="):
            _decode_sparse(encoded)

    @pytest.mark.unit
    def test_decode_rejects_empty_tensor(self) -> None:
        encoded = torch.tensor([], dtype=torch.float32)
        with pytest.raises(ValueError, match="empty"):
            _decode_sparse(encoded)

    @pytest.mark.unit
    def test_valid_decode_still_works(self) -> None:
        """Legitimate TopK encode→decode roundtrip must still work."""
        original = torch.randn(10, 20)
        encoded = _encode_sparse(original, k=0.5)
        decoded = _decode_sparse(encoded)
        assert decoded.shape == original.shape
        # TopK is lossy — only check that at least some values match.
        assert decoded.any()


# ============================================================
# M3: TopK index bounds validation — prevents scatter_ crash
# ============================================================


class TestM3TopKIndexBounds:
    """Out-of-range indices in TopK should be caught before scatter_."""

    @pytest.mark.unit
    def test_decode_rejects_out_of_range_index(self) -> None:
        """An index >= numel should raise ValueError."""
        # ndim=1, shape=[5], numel=5, k_count=1, index=999, value=1.0
        encoded = torch.tensor([1.0, 5.0, 5.0, 1.0, 999.0, 1.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="indices out of bounds"):
            _decode_sparse(encoded)

    @pytest.mark.unit
    def test_decode_rejects_negative_index(self) -> None:
        # ndim=1, shape=[5], numel=5, k_count=1, index=-1, value=1.0
        encoded = torch.tensor([1.0, 5.0, 5.0, 1.0, -1.0, 1.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="indices out of bounds"):
            _decode_sparse(encoded)

    @pytest.mark.unit
    def test_valid_indices_pass(self) -> None:
        # ndim=1, shape=[5], numel=5, k_count=2, indices=[0, 4], values=[1.0, 2.0]
        encoded = torch.tensor([1.0, 5.0, 5.0, 2.0, 0.0, 4.0, 1.0, 2.0], dtype=torch.float32)
        result = _decode_sparse(encoded)
        assert result.shape == (5,)
        assert result[0].item() == pytest.approx(1.0)
        assert result[4].item() == pytest.approx(2.0)


# ============================================================
# H1: BanList counter reset on ban expiry
# ============================================================


class TestH1BanListCounterReset:
    """After a ban expires, counters should be cleared for a clean slate."""

    @pytest.mark.unit
    def test_counters_reset_on_ban_expiry(self) -> None:
        bl = BanList(ban_duration_secs=0.1)
        peer = "peer_a"

        # Accumulate rejections until banned.
        for _ in range(6):
            bl.record_round(peer, rejected=True)
        assert bl.check_and_ban(peer, threshold=0.5)
        assert bl.is_banned(peer)

        # Wait for ban to expire.
        time.sleep(0.15)
        assert not bl.is_banned(peer)

        # Counters should be gone — peer starts fresh.
        # A single new rejection should NOT trigger an immediate re-ban.
        bl.record_round(peer, rejected=True)
        assert not bl.check_and_ban(peer, threshold=0.5)  # < 5 rounds

    @pytest.mark.unit
    def test_ban_expiry_resets_rejection_count(self) -> None:
        bl = BanList(ban_duration_secs=0.05)
        peer = "peer_b"

        for _ in range(10):
            bl.record_round(peer, rejected=True)
        bl.check_and_ban(peer, threshold=0.5)

        time.sleep(0.1)
        assert not bl.is_banned(peer)

        # After reset, internal dicts should not contain the peer.
        assert peer not in bl._rejections
        assert peer not in bl._rounds


# ============================================================
# H3: Gradient averaging preserves original dtype
# ============================================================


class TestH3DtypePreservation:
    """FedAvg should return tensors in the original dtype, not always float32."""

    @pytest.mark.unit
    def test_float16_gradients_preserved(self) -> None:
        grads_a = {"w": torch.randn(4, 4, dtype=torch.float16)}
        grads_b = {"w": torch.randn(4, 4, dtype=torch.float16)}

        contributions = [
            PeerGradient("a", grads_a, 100),
            PeerGradient("b", grads_b, 100),
        ]
        result = GradientAverager().average(contributions)
        assert result["w"].dtype == torch.float16

    @pytest.mark.unit
    def test_float32_gradients_unchanged(self) -> None:
        grads_a = {"w": torch.randn(4, 4, dtype=torch.float32)}
        grads_b = {"w": torch.randn(4, 4, dtype=torch.float32)}

        contributions = [
            PeerGradient("a", grads_a, 100),
            PeerGradient("b", grads_b, 100),
        ]
        result = GradientAverager().average(contributions)
        assert result["w"].dtype == torch.float32

    @pytest.mark.unit
    def test_bfloat16_gradients_preserved(self) -> None:
        grads_a = {"w": torch.randn(4, 4, dtype=torch.bfloat16)}
        grads_b = {"w": torch.randn(4, 4, dtype=torch.bfloat16)}

        contributions = [
            PeerGradient("a", grads_a, 100),
            PeerGradient("b", grads_b, 100),
        ]
        result = GradientAverager().average(contributions)
        assert result["w"].dtype == torch.bfloat16


# ============================================================
# M4: Checkpoint rotation
# ============================================================


class TestM4CheckpointRotation:
    """Old periodic checkpoints should be deleted after keep_n_checkpoints."""

    @pytest.mark.unit
    def test_cleanup_deletes_old_checkpoints(self, tmp_path: Path) -> None:
        """Create 5 checkpoints, keep_n=2, expect oldest 3 deleted."""
        from swarm_tune.config.settings import NodeSettings
        from swarm_tune.node.main import SwarmNode

        settings = NodeSettings(
            node_id="test_node",
            host="127.0.0.1",
            port=19000,
            model_name="mlp",
            checkpoint_dir=tmp_path,
            keep_n_checkpoints=2,
            device="cpu",
            log_level="WARNING",
        )

        node = SwarmNode(settings)

        # Create fake checkpoint files with increasing mtime.
        for i in range(5):
            ckpt = tmp_path / f"test_node_round_{(i + 1) * 10}.pt"
            ckpt.write_bytes(b"fake")
            # Ensure distinct mtimes.
            import os

            os.utime(ckpt, (i, i))

        node._cleanup_old_checkpoints()

        remaining = sorted(tmp_path.glob("test_node_round_*.pt"))
        assert len(remaining) == 2
        # The two newest should remain.
        names = {p.name for p in remaining}
        assert "test_node_round_40.pt" in names
        assert "test_node_round_50.pt" in names

    @pytest.mark.unit
    def test_cleanup_noop_when_under_limit(self, tmp_path: Path) -> None:
        """When fewer checkpoints exist than keep_n, nothing is deleted."""
        from swarm_tune.config.settings import NodeSettings
        from swarm_tune.node.main import SwarmNode

        settings = NodeSettings(
            node_id="test_node",
            host="127.0.0.1",
            port=19000,
            model_name="mlp",
            checkpoint_dir=tmp_path,
            keep_n_checkpoints=5,
            device="cpu",
            log_level="WARNING",
        )

        node = SwarmNode(settings)

        for i in range(3):
            ckpt = tmp_path / f"test_node_round_{(i + 1) * 10}.pt"
            ckpt.write_bytes(b"fake")

        node._cleanup_old_checkpoints()

        remaining = list(tmp_path.glob("test_node_round_*.pt"))
        assert len(remaining) == 3

    @pytest.mark.unit
    def test_final_checkpoint_not_affected(self, tmp_path: Path) -> None:
        """Final checkpoint (_final.pt) should never be deleted by rotation."""
        from swarm_tune.config.settings import NodeSettings
        from swarm_tune.node.main import SwarmNode

        settings = NodeSettings(
            node_id="test_node",
            host="127.0.0.1",
            port=19000,
            model_name="mlp",
            checkpoint_dir=tmp_path,
            keep_n_checkpoints=1,
            device="cpu",
            log_level="WARNING",
        )

        node = SwarmNode(settings)

        # Create a final checkpoint and some round checkpoints.
        (tmp_path / "test_node_final.pt").write_bytes(b"final")
        for i in range(3):
            ckpt = tmp_path / f"test_node_round_{(i + 1) * 10}.pt"
            ckpt.write_bytes(b"fake")

        node._cleanup_old_checkpoints()

        # Final checkpoint must survive.
        assert (tmp_path / "test_node_final.pt").exists()
        # Only 1 round checkpoint should remain.
        remaining = list(tmp_path.glob("test_node_round_*.pt"))
        assert len(remaining) == 1


# ============================================================
# TopKCompressor end-to-end with bounds checks active
# ============================================================


class TestTopKCompressorRoundtrip:
    """Ensure TopKCompressor still works correctly after C1/M3 hardening."""

    @pytest.mark.unit
    def test_compress_decompress_roundtrip(self) -> None:
        compressor = TopKCompressor(k=0.1)
        grads = {
            "weight": torch.randn(32, 64),
            "bias": torch.randn(32),
        }
        compressed = compressor.compress(grads)
        decompressed = compressor.decompress(compressed)

        assert set(decompressed.keys()) == set(grads.keys())
        for name in grads:
            assert decompressed[name].shape == grads[name].shape
            # TopK is lossy but top elements should be preserved.
            assert decompressed[name].any()

    @pytest.mark.unit
    def test_identity_compressor_unaffected(self) -> None:
        from swarm_tune.node.trainer.compressor import IdentityCompressor

        compressor = IdentityCompressor()
        grads = {"w": torch.randn(8, 8)}
        result = compressor.decompress(compressor.compress(grads))
        assert torch.equal(result["w"], grads["w"])
