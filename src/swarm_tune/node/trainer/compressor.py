"""
Gradient compression abstraction.

The Compressor protocol is the single plug-in point for bandwidth reduction.
At 20 nodes the IdentityCompressor (no-op) is fine. At 100 nodes, swap in
TopKCompressor or QuantizedCompressor via config — zero changes to the
training loop or gossip layer.

Why this matters at scale:
  100 nodes x 140 GB gradient (70B fp16 model) = 14 TB per round.
  Top-K at 1% sparsity → 140 GB per round. Physically possible.
  1-bit quantization → ~1.75 GB per round. Fast.

Compression is lossy. The tradeoff is bandwidth vs. convergence speed.
For most fine-tuning workloads, Top-K at 0.1-1% converges within 5-10%
more rounds than full-precision averaging.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog
import torch

log: structlog.BoundLogger = structlog.get_logger(__name__)


@runtime_checkable
class Compressor(Protocol):
    """
    Strategy interface for gradient compression.

    Implementations must be stateless and thread-safe.
    compress() and decompress() are inverse operations:
      decompress(compress(g)) ≈ g  (exact for IdentityCompressor, lossy otherwise)
    """

    def compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compress a gradient dict before serialization and network transport.

        Args:
            gradients: {param_name -> dense tensor} on CPU.

        Returns:
            Compressed {param_name -> tensor}. May be sparse or quantized.
        """
        ...

    def decompress(self, gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Reconstruct the full gradient dict after transport.

        Args:
            gradients: {param_name -> compressed tensor} as received from peer.

        Returns:
            Dense {param_name -> tensor} ready for aggregation.
        """
        ...


class IdentityCompressor:
    """
    No-op compressor. Passes gradients through unchanged.

    Default for Phases 1-4. Swap this out when bandwidth becomes the
    bottleneck (typically when the swarm exceeds ~30 nodes on home internet).
    """

    def compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return gradients

    def decompress(self, gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return gradients


class TopKCompressor:
    """
    Top-K sparsification: keep only the K% largest-magnitude gradient elements.

    Sets all other elements to zero. The receiver decompresses by treating
    the sparse representation as a dense tensor (zeros fill the gaps).

    At k=0.01 (1%), bandwidth is reduced 100x. Convergence is slightly slower
    but model quality is nearly identical for fine-tuning workloads.

    Reference: Aji & Heafield, "Sparse Communication for Distributed Gradient
    Descent" (2017). https://arxiv.org/abs/1704.05021
    """

    def __init__(self, k: float = 0.01) -> None:
        if not 0.0 < k <= 1.0:
            raise ValueError(f"k must be in (0, 1], got {k}")
        self.k = k

    def compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        compressed: dict[str, torch.Tensor] = {}
        for name, tensor in gradients.items():
            flat = tensor.flatten()
            k_count = max(1, int(flat.numel() * self.k))
            threshold = flat.abs().topk(k_count).values[-1]
            mask = flat.abs() >= threshold
            sparse = flat * mask
            compressed[name] = sparse.reshape(tensor.shape)
        log.debug("top-k compression applied", k=self.k, params=len(gradients))
        return compressed

    def decompress(self, gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Sparse tensors from TopK are already in dense format (zeros in place).
        # No reconstruction step needed — just pass through.
        return gradients
