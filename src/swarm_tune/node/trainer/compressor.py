"""
Gradient compression abstraction.

The Compressor protocol is the single plug-in point for bandwidth reduction.
At 20 nodes the IdentityCompressor (no-op) is fine. At 100 nodes, swap in
TopKCompressor or QuantizedCompressor via config — zero changes to the
training loop or gossip layer.

Why this matters at scale:
  100 nodes x 140 GB gradient (70B fp16 model) = 14 TB per round.
  Top-K at 1% sparsity → ~140 GB per round. Physically possible.
  1-bit quantization → ~1.75 GB per round. Fast.

Compression is lossy. The tradeoff is bandwidth vs. convergence speed.
For most fine-tuning workloads, Top-K at 0.1-1% converges within 5-10%
more rounds than full-precision averaging.

TopKCompressor wire format (self-describing, no side-channel state)
-------------------------------------------------------------------
Each compressed tensor is encoded as a 1-D float32 tensor whose layout is:

  [0]        ndim           — number of dimensions in original tensor
  [1..ndim]  shape_dim_i   — original shape (one float per dim)
  [ndim+1]   numel         — total elements in flattened original tensor
  [ndim+2]   k             — number of (index, value) pairs that follow
  [ndim+3 .. ndim+3+k-1]  indices (float, cast to long on decode)
  [ndim+3+k .. ndim+3+2k-1] values (float)

This format is:
  • Stateless: decompress() needs no external state.
  • Type-safe: it is still a plain torch.Tensor, compatible with
    GradientSerializer (torch.save / weights_only=True) and
    GradientExtractor.validate() (called AFTER decompress()).
  • Bandwidth-honest: wire size is proportional to k, not numel.
    At k=1%, wire size ~= 2 x k x 4 bytes (indices + values).
    Contrast with the PREVIOUS BROKEN implementation that stored a
    dense zero-filled tensor (same size as the uncompressed gradient).
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
            Compressed {param_name -> tensor}. May be in the TopK sparse
            encoding (see module docstring) or identical to input (Identity).
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


def _encode_sparse(tensor: torch.Tensor, k: float) -> torch.Tensor:
    """
    Encode a tensor into the TopK sparse wire format (see module docstring).

    Returns a 1-D float32 tensor containing the metadata header followed by
    the top-k (index, value) pairs. This is a real reduction in element count:
    output has (ndim + 3 + 2k) elements vs. numel in the original tensor.
    """
    flat = tensor.flatten().float()
    n = flat.numel()
    k_count = max(1, int(n * k))

    # topk by absolute magnitude — keeps the most informationally significant elements.
    _, indices = flat.abs().topk(k_count)
    values = flat[indices]

    # Self-describing header so decompress() needs no external state.
    shape_list = list(tensor.shape)
    header = torch.tensor(
        [float(len(shape_list))] + [float(d) for d in shape_list] + [float(n), float(k_count)],
        dtype=torch.float32,
    )
    return torch.cat([header, indices.float(), values])


_MAX_DECODE_NUMEL = 500_000_000  # ~2 GB at float32; prevents OOM from malicious headers


def _decode_sparse(encoded: torch.Tensor) -> torch.Tensor:
    """
    Decode a TopK sparse-encoded tensor back to a dense float32 tensor.

    Reads the self-describing header and reconstructs the dense gradient.
    Non-top-K elements are zero (the approximation inherent in Top-K compression).

    Security: validates numel and index bounds before allocating the dense tensor.
    A malicious peer could craft a header with numel=2^31 to OOM victims (C1)
    or out-of-range indices to crash scatter_ (M3).
    """
    if encoded.numel() < 1:
        raise ValueError("TopK encoded tensor is empty")

    ndim = int(encoded[0].item())
    if ndim < 0 or ndim > 10:
        raise ValueError(f"TopK ndim={ndim} is out of valid range [0, 10]")
    if encoded.numel() < 3 + ndim:
        raise ValueError(
            f"TopK encoded tensor too short: need {3 + ndim} header elements, got {encoded.numel()}"
        )

    shape = tuple(int(encoded[1 + i].item()) for i in range(ndim))
    numel = int(encoded[1 + ndim].item())
    k_count = int(encoded[2 + ndim].item())

    # C1: Prevent OOM from malicious numel values. A legitimate 70B model's
    # largest single tensor is ~200M elements (e.g. 50257 x 4096 embedding).
    if numel <= 0 or numel > _MAX_DECODE_NUMEL:
        raise ValueError(
            f"TopK numel={numel} is out of valid range [1, {_MAX_DECODE_NUMEL}]. "
            "Possible malicious payload attempting to exhaust memory."
        )

    # Validate shape consistency: product of shape dims must equal numel.
    shape_product = 1
    for d in shape:
        if d <= 0:
            raise ValueError(f"TopK shape contains non-positive dimension: {shape}")
        shape_product *= d
    if shape_product != numel:
        raise ValueError(
            f"TopK shape {shape} (product={shape_product}) does not match numel={numel}"
        )

    if k_count < 0 or k_count > numel:
        raise ValueError(f"TopK k_count={k_count} is out of valid range [0, {numel}]")

    expected_len = 3 + ndim + 2 * k_count
    if encoded.numel() < expected_len:
        raise ValueError(
            f"TopK encoded tensor too short: need {expected_len} elements, got {encoded.numel()}"
        )

    offset = 3 + ndim
    indices = encoded[offset : offset + k_count].long()
    values = encoded[offset + k_count : offset + 2 * k_count]

    # M3: Validate index bounds before scatter to prevent crash from malicious indices.
    if k_count > 0:
        idx_max = indices.max().item()
        idx_min = indices.min().item()
        if idx_min < 0 or idx_max >= numel:
            raise ValueError(
                f"TopK indices out of bounds: min={idx_min}, max={idx_max}, numel={numel}"
            )

    dense = torch.zeros(numel, dtype=torch.float32)
    dense.scatter_(0, indices, values)
    return dense.reshape(shape)


class TopKCompressor:
    """
    Top-K sparsification: keep only the K% largest-magnitude gradient elements.

    Wire format is self-describing (see module docstring). Unlike the previous
    implementation which stored a dense zero-filled tensor (no actual bandwidth
    savings), this version encodes only the (index, value) pairs, achieving
    real wire-size reduction proportional to k.

    At k=0.01 (1%):
      - Wire elements per tensor: 2k + ndim + 3  (indices + values + header)
      - vs. numel elements for the uncompressed gradient
      - For a 768x768 weight matrix (numel=589,824): ~11,800 elements vs 589,824
      - Compression ratio: ~50x

    Convergence: slightly slower than full-precision FedAvg but model quality
    is nearly identical for fine-tuning workloads (Aji & Heafield 2017).

    Reference: Aji & Heafield, "Sparse Communication for Distributed Gradient
    Descent" (2017). https://arxiv.org/abs/1704.05021
    """

    def __init__(self, k: float = 0.01) -> None:
        if not 0.0 < k <= 1.0:
            raise ValueError(f"k must be in (0, 1], got {k}")
        self.k = k

    def compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        compressed: dict[str, torch.Tensor] = {}
        total_in = 0
        total_out = 0
        for name, tensor in gradients.items():
            encoded = _encode_sparse(tensor, self.k)
            compressed[name] = encoded
            total_in += tensor.numel()
            total_out += encoded.numel()
        ratio = total_in / max(total_out, 1)
        log.debug(
            "top-k compression applied",
            k=self.k,
            params=len(gradients),
            elements_in=total_in,
            elements_out=total_out,
            compression_ratio=f"{ratio:.1f}x",
        )
        return compressed

    def decompress(self, gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        dense: dict[str, torch.Tensor] = {}
        for name, encoded in gradients.items():
            dense[name] = _decode_sparse(encoded)
        log.debug("top-k decompression applied", params=len(gradients))
        return dense
