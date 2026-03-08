"""
Gradient extraction from a PyTorch model after a backward pass.

After loss.backward(), every parameter's .grad tensor is populated.
This module extracts those tensors into a plain dict so they can be
serialized and sent over the network.

This is the core of why we do NOT use PyTorch DDP: DDP calls
all-reduce internally and assumes a low-latency bus. We extract
gradients manually so we can apply custom aggregation logic
(timeout, partial averaging, anomaly filtering) before any update.
"""

from __future__ import annotations

import math
import warnings

import structlog
import torch
import torch.nn as nn

log: structlog.BoundLogger = structlog.get_logger(__name__)


class GradientExtractor:
    """
    Extracts and validates gradients from a model after backward().

    Returns a dict mapping parameter name -> gradient tensor.
    Parameters with None gradients (frozen layers) are excluded.
    """

    def extract(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """
        Extract all non-None gradients from a model.

        Args:
            model: a PyTorch module after loss.backward() has been called.

        Returns:
            dict of {param_name: grad_tensor} — all on CPU for serialization.

        Raises:
            ValueError: if the model has no gradients (backward not called yet).
        """
        gradients: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            # Move to CPU for serialization — device-agnostic transport
            gradients[name] = param.grad.detach().cpu().clone()

        if not gradients:
            raise ValueError(
                "No gradients found. Ensure loss.backward() was called before extraction."
            )

        log.debug(
            "gradients extracted",
            num_params=len(gradients),
            total_elements=sum(g.numel() for g in gradients.values()),
        )
        return gradients

    def validate(
        self,
        gradients: dict[str, torch.Tensor],
        max_norm_rms: float = 10.0,
        # Legacy alias so existing call-sites using max_norm= still work.
        max_norm: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Sanity-check received gradients before aggregation.

        Uses per-element RMS norm (tensor.norm() / sqrt(numel)) rather than
        the absolute L2 norm. This makes the threshold model-size-agnostic:
        a 70B-parameter embedding table and a 2-layer MLP both have an RMS
        norm around 1.0 when training normally, regardless of absolute tensor size.

        The old absolute L2 norm (1e4) incorrectly rejected legitimate gradients
        from large tensors like LLaMA's token embedding (50257 x 4096) whose
        healthy L2 norm can naturally exceed 1e4 early in fine-tuning.

        Args:
            gradients:    dict of {param_name: tensor} from a peer.
            max_norm_rms: per-element RMS norm threshold. Default 10.0 gives
                          ~10-sigma headroom above a standard-normal initialised
                          model's expected gradient RMS of ~1.0.
            max_norm:     legacy positional alias for max_norm_rms (backwards
                          compat). If both are given, max_norm takes precedence.

        Returns:
            The same dict if valid.

        Raises:
            ValueError: if any gradient fails validation.
        """
        # Backwards-compatibility: old call-sites pass max_norm=<float>.
        # Emit a deprecation warning — max_norm was an absolute L2 norm (model-size
        # dependent) whereas max_norm_rms is per-element RMS (model-agnostic).
        # New code should always use max_norm_rms.
        if max_norm is not None:
            warnings.warn(
                "max_norm= is deprecated; use max_norm_rms= instead. "
                "max_norm was an absolute L2 norm (model-size dependent). "
                "max_norm_rms is per-element RMS (model-agnostic, default 10.0).",
                DeprecationWarning,
                stacklevel=2,
            )
        threshold = max_norm if max_norm is not None else max_norm_rms

        for name, tensor in gradients.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError(f"Gradient '{name}' contains NaN or Inf values.")

            # Per-element RMS norm: scale-free and model-architecture-agnostic.
            numel = tensor.numel()
            if numel == 0:
                continue
            rms_norm = tensor.norm().item() / math.sqrt(numel)
            if rms_norm > threshold:
                raise ValueError(
                    f"Gradient '{name}' per-element RMS norm {rms_norm:.4f} "
                    f"exceeds threshold {threshold}. "
                    "Possible gradient poisoning or exploding gradients — rejecting."
                )
        return gradients
