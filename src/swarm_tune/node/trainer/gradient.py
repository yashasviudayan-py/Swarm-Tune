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
        max_norm: float = 1e4,
    ) -> dict[str, torch.Tensor]:
        """
        Basic sanity check on received gradients before aggregation.

        Rejects gradients with NaN/Inf values or implausibly large norms.
        This is a first line of defence against gradient poisoning.

        Args:
            gradients: dict of {param_name: tensor} from a peer.
            max_norm: per-tensor L2 norm threshold. Exceeding this is suspicious.

        Returns:
            The same dict if valid.

        Raises:
            ValueError: if any gradient fails validation.
        """
        for name, tensor in gradients.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError(f"Gradient '{name}' contains NaN or Inf values.")
            norm = tensor.norm().item()
            if norm > max_norm:
                raise ValueError(
                    f"Gradient '{name}' norm {norm:.2f} exceeds threshold {max_norm}. "
                    "Possible gradient poisoning — rejecting."
                )
        return gradients
