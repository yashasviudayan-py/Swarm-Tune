"""
Gradient averaging — the core math of federated learning.

Given gradient dicts from N peers (each trained on a different data shard),
compute a weighted average where each peer's contribution is proportional
to its local dataset size. This is Federated Averaging (FedAvg).

Reference: McMahan et al., "Communication-Efficient Learning of Deep
Networks from Decentralized Data" (2017). https://arxiv.org/abs/1602.05629

Why weighted average instead of simple mean?
  - Peers may hold different-sized data shards.
  - A peer with 10,000 samples should contribute more than one with 100.
  - Weighted average is unbiased for the true population gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import torch

log: structlog.BoundLogger = structlog.get_logger(__name__)


@dataclass
class PeerGradient:
    """A gradient contribution from a single peer."""

    peer_id: str
    gradients: dict[str, torch.Tensor]
    dataset_size: int  # number of samples this peer trained on


class GradientAverager:
    """
    Computes a dataset-size-weighted average of gradients from multiple peers.

    This is a pure math class — it has no network or I/O concerns.
    The TimeoutAggregator feeds it the gradients it has collected.
    """

    def average(self, contributions: list[PeerGradient]) -> dict[str, torch.Tensor]:
        """
        Compute the weighted average gradient across all peer contributions.

        Args:
            contributions: list of PeerGradient from live peers in this round.

        Returns:
            dict {param_name -> averaged tensor} ready to be applied to the model.

        Raises:
            ValueError: if contributions is empty or parameter shapes are inconsistent.
        """
        if not contributions:
            raise ValueError("Cannot average — no gradient contributions provided.")

        total_samples = sum(c.dataset_size for c in contributions)
        if total_samples == 0:
            raise ValueError("Total dataset size across peers is zero.")

        # Collect all parameter names from the first contribution
        # (all peers must share the same model architecture)
        param_names = set(contributions[0].gradients.keys())
        averaged: dict[str, torch.Tensor] = {}

        for name in param_names:
            weighted_sum: torch.Tensor | None = None

            for contrib in contributions:
                if name not in contrib.gradients:
                    log.warning(
                        "peer missing gradient for parameter",
                        peer_id=contrib.peer_id,
                        param=name,
                    )
                    continue

                grad = contrib.gradients[name].float()
                weight = contrib.dataset_size / total_samples
                weighted_grad = grad * weight

                if weighted_sum is None:
                    weighted_sum = weighted_grad
                else:
                    weighted_sum = weighted_sum + weighted_grad

            if weighted_sum is None:
                raise ValueError(f"No contributions found for parameter '{name}'.")

            averaged[name] = weighted_sum

        log.info(
            "gradients averaged",
            num_peers=len(contributions),
            total_samples=total_samples,
            num_params=len(averaged),
        )
        return averaged
