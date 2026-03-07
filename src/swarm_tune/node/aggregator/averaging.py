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

Intersection-based parameter averaging
--------------------------------------
Previous implementation raised ValueError when contributions had different
parameter sets (broken for multi-shard model-parallel mode). The fix uses
the INTERSECTION of parameter names across all contributions:

  - Data-parallel mode (all nodes train all layers, shard_total=1):
    Intersection == all params. Behaviour is identical to before.

  - Multi-shard model-parallel mode (nodes train different layer subsets):
    Nodes with the SAME shard index have the same param set -> intersection
    = their full shared params -> correct FedAvg among same-shard peers.
    Nodes with DIFFERENT shard indices have no overlapping params ->
    intersection = empty -> ValueError("No common parameters"). In this
    case the caller falls back to the local gradient, which is the correct
    behaviour for independent layer shards. A warning is emitted so the
    operator knows to route same-shard nodes into the same cluster.

Streaming memory model
----------------------
average() processes one parameter at a time and deletes each peer's copy
immediately after accumulation. Peak RAM = O(1 parameter * N_peers) instead
of O(all parameters * N_peers). This makes the implementation viable for
real LLM-sized models where the naive approach would OOM.

IMPORTANT: average() MUTATES the contributions list by removing tensors from
each PeerGradient.gradients dict as it processes them. Callers must not
access contribution.gradients after calling average(). The TimeoutAggregator
only calls average() once per round (immediately before discarding
contributions), so this is safe in the current architecture.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field

import structlog
import torch

log: structlog.BoundLogger = structlog.get_logger(__name__)


@dataclass
class PeerGradient:
    """A gradient contribution from a single peer."""

    peer_id: str
    gradients: dict[str, torch.Tensor]
    dataset_size: int  # number of samples this peer trained on
    # Optional: IP address of the submitting peer (used for subnet Sybil cap).
    # Empty string when Sybil resistance is disabled or IP is unknown.
    peer_ip: str = field(default="")


def _subnet_key(ip: str, prefix_len: int) -> str:
    """
    Return the network address string for an IP given a prefix length.

    IPv6 addresses are handled separately: applying an IPv4 /N prefix to an
    IPv6 address via ip_interface() succeeds (no ValueError) but produces
    semantically wrong subnet grouping. For IPv6, we use /64 as the natural
    subnet boundary regardless of the configured prefix_len (which is an
    IPv4-centric concept).
    """
    try:
        addr = ipaddress.ip_address(ip)
        effective_prefix = prefix_len if isinstance(addr, ipaddress.IPv4Address) else 64
        iface = ipaddress.ip_interface(f"{ip}/{effective_prefix}")
        return str(iface.network.network_address)
    except ValueError:
        return ip  # unparseable IP: treat as its own subnet


def _apply_subnet_cap(
    contributions: list[PeerGradient],
    prefix_len: int,
    max_subnet_weight: float,
) -> list[PeerGradient]:
    """
    Clamp the effective dataset size of any single /N subnet to max_subnet_weight
    (expressed as a multiplier of the smallest non-zero dataset_size seen).

    Peers without an IP are treated as their own unique subnet.

    This prevents a single operator running many nodes on the same IP block
    from dominating FedAvg — the subnet's total contribution is capped at
    one representative node's worth.
    """
    if not any(c.peer_ip for c in contributions):
        return contributions  # no IP info — skip capping

    # Group contributions by subnet.
    subnet_groups: dict[str, list[PeerGradient]] = {}
    for c in contributions:
        key = _subnet_key(c.peer_ip, prefix_len) if c.peer_ip else f"unknown:{c.peer_id}"
        subnet_groups.setdefault(key, []).append(c)

    # The cap is max_subnet_weight * (size of smallest non-zero shard).
    min_size = min((c.dataset_size for c in contributions if c.dataset_size > 0), default=1)
    cap_total = max(1, round(max_subnet_weight * min_size))

    capped: list[PeerGradient] = []
    for subnet, group in subnet_groups.items():
        total = sum(c.dataset_size for c in group)
        if total <= cap_total or len(group) == 1:
            capped.extend(group)
        else:
            # Scale down each member's dataset_size proportionally.
            # Use round() not int() — int() truncates, causing the subnet's
            # total effective weight to be slightly under cap_total (biased low).
            # round() distributes rounding error evenly across peers.
            ratio = cap_total / total
            scaled = [
                PeerGradient(
                    peer_id=c.peer_id,
                    gradients=c.gradients,
                    dataset_size=max(1, round(c.dataset_size * ratio)),
                    peer_ip=c.peer_ip,
                )
                for c in group
            ]
            log.warning(
                "sybil cap applied to subnet",
                subnet=subnet,
                num_peers=len(group),
                original_total=total,
                capped_total=sum(s.dataset_size for s in scaled),
            )
            capped.extend(scaled)

    return capped


class GradientAverager:
    """
    Computes a dataset-size-weighted average of gradients from multiple peers.

    Uses intersection-based parameter matching: only parameters present in
    ALL contributions are averaged. This makes the averager correct in both
    data-parallel mode (all nodes have all params -> intersection = all params)
    and multi-shard model-parallel mode (same-shard nodes have the same params
    -> intersection = their shared params; cross-shard nodes have no overlap ->
    intersection = empty -> ValueError, caller falls back to local gradient).

    When sybil_resistance=True, peers sharing the same IP /N subnet have their
    combined contribution capped so no single operator can dominate FedAvg.

    Memory model (streaming):
        average() processes parameters one at a time, deleting each peer's
        tensor immediately after accumulation. Peak RAM scales as
        O(1 parameter x N_peers) rather than O(all parameters x N_peers),
        making the averager viable for 70B+ parameter models.

    SIDE EFFECT: average() mutates input contributions by removing gradient
        tensors from each PeerGradient.gradients dict. Do not access
        contribution.gradients after calling average().
    """

    def __init__(
        self,
        sybil_resistance: bool = False,
        subnet_prefix: int = 24,
        max_subnet_weight: float = 1.0,
    ) -> None:
        self._sybil_resistance = sybil_resistance
        self._subnet_prefix = subnet_prefix
        self._max_subnet_weight = max_subnet_weight

    def average(self, contributions: list[PeerGradient]) -> dict[str, torch.Tensor]:
        """
        Compute the weighted average gradient across all peer contributions.

        Uses intersection of parameter names: only params present in ALL
        contributions are averaged. If contributions have different param
        sets (e.g. multi-shard nodes with different layer assignments), only
        the common subset is averaged, with a warning logged for the dropped
        parameters.

        Streaming memory model: tensors are deleted from each contribution's
        gradient dict immediately after accumulation. Peak RAM is proportional
        to (one parameter tensor x N_peers), not (all parameters x N_peers).

        Args:
            contributions: list of PeerGradient from live peers in this round.
                           WARNING: this list is mutated — gradient tensors are
                           removed from each contribution as they are processed.

        Returns:
            dict {param_name -> averaged tensor} ready to be applied to the model.

        Raises:
            ValueError: if contributions is empty or no common parameters exist.
        """
        if not contributions:
            raise ValueError("Cannot average — no gradient contributions provided.")

        # Apply Sybil subnet cap before computing weights (Phase 5+).
        if self._sybil_resistance:
            contributions = _apply_subnet_cap(
                contributions, self._subnet_prefix, self._max_subnet_weight
            )

        total_samples = sum(c.dataset_size for c in contributions)
        if total_samples == 0:
            raise ValueError("Total dataset size across peers is zero.")

        # Compute the intersection of parameter names across all contributions.
        # This is robust to multi-shard mode where different nodes have different
        # trainable layers. In data-parallel mode (the common case), all nodes
        # have the same params and the intersection equals the full param set.
        param_names: set[str] = set(contributions[0].gradients.keys())
        for contrib in contributions[1:]:
            contrib_keys = set(contrib.gradients.keys())
            dropped = param_names - contrib_keys
            if dropped:
                log.warning(
                    "peer has different parameter set — using intersection only",
                    peer_id=contrib.peer_id,
                    params_dropped=sorted(dropped),
                    hint=(
                        "In multi-shard mode this is expected for nodes with different "
                        "shard_index. Set cluster_id=shard_index to route same-shard "
                        "nodes together so they only exchange relevant gradients."
                    ),
                )
            param_names &= contrib_keys

        if not param_names:
            raise ValueError(
                "No common parameters found across contributions. "
                "In multi-shard mode, nodes with different shard_index have no "
                "overlapping parameters — gradient exchange between them is undefined. "
                "Set cluster_id=shard_index to group same-shard nodes together."
            )

        averaged: dict[str, torch.Tensor] = {}

        # --- Streaming accumulation (memory-efficient) ---
        # Process one parameter at a time. After accumulating all peer contributions
        # for a parameter, immediately delete that tensor from every contribution.
        # Peak RAM = size of (one parameter) x (N_peers + 1 weighted_sum) instead of
        # all parameters x N_peers, which makes this viable for 70B-parameter models.
        for name in sorted(param_names):  # sorted for deterministic processing order
            weighted_sum: torch.Tensor | None = None

            for contrib in contributions:
                # Pop (not get) so the tensor is removed from the dict after use,
                # releasing the reference immediately after we accumulate it.
                grad = contrib.gradients.pop(name, None)
                if grad is None:
                    continue

                weight = contrib.dataset_size / total_samples
                # mul_ is in-place to avoid a third allocation; float() upcasts
                # from fp16/bf16 to fp32 for numerically stable accumulation.
                term = grad.float().mul_(weight)

                if weighted_sum is None:
                    weighted_sum = term
                else:
                    weighted_sum.add_(term)

                del grad, term  # drop reference immediately

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
