"""
Aggregation strategy abstraction.

Controls how gradient contributions from peers are combined into
a single averaged gradient that is applied to the local model.

At 20 nodes: FlatAggregation collects from all peers in one round.
At 100 nodes: HierarchicalAggregation runs two levels:
  Level 1 — each cluster of ~10 nodes averages locally.
  Level 2 — cluster leaders average their local results globally.

This reduces per-round latency from O(N) to O(√N) and cuts each
node's bandwidth requirement by a factor of ~10.

The SwarmNode training loop calls strategy.aggregate() and never
knows which strategy is running. Swap via config, not code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog
import torch

from swarm_tune.node.aggregator.averaging import GradientAverager, PeerGradient

log: structlog.BoundLogger = structlog.get_logger(__name__)


@runtime_checkable
class AggregationStrategy(Protocol):
    """Strategy interface for combining peer gradient contributions."""

    def aggregate(self, contributions: list[PeerGradient]) -> dict[str, torch.Tensor]:
        """
        Combine gradient contributions into a single averaged gradient.

        Args:
            contributions: PeerGradient list from the TimeoutAggregator.

        Returns:
            {param_name -> averaged tensor} ready to apply to the model.
        """
        ...


class FlatAggregation:
    """
    Single-level FedAvg across all contributing peers.

    The default strategy for Phases 1-4 and swarms up to ~30 nodes.
    Simple, correct, and easy to reason about.

    Limitation: at 100 nodes the timeout window must accommodate the
    slowest of 99 peers. Hierarchical aggregation solves this.
    """

    def __init__(self) -> None:
        self._averager = GradientAverager()

    def aggregate(self, contributions: list[PeerGradient]) -> dict[str, torch.Tensor]:
        log.debug("flat aggregation", num_peers=len(contributions))
        return self._averager.average(contributions)


class HierarchicalAggregation:
    """
    Two-level FedAvg for large swarms (100+ nodes).

    Level 1: each cluster pre-averages its members' gradients locally.
    Level 2: cluster-level averages are re-averaged globally.

    This class is a stub. Implementation is a Phase 5+ concern.
    It exists now so the SwarmNode training loop never needs to change.

    When implemented:
    - The node must know which peers are in its cluster (ClusterPeerSelector).
    - Cluster leaders must be elected (e.g. lowest peer_id in cluster).
    - Cross-cluster gossip uses a separate libp2p topic.
    """

    def __init__(self, cluster_id: int, cluster_size: int) -> None:
        self.cluster_id = cluster_id
        self.cluster_size = cluster_size
        self._averager = GradientAverager()

    def aggregate(self, contributions: list[PeerGradient]) -> dict[str, torch.Tensor]:
        # TODO(phase-5): implement two-level hierarchical averaging
        # For now, fall back to flat averaging (safe, correct, not optimal at scale)
        log.warning(
            "HierarchicalAggregation not yet implemented — falling back to flat averaging",
            cluster_id=self.cluster_id,
        )
        return self._averager.average(contributions)
