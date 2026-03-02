"""Unit tests for gradient averaging and timeout aggregation."""

from __future__ import annotations

import asyncio

import pytest
import torch

from swarm_tune.node.aggregator.averaging import GradientAverager, PeerGradient
from swarm_tune.node.aggregator.timeout import TimeoutAggregator


class TestGradientAverager:
    """Tests for the FedAvg weighted averaging math."""

    def test_equal_weights_is_simple_mean(self, three_peer_gradients: list[PeerGradient]) -> None:
        """With equal dataset sizes, weighted average == simple mean."""
        averager = GradientAverager()
        result = averager.average(three_peer_gradients)

        # Manually compute expected mean for one parameter
        param = "0.weight"
        expected = torch.stack([c.gradients[param] for c in three_peer_gradients]).mean(0)
        assert torch.allclose(result[param], expected, atol=1e-5)

    def test_weighted_by_dataset_size(self, simple_gradients: dict[str, torch.Tensor]) -> None:
        """A peer with 2x the data should contribute 2x the weight."""
        ones = {k: torch.ones_like(v) for k, v in simple_gradients.items()}
        zeros = {k: torch.zeros_like(v) for k, v in simple_gradients.items()}
        big = PeerGradient("big", ones, 200)
        small = PeerGradient("small", zeros, 100)

        result = GradientAverager().average([big, small])
        # Expected: (200/300) * 1 + (100/300) * 0 = 0.667...
        expected_val = 200 / 300
        expected = torch.full_like(result["0.bias"], expected_val)
        assert torch.allclose(result["0.bias"], expected, atol=1e-5)

    def test_empty_contributions_raises(self) -> None:
        with pytest.raises(ValueError, match="no gradient contributions"):
            GradientAverager().average([])

    def test_all_params_present_in_result(self, three_peer_gradients: list[PeerGradient]) -> None:
        result = GradientAverager().average(three_peer_gradients)
        expected_keys = set(three_peer_gradients[0].gradients.keys())
        assert set(result.keys()) == expected_keys


class TestTimeoutAggregator:
    """Tests for the straggler tolerance mechanism."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_resolves_when_min_peers_met(
        self,
        base_settings: object,
        three_peer_gradients: list[PeerGradient],
    ) -> None:
        agg = TimeoutAggregator(base_settings)  # type: ignore[arg-type]
        agg.open_round(0)

        # Submit 2 gradients (min_peers_for_round = 2)
        agg.submit(three_peer_gradients[0])
        agg.submit(three_peer_gradients[1])

        contributions = await asyncio.wait_for(agg.wait(), timeout=3.0)
        assert len(contributions) == 2

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_partial_result_on_timeout(
        self,
        base_settings: object,
        three_peer_gradients: list[PeerGradient],
    ) -> None:
        """With only 1 peer (below min), wait() should timeout and return partial."""
        agg = TimeoutAggregator(base_settings)  # type: ignore[arg-type]
        agg.open_round(0)
        agg.submit(three_peer_gradients[0])  # only 1, min is 2

        contributions = await asyncio.wait_for(agg.wait(), timeout=5.0)
        assert len(contributions) == 1

    @pytest.mark.unit
    def test_duplicate_submission_ignored(
        self,
        base_settings: object,
        peer_gradient: PeerGradient,
    ) -> None:
        agg = TimeoutAggregator(base_settings)  # type: ignore[arg-type]
        agg.open_round(0)
        agg.submit(peer_gradient)
        agg.submit(peer_gradient)  # duplicate
        assert len(agg._contributions) == 1

    @pytest.mark.unit
    def test_insufficient_peers_raises_on_get(
        self,
        base_settings: object,
        peer_gradient: PeerGradient,
    ) -> None:
        agg = TimeoutAggregator(base_settings)  # type: ignore[arg-type]
        agg.open_round(0)
        agg.submit(peer_gradient)  # only 1, min is 2

        with pytest.raises(ValueError, match="Deferring round"):
            agg.get_averaged_gradients()
