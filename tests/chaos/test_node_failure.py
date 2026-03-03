"""
Chaos tests: fault injection to prove straggler tolerance.

These tests simulate the real-world scenario where nodes drop mid-training.
They are slow (involve real timeouts) and are tagged @pytest.mark.chaos.

Run with: make test-chaos
"""

from __future__ import annotations

import pytest
import torch

from swarm_tune.node.aggregator.averaging import PeerGradient
from swarm_tune.node.aggregator.timeout import TimeoutAggregator


@pytest.mark.chaos
class TestNodeFailureTolerance:
    """The swarm must survive nodes going offline mid-round."""

    @pytest.mark.anyio
    async def test_training_proceeds_when_node_drops(
        self,
        base_settings: object,
        three_peer_gradients: list[PeerGradient],
    ) -> None:
        """
        Scenario: 3 peers are expected but only 2 submit before timeout.
        The round should still complete (partial aggregation).
        """
        agg = TimeoutAggregator(base_settings)  # type: ignore[arg-type]
        agg.open_round(0)

        # Submit 2 of 3 peers — peer_2 "drops offline"
        agg.submit(three_peer_gradients[0])
        agg.submit(three_peer_gradients[1])
        # peer_2 is silent

        contributions = await agg.wait()
        assert len(contributions) == 2
        # Min peers (2) was met, so averaging should succeed
        result = agg.get_averaged_gradients()
        assert isinstance(result, dict)
        assert len(result) > 0

    @pytest.mark.anyio
    async def test_all_nodes_drop_defers_round(
        self,
        base_settings: object,
    ) -> None:
        """
        Scenario: no peers submit within the timeout window.
        The round must be deferred, not crash the node.
        """
        agg = TimeoutAggregator(base_settings)  # type: ignore[arg-type]
        agg.open_round(0)
        # Submit nothing — simulate total partition

        contributions = await agg.wait()
        assert contributions == []

        with pytest.raises(ValueError, match="Deferring round"):
            agg.get_averaged_gradients()

    @pytest.mark.anyio
    async def test_late_joining_node_participates_next_round(
        self,
        base_settings: object,
        three_peer_gradients: list[PeerGradient],
    ) -> None:
        """
        Scenario: peer_2 misses round 0 but rejoins for round 1.
        Round 1 should include all 3 peers.
        """
        agg = TimeoutAggregator(base_settings)  # type: ignore[arg-type]

        # Round 0: only peers 0 and 1
        agg.open_round(0)
        agg.submit(three_peer_gradients[0])
        agg.submit(three_peer_gradients[1])
        await agg.wait()
        r0_result = agg.get_averaged_gradients()
        assert r0_result is not None

        # Round 1: all three peers (peer_2 rejoined)
        agg.open_round(1)
        agg.submit(three_peer_gradients[0])
        agg.submit(three_peer_gradients[1])
        agg.submit(three_peer_gradients[2])  # peer_2 is back
        await agg.wait()
        r1_result = agg.get_averaged_gradients()

        # Round 1 averages 3 peers, round 0 averaged 2 — results should differ
        for name in r0_result:
            assert not torch.allclose(r0_result[name], r1_result[name], atol=1e-8), (
                f"Round 0 and round 1 results should differ for '{name}' "
                "since different peers contributed"
            )

    @pytest.mark.anyio
    async def test_duplicate_peer_submission_is_idempotent(
        self,
        base_settings: object,
        three_peer_gradients: list[PeerGradient],
    ) -> None:
        """
        A peer that submits gradients twice (e.g., due to network retry)
        must not be double-counted in the average.
        """
        agg = TimeoutAggregator(base_settings)  # type: ignore[arg-type]
        agg.open_round(0)

        agg.submit(three_peer_gradients[0])
        agg.submit(three_peer_gradients[0])  # retry — must be ignored
        agg.submit(three_peer_gradients[1])

        await agg.wait()
        assert len(agg._contributions) == 2  # not 3
