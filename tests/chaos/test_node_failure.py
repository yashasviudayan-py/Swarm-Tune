"""
Chaos tests: fault injection to prove straggler tolerance and adversarial resistance.

Two categories:

TestNodeFailureTolerance — nodes drop mid-training.
  Proves that partial aggregation, deferred rounds, node re-join, and
  duplicate-submission deduplication all work correctly.

TestAdversarialGradientRejection — a peer broadcasts poisoned gradients.
  Proves the Phase 4 security gate: NaN values, out-of-bounds norms, and
  malformed bytes are all detected and rejected before reaching the aggregator.
  The swarm must continue training on the honest peers' contributions.

Run with: make test-chaos
"""

from __future__ import annotations

import pytest
import torch

from swarm_tune.node.aggregator.averaging import GradientAverager, PeerGradient
from swarm_tune.node.aggregator.timeout import TimeoutAggregator
from swarm_tune.node.trainer.gradient import GradientExtractor
from swarm_tune.node.trainer.serializer import GradientSerializer


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


@pytest.mark.chaos
class TestAdversarialGradientRejection:
    """
    Phase 4 security gate: poisoned gradients must be detected and rejected
    before reaching the aggregator.  The swarm must continue training on
    the remaining honest peers' contributions.

    These tests mirror what happens in the Docker simulation when
    node_5_adversarial broadcasts NaN payloads every round.
    """

    def test_nan_gradient_rejected_by_validator(
        self, simple_gradients: dict[str, torch.Tensor]
    ) -> None:
        """NaN values anywhere in a gradient tensor trigger rejection."""
        poisoned = {
            name: torch.full_like(g, float("nan")) for name, g in simple_gradients.items()
        }
        with pytest.raises(ValueError, match="NaN or Inf"):
            GradientExtractor().validate(poisoned)

    def test_inf_gradient_rejected_by_validator(
        self, simple_gradients: dict[str, torch.Tensor]
    ) -> None:
        """Inf values (another common poisoning vector) are also caught."""
        poisoned = {
            name: torch.full_like(g, float("inf")) for name, g in simple_gradients.items()
        }
        with pytest.raises(ValueError, match="NaN or Inf"):
            GradientExtractor().validate(poisoned)

    def test_large_norm_gradient_rejected(
        self, simple_gradients: dict[str, torch.Tensor]
    ) -> None:
        """Gradients with implausibly large L2 norm are rejected (poisoning defence)."""
        # Scale gradients so their norm far exceeds max_norm=1e4
        poisoned = {name: g * 1e8 for name, g in simple_gradients.items()}
        with pytest.raises(ValueError, match="norm"):
            GradientExtractor().validate(poisoned, max_norm=1e4)

    def test_malformed_bytes_rejected_at_deserialization(self) -> None:
        """Completely invalid bytes from a hostile peer never reach the validator."""
        with pytest.raises(ValueError):
            GradientSerializer().deserialize(b"this_is_not_a_swrm_payload")

    def test_swrm_wrong_magic_rejected(self) -> None:
        """Bytes with wrong magic header are caught before torch.load."""
        bad = b"XXXX" + b"\x00" * 20
        with pytest.raises(ValueError, match=r"[Mm]agic"):
            GradientSerializer().deserialize(bad)

    def test_swarm_continues_after_adversarial_rejection(
        self,
        three_peer_gradients: list[PeerGradient],
        simple_gradients: dict[str, torch.Tensor],
    ) -> None:
        """
        End-to-end adversarial scenario:

        1. An adversarial peer produces NaN-poisoned gradients.
        2. validate() rejects them — they never reach the aggregator.
        3. The 3 honest peers' contributions are averaged correctly.
        4. The averaged result is finite for every parameter.

        This mirrors the _on_peer_gradient handler's try/except path in
        production: rejection is logged, the round proceeds on honest data.
        """
        adversarial = PeerGradient(
            peer_id="node_5_adversarial",
            gradients={
                name: torch.full_like(g, float("nan")) for name, g in simple_gradients.items()
            },
            dataset_size=100,
        )

        validator = GradientExtractor()
        honest_contributions: list[PeerGradient] = []

        for contrib in [*three_peer_gradients, adversarial]:
            try:
                validator.validate(contrib.gradients)
                honest_contributions.append(contrib)
            except ValueError:
                pass  # adversarial peer rejected — swarm continues

        assert len(honest_contributions) == 3, (
            "All 3 honest peers should pass validation; adversarial peer should be excluded"
        )

        averaged = GradientAverager().average(honest_contributions)

        for name, tensor in averaged.items():
            assert torch.isfinite(tensor).all(), (
                f"Averaged gradient for '{name}' contains non-finite values — "
                "adversarial contribution must have leaked into the average"
            )
