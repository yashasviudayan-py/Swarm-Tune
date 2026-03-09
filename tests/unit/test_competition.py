"""Unit tests for the competition logic module (Phase 8)."""

from __future__ import annotations

import json

import pytest

from swarm_tune.runs.competition import (
    determine_winner,
    make_result,
    parse_perplexity,
)


@pytest.mark.unit
class TestParsePerplexity:
    def test_parses_simple_output(self) -> None:
        output = "Loading model...\nperplexity: 42.73\n"
        assert parse_perplexity(output) == pytest.approx(42.73)

    def test_parses_integer_perplexity(self) -> None:
        output = "perplexity: 100\n"
        assert parse_perplexity(output) == pytest.approx(100.0)

    def test_parses_multiline_output(self) -> None:
        output = (
            "Loading model: gpt2\n"
            "Checkpoint:    checkpoints/team_alpha.pt\n"
            "Dataset:       wikitext / wikitext-103-raw-v1\n"
            "Batches evaluated: 200\n"
            "Tokens evaluated:  409,600\n"
            "Average loss:      3.8500\n"
            "perplexity: 47.12\n"
        )
        assert parse_perplexity(output) == pytest.approx(47.12)

    def test_parses_with_leading_whitespace(self) -> None:
        output = "  perplexity: 55.55\n"
        assert parse_perplexity(output) == pytest.approx(55.55)

    def test_raises_when_no_perplexity_line(self) -> None:
        output = "error: checkpoint not found\n"
        with pytest.raises(ValueError, match="No 'perplexity: <float>'"):
            parse_perplexity(output)

    def test_raises_on_empty_output(self) -> None:
        with pytest.raises(ValueError, match="No 'perplexity: <float>'"):
            parse_perplexity("")

    def test_does_not_match_partial_word(self) -> None:
        # "total_perplexity: 10.0" should NOT match — it must start the line
        output = "total_perplexity: 10.0\n"
        with pytest.raises(ValueError):
            parse_perplexity(output)


@pytest.mark.unit
class TestDetermineWinner:
    def test_team_a_wins(self) -> None:
        # Lower perplexity = better
        assert determine_winner("alpha", 30.0, "beta", 50.0) == "alpha"

    def test_team_b_wins(self) -> None:
        assert determine_winner("alpha", 80.0, "beta", 45.0) == "beta"

    def test_tie_within_tolerance(self) -> None:
        # 0.3 difference < default tolerance 0.5
        assert determine_winner("alpha", 50.0, "beta", 50.3) == "tie"

    def test_tie_exact(self) -> None:
        assert determine_winner("alpha", 50.0, "beta", 50.0) == "tie"

    def test_tie_boundary_exactly_at_tolerance(self) -> None:
        # Exactly at tolerance → tie (≤, not <)
        assert determine_winner("alpha", 50.0, "beta", 50.5, tie_tolerance=0.5) == "tie"

    def test_no_tie_just_above_tolerance(self) -> None:
        # 0.51 > 0.5 → team with lower ppl wins
        result = determine_winner("alpha", 50.0, "beta", 50.51, tie_tolerance=0.5)
        assert result == "alpha"

    def test_custom_tolerance(self) -> None:
        # With tolerance=5.0, a 4.9 difference is still a tie
        assert determine_winner("alpha", 40.0, "beta", 44.9, tie_tolerance=5.0) == "tie"

    def test_raises_on_nan_perplexity(self) -> None:
        import math

        with pytest.raises(ValueError, match="positive finite"):
            determine_winner("alpha", math.nan, "beta", 50.0)

    def test_raises_on_inf_perplexity(self) -> None:
        import math

        with pytest.raises(ValueError, match="positive finite"):
            determine_winner("alpha", 30.0, "beta", math.inf)

    def test_raises_on_zero_perplexity(self) -> None:
        with pytest.raises(ValueError, match="positive finite"):
            determine_winner("alpha", 0.0, "beta", 50.0)

    def test_raises_on_negative_perplexity(self) -> None:
        with pytest.raises(ValueError, match="positive finite"):
            determine_winner("alpha", -5.0, "beta", 50.0)


@pytest.mark.unit
class TestMakeResult:
    def _make(self, **overrides: object) -> dict[str, object]:
        defaults: dict[str, object] = {
            "competition_id": "gpt2-competition-001",
            "team_a_id": "alpha",
            "team_a_ppl": 47.12,
            "team_a_checkpoint": "checkpoints/alpha.pt",
            "team_b_id": "beta",
            "team_b_ppl": 55.30,
            "team_b_checkpoint": "checkpoints/beta.pt",
        }
        defaults.update(overrides)
        return make_result(**defaults)  # type: ignore[arg-type]

    def test_result_has_required_keys(self) -> None:
        result = self._make()
        assert "competition_id" in result
        assert "teams" in result
        assert "winner" in result
        assert "tie_tolerance" in result
        assert "verified_at" in result
        assert "verification_command" in result

    def test_winner_correct(self) -> None:
        result = self._make(team_a_ppl=40.0, team_b_ppl=60.0)
        assert result["winner"] == "alpha"

    def test_loser_identified(self) -> None:
        result = self._make(team_a_ppl=70.0, team_b_ppl=50.0)
        assert result["winner"] == "beta"

    def test_tie_result(self) -> None:
        result = self._make(team_a_ppl=50.0, team_b_ppl=50.2, tie_tolerance=0.5)
        assert result["winner"] == "tie"

    def test_teams_dict_contains_both_teams(self) -> None:
        result = self._make()
        teams = result["teams"]
        assert isinstance(teams, dict)
        assert "alpha" in teams
        assert "beta" in teams

    def test_teams_have_perplexity_and_checkpoint(self) -> None:
        result = self._make()
        teams = result["teams"]
        for team_data in teams.values():
            assert "perplexity" in team_data
            assert "checkpoint" in team_data

    def test_perplexity_rounded_to_4_places(self) -> None:
        result = self._make(team_a_ppl=47.123456789)
        teams = result["teams"]
        ppl = teams["alpha"]["perplexity"]
        # round() to 4 decimal places
        assert ppl == round(47.123456789, 4)

    def test_result_is_json_serialisable(self) -> None:
        result = self._make()
        serialised = json.dumps(result)
        decoded = json.loads(serialised)
        assert decoded["competition_id"] == "gpt2-competition-001"

    def test_verified_at_is_iso_utc(self) -> None:
        result = self._make()
        # Should end with +00:00 or Z (UTC marker)
        verified_at = result["verified_at"]
        assert isinstance(verified_at, str)
        assert "T" in verified_at  # ISO-8601 datetime separator
        assert "+00:00" in verified_at or verified_at.endswith("Z")

    def test_competition_id_preserved(self) -> None:
        result = self._make(competition_id="my-comp-42")
        assert result["competition_id"] == "my-comp-42"
