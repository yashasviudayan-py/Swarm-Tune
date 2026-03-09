"""
Competition Logic — Phase 8

Pure, stateless functions for Phase 8 competition orchestration.
No I/O, no network, no side effects — fully testable in isolation.

The competition winner is the team with the *lowest* perplexity on the
WikiText-103 test split (lower = better language model).

Results are written to a JSON file that any third party can verify by
re-running `make benchmark CHECKPOINT=<path>` on the published checkpoints.
No central authority required.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

# ---------------------------------------------------------------------------
# Perplexity parsing
# ---------------------------------------------------------------------------

# Matches "perplexity: 42.73" anywhere in a block of text.
_PPL_PATTERN = re.compile(r"^\s*perplexity:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)


def parse_perplexity(output: str) -> float:
    """
    Parse the perplexity value printed by scripts/benchmark.py.

    The benchmark script always outputs a line of the form:
        perplexity: <float>

    Args:
        output: full stdout captured from the benchmark process.

    Returns:
        Perplexity as a float (lower = better).

    Raises:
        ValueError: if no valid perplexity line is found.
    """
    match = _PPL_PATTERN.search(output)
    if match is None:
        raise ValueError(
            f"No 'perplexity: <float>' line found in benchmark output.\nOutput was:\n{output[:500]}"
        )
    return float(match.group(1))


# ---------------------------------------------------------------------------
# Winner determination
# ---------------------------------------------------------------------------


def determine_winner(
    team_a_id: str,
    team_a_ppl: float,
    team_b_id: str,
    team_b_ppl: float,
    tie_tolerance: float = 0.5,
) -> str:
    """
    Determine the competition winner by perplexity (lower wins).

    Args:
        team_a_id:     identifier for Team A.
        team_a_ppl:    Team A's perplexity (must be positive finite float).
        team_b_id:     identifier for Team B.
        team_b_ppl:    Team B's perplexity (must be positive finite float).
        tie_tolerance: absolute perplexity difference below which we call a tie.
                       Default 0.5 — differences smaller than this are noise.

    Returns:
        The winning team_id, or "tie" if |team_a_ppl - team_b_ppl| ≤ tie_tolerance.

    Raises:
        ValueError: if any perplexity is not a positive finite number.
    """
    import math

    for label, ppl in [("team_a_ppl", team_a_ppl), ("team_b_ppl", team_b_ppl)]:
        if not (math.isfinite(ppl) and ppl > 0):
            raise ValueError(f"{label}={ppl!r} must be a positive finite number")

    if abs(team_a_ppl - team_b_ppl) <= tie_tolerance:
        return "tie"
    return team_a_id if team_a_ppl < team_b_ppl else team_b_id


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


def make_result(
    competition_id: str,
    team_a_id: str,
    team_a_ppl: float,
    team_a_checkpoint: str,
    team_b_id: str,
    team_b_ppl: float,
    team_b_checkpoint: str,
    tie_tolerance: float = 0.5,
) -> dict[str, Any]:
    """
    Build the canonical competition result dict.

    The returned dict is JSON-serialisable and captures everything needed
    for independent verification:
      - checkpoint paths (so anyone can re-run benchmark)
      - perplexity scores
      - winner
      - timestamp (UTC ISO-8601)

    Args:
        competition_id:   unique competition identifier from the manifest.
        team_a_id:        Team A's team_id.
        team_a_ppl:       Team A's perplexity score.
        team_a_checkpoint: path to Team A's reconstructed checkpoint.
        team_b_id:        Team B's team_id.
        team_b_ppl:       Team B's perplexity score.
        team_b_checkpoint: path to Team B's reconstructed checkpoint.
        tie_tolerance:    passed to determine_winner.

    Returns:
        dict with keys: competition_id, teams, winner, tie_tolerance,
                         verified_at (UTC timestamp), verification_command.
    """
    winner = determine_winner(team_a_id, team_a_ppl, team_b_id, team_b_ppl, tie_tolerance)
    return {
        "competition_id": competition_id,
        "teams": {
            team_a_id: {
                "perplexity": round(team_a_ppl, 4),
                "checkpoint": team_a_checkpoint,
            },
            team_b_id: {
                "perplexity": round(team_b_ppl, 4),
                "checkpoint": team_b_checkpoint,
            },
        },
        "winner": winner,
        "tie_tolerance": tie_tolerance,
        "verified_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "verification_command": (
            "make benchmark CHECKPOINT=<checkpoint_path>  # lower perplexity = better"
        ),
    }
