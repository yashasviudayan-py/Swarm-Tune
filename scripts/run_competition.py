#!/usr/bin/env python3
"""
Swarm-Tune Competition Coordinator — Phase 8

Benchmarks two team checkpoints and declares a winner by perplexity.
Lower perplexity = better language model = winner.

This script is the *final step* after each team has independently run their
swarm and reconstructed a full-model checkpoint via:

    make reconstruct CHECKPOINT_DIR=checkpoints/ MODEL=gpt2

Usage:
    python scripts/run_competition.py \\
        --competition-id gpt2-competition-001 \\
        --team-a-id team-alpha --team-a-checkpoint checkpoints/team_alpha.pt \\
        --team-b-id team-beta  --team-b-checkpoint checkpoints/team_beta.pt \\
        --output results/competition_result.json

    make competition \\
        COMPETITION_ID=gpt2-competition-001 \\
        TEAM_A_ID=team-alpha TEAM_A_CHECKPOINT=ckpts/alpha.pt \\
        TEAM_B_ID=team-beta  TEAM_B_CHECKPOINT=ckpts/beta.pt

No central server required. Any third party can independently verify the
result by running `make benchmark CHECKPOINT=<path>` on the published
checkpoints. Results are identical because perplexity evaluation is
deterministic: same checkpoint + same test split → same number, always.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark two team checkpoints and declare a competition winner."
    )
    p.add_argument(
        "--competition-id",
        default="",
        help="Competition identifier (from the run manifest). Default: empty string.",
    )
    p.add_argument(
        "--team-a-id",
        required=True,
        help="Team A identifier (e.g. 'team-alpha').",
    )
    p.add_argument(
        "--team-a-checkpoint",
        required=True,
        type=Path,
        help="Path to Team A's reconstructed checkpoint .pt file.",
    )
    p.add_argument(
        "--team-b-id",
        required=True,
        help="Team B identifier (e.g. 'team-beta').",
    )
    p.add_argument(
        "--team-b-checkpoint",
        required=True,
        type=Path,
        help="Path to Team B's reconstructed checkpoint .pt file.",
    )
    p.add_argument(
        "--model-name",
        default="gpt2",
        help="HuggingFace model architecture (must match both checkpoints). Default: gpt2",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the JSON results file. Default: competition_result.json",
    )
    p.add_argument(
        "--benchmark-script",
        type=Path,
        default=None,
        help=(
            "Path to the benchmark script. "
            "Default: scripts/benchmark.py relative to this script's directory."
        ),
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum evaluation batches per checkpoint (useful for quick tests).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Compute device for benchmarking. Default: cpu",
    )
    p.add_argument(
        "--tie-tolerance",
        type=float,
        default=0.5,
        help=(
            "Perplexity difference below which the result is declared a tie. "
            "Default: 0.5 (differences smaller than this are measurement noise)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip actual benchmarking. Use 0.0 as perplexity for both teams. "
            "Useful for testing the orchestration pipeline."
        ),
    )
    return p.parse_args()


def _benchmark(
    checkpoint: Path,
    model_name: str,
    benchmark_script: Path,
    device: str,
    max_batches: int | None,
) -> float:
    """
    Run the benchmark script on a checkpoint and return its perplexity.

    Streams stdout to the terminal in real time so the user can see progress,
    and also captures it for parsing.
    """
    from swarm_tune.runs.competition import parse_perplexity

    cmd = [
        sys.executable,
        str(benchmark_script),
        "--checkpoint",
        str(checkpoint),
        "--model-name",
        model_name,
        "--device",
        device,
    ]
    if max_batches is not None:
        cmd += ["--max-batches", str(max_batches)]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)  # noqa: S603

    if result.returncode != 0:
        # stderr was not captured, it already printed to terminal
        raise RuntimeError(
            f"Benchmark script exited with code {result.returncode} for checkpoint {checkpoint}"
        )

    output = result.stdout
    print(output, end="")  # echo captured stdout to terminal
    return parse_perplexity(output)


def main() -> None:
    args = _parse_args()

    # Resolve benchmark script path.
    benchmark_script = args.benchmark_script
    if benchmark_script is None:
        benchmark_script = Path(__file__).parent / "benchmark.py"

    if not benchmark_script.exists():
        print(f"error: benchmark script not found: {benchmark_script}", file=sys.stderr)
        sys.exit(1)

    # Validate checkpoint paths (unless dry-run).
    if not args.dry_run:
        for label, ckpt in [
            (f"Team A ({args.team_a_id})", args.team_a_checkpoint),
            (f"Team B ({args.team_b_id})", args.team_b_checkpoint),
        ]:
            if not ckpt.exists():
                print(f"error: {label} checkpoint not found: {ckpt}", file=sys.stderr)
                sys.exit(1)

    output_path = args.output or Path("competition_result.json")

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("=" * 64)
    print("  Swarm-Tune Competition — Phase 8")
    print("=" * 64)
    if args.competition_id:
        print(f"  Competition: {args.competition_id}")
    print(f"  Team A: {args.team_a_id}  ({args.team_a_checkpoint})")
    print(f"  Team B: {args.team_b_id}  ({args.team_b_checkpoint})")
    print(f"  Model:  {args.model_name}")
    print(f"  Device: {args.device}")
    print()

    # ------------------------------------------------------------------
    # Benchmark Team A
    # ------------------------------------------------------------------
    if args.dry_run:
        print(f"[dry-run] Skipping benchmark for {args.team_a_id} — using perplexity 1.0")
        team_a_ppl = 1.0
    else:
        print(f"Benchmarking Team A: {args.team_a_id}")
        print("-" * 48)
        try:
            team_a_ppl = _benchmark(
                args.team_a_checkpoint,
                args.model_name,
                benchmark_script,
                args.device,
                args.max_batches,
            )
        except (RuntimeError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Team A perplexity: {team_a_ppl:.4f}")
        print()

    # ------------------------------------------------------------------
    # Benchmark Team B
    # ------------------------------------------------------------------
    if args.dry_run:
        print(f"[dry-run] Skipping benchmark for {args.team_b_id} — using perplexity 1.0")
        team_b_ppl = 1.0
    else:
        print(f"Benchmarking Team B: {args.team_b_id}")
        print("-" * 48)
        try:
            team_b_ppl = _benchmark(
                args.team_b_checkpoint,
                args.model_name,
                benchmark_script,
                args.device,
                args.max_batches,
            )
        except (RuntimeError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Team B perplexity: {team_b_ppl:.4f}")
        print()

    # ------------------------------------------------------------------
    # Determine winner and write results
    # ------------------------------------------------------------------
    from swarm_tune.runs.competition import make_result

    result = make_result(
        competition_id=args.competition_id,
        team_a_id=args.team_a_id,
        team_a_ppl=team_a_ppl,
        team_a_checkpoint=str(args.team_a_checkpoint),
        team_b_id=args.team_b_id,
        team_b_ppl=team_b_ppl,
        team_b_checkpoint=str(args.team_b_checkpoint),
        tie_tolerance=args.tie_tolerance,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print(f"  {args.team_a_id:<20} perplexity: {team_a_ppl:.4f}")
    print(f"  {args.team_b_id:<20} perplexity: {team_b_ppl:.4f}")
    print()

    winner = result["winner"]
    if winner == "tie":
        print(
            f"  RESULT: TIE  (difference ≤ {args.tie_tolerance})\n"
            f"  Both teams are within measurement noise of each other."
        )
    else:
        loser = args.team_b_id if winner == args.team_a_id else args.team_a_id
        winner_ppl = team_a_ppl if winner == args.team_a_id else team_b_ppl
        loser_ppl = team_b_ppl if winner == args.team_a_id else team_a_ppl
        margin = abs(team_a_ppl - team_b_ppl)
        print(f"  WINNER: {winner}")
        print(f"    {winner}: {winner_ppl:.4f}  vs  {loser}: {loser_ppl:.4f}")
        print(f"    Margin: {margin:.4f} perplexity points")

    print()
    print(f"  Results written to: {output_path}")
    print()
    print(
        "  Anyone can independently verify:\n"
        f"    make benchmark CHECKPOINT=<path> MODEL={args.model_name}"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
