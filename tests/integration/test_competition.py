"""
Integration tests for Phase 8 competition orchestration.

Tests wire together the real RunManifest + competition module and exercise
the run_competition.py script end-to-end via subprocess (no actual model
downloads — the fake benchmark script echoes a fixed perplexity).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from swarm_tune.runs.manifest import RunManifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fake_benchmark(path: Path, perplexity: float) -> None:
    """Write a minimal benchmark substitute that prints a fixed perplexity."""
    path.write_text(
        textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import argparse, sys
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--model-name", default="gpt2")
        p.add_argument("--device", default="cpu")
        p.add_argument("--max-batches", type=int, default=None)
        args = p.parse_args()
        if not __import__("pathlib").Path(args.checkpoint).exists():
            print(f"error: checkpoint not found: {{args.checkpoint}}", file=sys.stderr)
            sys.exit(1)
        print("Fake benchmark running...")
        print(f"perplexity: {perplexity:.2f}")
        """),
        encoding="utf-8",
    )


def _write_dummy_checkpoint(path: Path) -> None:
    """Write a minimal torch checkpoint (just a dict) so the path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dummy": torch.tensor([1.0])}, path)


def _run_competition(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run scripts/run_competition.py with the given args and return the result."""
    script = Path(__file__).parent.parent.parent / "scripts" / "run_competition.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCompetitionManifests:
    def test_gpt2_competition_001_is_valid(self) -> None:
        """The bundled 4-node competition manifest loads correctly."""
        m = RunManifest.load_by_id("gpt2-competition-001")
        assert m.competition_id == "gpt2-competition-001"
        assert m.num_shards == 4
        assert m.num_rounds == 50
        assert m.model_name == "gpt2"

    def test_gpt2_competition_2v2_is_valid(self) -> None:
        """The 2-node competition manifest loads and has correct fields."""
        m = RunManifest.load_by_id("gpt2-competition-2v2")
        assert m.competition_id == "gpt2-competition-2v2"
        assert m.num_shards == 2
        assert m.num_rounds == 50
        assert m.min_peers == 1

    def test_two_teams_same_competition_different_team_ids(self) -> None:
        """Two teams with the same competition_id but different team_ids are valid."""
        m_alpha = RunManifest(
            run_id="gpt2-competition-001",
            competition_id="gpt2-competition-001",
            team_id="team-alpha",
            num_shards=4,
        )
        m_beta = RunManifest(
            run_id="gpt2-competition-001",
            competition_id="gpt2-competition-001",
            team_id="team-beta",
            num_shards=4,
        )
        assert m_alpha.competition_id == m_beta.competition_id
        assert m_alpha.team_id != m_beta.team_id

    def test_team_env_includes_node_assignment(self) -> None:
        """to_env() correctly assigns each node in the competition manifest."""
        m = RunManifest.load_by_id("gpt2-competition-2v2")
        for idx in range(m.num_shards):
            env = m.to_env(idx)
            assert env["SWARM_DATA_SHARD_INDEX"] == str(idx)
            assert env["SWARM_DATA_SHARD_TOTAL"] == "2"

    def test_2v2_manifest_hyperparameters_locked(self) -> None:
        """Competition manifest values are locked to prevent accidental changes."""
        m = RunManifest.load_by_id("gpt2-competition-2v2")
        assert m.learning_rate == pytest.approx(1e-4)
        assert m.batch_size == 8
        assert m.aggregation_timeout_secs == pytest.approx(30.0)
        assert m.compression == "none"


# ---------------------------------------------------------------------------
# Script tests via subprocess
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunCompetitionScript:
    def test_dry_run_succeeds(self, tmp_path: Path) -> None:
        """--dry-run exits 0 and writes a results JSON without real benchmarking."""
        ckpt_a = tmp_path / "alpha.pt"
        ckpt_b = tmp_path / "beta.pt"
        _write_dummy_checkpoint(ckpt_a)
        _write_dummy_checkpoint(ckpt_b)
        out = tmp_path / "result.json"

        result = _run_competition(
            [
                "--team-a-id",
                "alpha",
                "--team-a-checkpoint",
                str(ckpt_a),
                "--team-b-id",
                "beta",
                "--team-b-checkpoint",
                str(ckpt_b),
                "--output",
                str(out),
                "--dry-run",
            ]
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()

    def test_dry_run_result_json_is_valid(self, tmp_path: Path) -> None:
        ckpt_a = tmp_path / "alpha.pt"
        ckpt_b = tmp_path / "beta.pt"
        _write_dummy_checkpoint(ckpt_a)
        _write_dummy_checkpoint(ckpt_b)
        out = tmp_path / "result.json"

        _run_competition(
            [
                "--team-a-id",
                "alpha",
                "--team-a-checkpoint",
                str(ckpt_a),
                "--team-b-id",
                "beta",
                "--team-b-checkpoint",
                str(ckpt_b),
                "--output",
                str(out),
                "--dry-run",
            ]
        )

        data = json.loads(out.read_text())
        assert "competition_id" in data
        assert "teams" in data
        assert "winner" in data  # "tie" — both use placeholder 1.0
        assert "verified_at" in data
        assert "alpha" in data["teams"]
        assert "beta" in data["teams"]

    def test_fake_benchmark_alpha_wins(self, tmp_path: Path) -> None:
        """Team with lower perplexity wins — uses a fake benchmark script."""
        fake_bench_a = tmp_path / "bench_a.py"
        fake_bench_b = tmp_path / "bench_b.py"
        _write_fake_benchmark(fake_bench_a, perplexity=42.0)  # alpha is better
        _write_fake_benchmark(fake_bench_b, perplexity=60.0)

        ckpt_a = tmp_path / "alpha.pt"
        ckpt_b = tmp_path / "beta.pt"
        _write_dummy_checkpoint(ckpt_a)
        _write_dummy_checkpoint(ckpt_b)
        out = tmp_path / "result.json"

        # We need a single benchmark script — write one that dispatches by checkpoint path.
        fake_bench = tmp_path / "bench.py"
        fake_bench.write_text(
            textwrap.dedent("""\
            #!/usr/bin/env python3
            import argparse, sys
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--model-name", default="gpt2")
            p.add_argument("--device", default="cpu")
            p.add_argument("--max-batches", type=int, default=None)
            args = p.parse_args()
            if not Path(args.checkpoint).exists():
                print("error: not found", file=sys.stderr); sys.exit(1)
            # Match on filename only (not full path) to avoid false matches on
            # the tmp_path directory name which also contains "alpha".
            fname = Path(args.checkpoint).name
            if "alpha" in fname:
                print("perplexity: 42.00")
            else:
                print("perplexity: 60.00")
            """),
            encoding="utf-8",
        )

        result = _run_competition(
            [
                "--team-a-id",
                "alpha",
                "--team-a-checkpoint",
                str(ckpt_a),
                "--team-b-id",
                "beta",
                "--team-b-checkpoint",
                str(ckpt_b),
                "--benchmark-script",
                str(fake_bench),
                "--output",
                str(out),
            ]
        )
        assert result.returncode == 0, result.stderr

        data = json.loads(out.read_text())
        assert data["winner"] == "alpha"
        assert data["teams"]["alpha"]["perplexity"] == pytest.approx(42.0)
        assert data["teams"]["beta"]["perplexity"] == pytest.approx(60.0)

    def test_fake_benchmark_beta_wins(self, tmp_path: Path) -> None:
        ckpt_a = tmp_path / "alpha.pt"
        ckpt_b = tmp_path / "beta.pt"
        _write_dummy_checkpoint(ckpt_a)
        _write_dummy_checkpoint(ckpt_b)
        out = tmp_path / "result.json"

        fake_bench = tmp_path / "bench.py"
        fake_bench.write_text(
            textwrap.dedent("""\
            #!/usr/bin/env python3
            import argparse, sys
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--model-name", default="gpt2")
            p.add_argument("--device", default="cpu")
            p.add_argument("--max-batches", type=int, default=None)
            args = p.parse_args()
            if not Path(args.checkpoint).exists():
                print("error: not found", file=sys.stderr); sys.exit(1)
            if "alpha" in args.checkpoint:
                print("perplexity: 80.00")
            else:
                print("perplexity: 35.00")
            """),
            encoding="utf-8",
        )

        result = _run_competition(
            [
                "--team-a-id",
                "alpha",
                "--team-a-checkpoint",
                str(ckpt_a),
                "--team-b-id",
                "beta",
                "--team-b-checkpoint",
                str(ckpt_b),
                "--benchmark-script",
                str(fake_bench),
                "--output",
                str(out),
            ]
        )
        assert result.returncode == 0, result.stderr

        data = json.loads(out.read_text())
        assert data["winner"] == "beta"

    def test_tie_when_perplexity_difference_small(self, tmp_path: Path) -> None:
        ckpt_a = tmp_path / "alpha.pt"
        ckpt_b = tmp_path / "beta.pt"
        _write_dummy_checkpoint(ckpt_a)
        _write_dummy_checkpoint(ckpt_b)
        out = tmp_path / "result.json"

        fake_bench = tmp_path / "bench.py"
        fake_bench.write_text(
            textwrap.dedent("""\
            #!/usr/bin/env python3
            import argparse, sys
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--model-name", default="gpt2")
            p.add_argument("--device", default="cpu")
            p.add_argument("--max-batches", type=int, default=None)
            args = p.parse_args()
            if not Path(args.checkpoint).exists():
                print("error: not found", file=sys.stderr); sys.exit(1)
            # Both teams within 0.3 of each other → should be a tie
            if "alpha" in args.checkpoint:
                print("perplexity: 50.10")
            else:
                print("perplexity: 50.30")
            """),
            encoding="utf-8",
        )

        result = _run_competition(
            [
                "--team-a-id",
                "alpha",
                "--team-a-checkpoint",
                str(ckpt_a),
                "--team-b-id",
                "beta",
                "--team-b-checkpoint",
                str(ckpt_b),
                "--benchmark-script",
                str(fake_bench),
                "--tie-tolerance",
                "0.5",
                "--output",
                str(out),
            ]
        )
        assert result.returncode == 0, result.stderr

        data = json.loads(out.read_text())
        assert data["winner"] == "tie"

    def test_missing_checkpoint_exits_nonzero(self, tmp_path: Path) -> None:
        out = tmp_path / "result.json"
        result = _run_competition(
            [
                "--team-a-id",
                "alpha",
                "--team-a-checkpoint",
                str(tmp_path / "nonexistent_alpha.pt"),
                "--team-b-id",
                "beta",
                "--team-b-checkpoint",
                str(tmp_path / "nonexistent_beta.pt"),
                "--output",
                str(out),
            ]
        )
        assert result.returncode != 0

    def test_competition_id_preserved_in_output(self, tmp_path: Path) -> None:
        ckpt_a = tmp_path / "alpha.pt"
        ckpt_b = tmp_path / "beta.pt"
        _write_dummy_checkpoint(ckpt_a)
        _write_dummy_checkpoint(ckpt_b)
        out = tmp_path / "result.json"

        _run_competition(
            [
                "--competition-id",
                "gpt2-competition-2v2",
                "--team-a-id",
                "alpha",
                "--team-a-checkpoint",
                str(ckpt_a),
                "--team-b-id",
                "beta",
                "--team-b-checkpoint",
                str(ckpt_b),
                "--output",
                str(out),
                "--dry-run",
            ]
        )

        data = json.loads(out.read_text())
        assert data["competition_id"] == "gpt2-competition-2v2"

    def test_output_created_in_nested_dir(self, tmp_path: Path) -> None:
        """Output path with non-existent parent directories should work."""
        ckpt_a = tmp_path / "alpha.pt"
        ckpt_b = tmp_path / "beta.pt"
        _write_dummy_checkpoint(ckpt_a)
        _write_dummy_checkpoint(ckpt_b)
        out = tmp_path / "results" / "nested" / "result.json"

        result = _run_competition(
            [
                "--team-a-id",
                "alpha",
                "--team-a-checkpoint",
                str(ckpt_a),
                "--team-b-id",
                "beta",
                "--team-b-checkpoint",
                str(ckpt_b),
                "--output",
                str(out),
                "--dry-run",
            ]
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()


# ---------------------------------------------------------------------------
# End-to-end: two swarms produce checkpoints, winner is determined
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEndToEndCompetition:
    """
    Simulates the full Phase 8 flow without a real network:
      1. Two 'swarms' (single-node each) train an MLP for a few rounds
      2. Each saves a checkpoint
      3. run_competition.py compares them with a fake benchmark
      4. Winner is correctly identified
    """

    def test_two_swarms_produce_comparable_checkpoints(self, tmp_path: Path) -> None:
        """Each team's checkpoint path and perplexity is captured in the result."""
        from swarm_tune.config.settings import NodeSettings
        from swarm_tune.node.trainer.gradient import GradientExtractor
        from swarm_tune.node.trainer.model import ModelShard

        # Train two minimal MLP swarms independently for 3 rounds each.
        checkpoints: list[Path] = []
        for team_idx in range(2):
            settings = NodeSettings(
                node_id=f"team_{team_idx}_node_0",
                host="127.0.0.1",
                port=19100 + team_idx,
                bootstrap_peers=[],
                model_name="mlp",
                learning_rate=1e-3,
                batch_size=4,
                num_rounds=3,
                aggregation_timeout_secs=2.0,
                min_peers_for_round=1,
                device="cpu",
                log_level="WARNING",
                log_format="console",
            )
            shard = ModelShard(settings)
            shard.load()
            extractor = GradientExtractor()

            for _ in range(3):
                inputs = torch.randn(4, 128)
                output = shard.forward(inputs)
                loss = torch.nn.functional.mse_loss(output, inputs)
                shard.backward(loss)
                grads = extractor.extract(shard.model)
                shard.apply_averaged_gradients(grads)

            ckpt_path = tmp_path / f"team_{team_idx}_final.pt"
            torch.save(shard.model.state_dict(), ckpt_path)
            checkpoints.append(ckpt_path)

        assert all(ckpt.exists() for ckpt in checkpoints)

        # Wire up competition coordinator with a fake benchmark script.
        fake_bench = tmp_path / "bench.py"
        # Team 0 has lower perplexity (wins).
        fake_bench.write_text(
            textwrap.dedent("""\
            #!/usr/bin/env python3
            import argparse, sys
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--model-name", default="gpt2")
            p.add_argument("--device", default="cpu")
            p.add_argument("--max-batches", type=int, default=None)
            args = p.parse_args()
            if not Path(args.checkpoint).exists():
                print("error: not found", file=sys.stderr); sys.exit(1)
            if "team_0" in args.checkpoint:
                print("perplexity: 38.50")
            else:
                print("perplexity: 52.10")
            """),
            encoding="utf-8",
        )

        out = tmp_path / "competition_result.json"
        result = _run_competition(
            [
                "--competition-id",
                "gpt2-competition-2v2",
                "--team-a-id",
                "team-0",
                "--team-a-checkpoint",
                str(checkpoints[0]),
                "--team-b-id",
                "team-1",
                "--team-b-checkpoint",
                str(checkpoints[1]),
                "--benchmark-script",
                str(fake_bench),
                "--output",
                str(out),
            ]
        )

        assert result.returncode == 0, result.stderr
        data = json.loads(out.read_text())

        assert data["winner"] == "team-0"
        assert data["competition_id"] == "gpt2-competition-2v2"
        assert data["teams"]["team-0"]["perplexity"] == pytest.approx(38.50)
        assert data["teams"]["team-1"]["perplexity"] == pytest.approx(52.10)
