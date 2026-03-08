"""Unit tests for the RunManifest model."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from swarm_tune.runs.manifest import RunManifest


@pytest.mark.unit
class TestRunManifest:
    def _make_manifest(self, **overrides: object) -> RunManifest:
        defaults = {
            "run_id": "test-run-001",
            "model_name": "gpt2",
            "dataset_name": "wikitext",
            "dataset_config": "wikitext-103-raw-v1",
            "num_shards": 4,
            "num_rounds": 50,
            "min_peers": 2,
        }
        defaults.update(overrides)
        return RunManifest(**defaults)  # type: ignore[arg-type]

    def test_load_from_json(self, tmp_path: Path) -> None:
        data = {
            "run_id": "load-test-001",
            "num_shards": 3,
            "num_rounds": 10,
        }
        f = tmp_path / "load-test-001.json"
        f.write_text(json.dumps(data))
        m = RunManifest.load(f)
        assert m.run_id == "load-test-001"
        assert m.num_shards == 3
        assert m.num_rounds == 10
        assert m.model_name == "gpt2"  # default

    def test_load_by_id(self, tmp_path: Path) -> None:
        data = {"run_id": "my-run-001", "num_shards": 2}
        (tmp_path / "my-run-001.json").write_text(json.dumps(data))
        m = RunManifest.load_by_id("my-run-001", runs_dir=tmp_path)
        assert m.run_id == "my-run-001"

    def test_load_by_id_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no-such-run"):
            RunManifest.load_by_id("no-such-run", runs_dir=tmp_path)

    def test_to_env_correct_shard_assignment(self) -> None:
        m = self._make_manifest(num_shards=4)
        for idx in range(4):
            env = m.to_env(idx, port=9000 + idx)
            assert env["SWARM_DATA_SHARD_INDEX"] == str(idx)
            assert env["SWARM_DATA_SHARD_TOTAL"] == "4"
            assert env["SWARM_NODE_ID"] == f"node_{idx}"
            assert env["SWARM_PORT"] == str(9000 + idx)

    def test_to_env_model_sharding(self) -> None:
        m = self._make_manifest(num_shards=4, model_shard_total=2)
        assert m.to_env(0)["SWARM_MODEL_SHARD_INDEX"] == "0"
        assert m.to_env(1)["SWARM_MODEL_SHARD_INDEX"] == "1"
        assert m.to_env(2)["SWARM_MODEL_SHARD_INDEX"] == "0"  # 2 % 2 == 0
        assert m.to_env(3)["SWARM_MODEL_SHARD_INDEX"] == "1"  # 3 % 2 == 1

    def test_to_env_out_of_range(self) -> None:
        m = self._make_manifest(num_shards=4)
        with pytest.raises(ValueError, match="out of range"):
            m.to_env(4)
        with pytest.raises(ValueError, match="out of range"):
            m.to_env(-1)

    def test_to_env_bootstrap_peers(self) -> None:
        m = self._make_manifest(bootstrap_peers=["/ip4/1.2.3.4/tcp/9000/p2p/abc"])
        env = m.to_env(1)
        assert "SWARM_BOOTSTRAP_PEERS" in env
        decoded = json.loads(env["SWARM_BOOTSTRAP_PEERS"])
        assert decoded == ["/ip4/1.2.3.4/tcp/9000/p2p/abc"]

    def test_to_env_no_bootstrap_when_empty(self) -> None:
        m = self._make_manifest()
        env = m.to_env(0)
        assert "SWARM_BOOTSTRAP_PEERS" not in env

    def test_write_env_file(self, tmp_path: Path) -> None:
        m = self._make_manifest()
        out = tmp_path / "test.env"
        m.write_env_file(out, node_index=1, port=9001, extra={"SWARM_DEVICE": "cpu"})
        content = out.read_text()
        assert "SWARM_DATA_SHARD_INDEX=1" in content
        assert "SWARM_DATA_SHARD_TOTAL=4" in content
        assert "SWARM_PORT=9001" in content
        assert "SWARM_DEVICE=cpu" in content
        # Comments should be present
        assert "# Swarm-Tune" in content
        # Should NOT share private commentary
        assert "DO NOT share" in content

    def test_write_env_file_no_keys_in_comments(self, tmp_path: Path) -> None:
        """The .env file must not expose security-sensitive values in comments."""
        m = self._make_manifest()
        out = tmp_path / "test.env"
        m.write_env_file(out, node_index=0, port=9000)
        lines = out.read_text().splitlines()
        comment_lines = [ln for ln in lines if ln.startswith("#")]
        for line in comment_lines:
            assert "SWARM_" not in line, f"Secret key in comment: {line}"

    def test_competition_manifest(self) -> None:
        m = self._make_manifest(
            competition_id="comp-001",
            team_id="team-alpha",
            num_rounds=50,
        )
        assert m.competition_id == "comp-001"
        assert m.team_id == "team-alpha"

    def test_load_real_gpt2_manifest(self) -> None:
        """Smoke test: ensure the bundled gpt2-wikitrain-001.json is valid."""
        m = RunManifest.load_by_id("gpt2-wikitrain-001")
        assert m.run_id == "gpt2-wikitrain-001"
        assert m.model_name == "gpt2"
        assert m.num_shards == 4
        assert m.num_rounds == 100

    def test_write_env_file_is_parseable(self, tmp_path: Path) -> None:
        """Every non-comment line in the .env must be in KEY=value format."""
        m = self._make_manifest()
        out = tmp_path / "parse.env"
        m.write_env_file(out, node_index=0)
        for line in out.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assert "=" in stripped, f"Non-KV line in .env: {stripped!r}"
            key, _, _ = stripped.partition("=")
            assert key.isupper() or key.startswith("SWARM_"), f"Unexpected key format: {key!r}"
