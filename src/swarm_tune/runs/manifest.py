"""
Run Manifest — Training Campaign Definition

A RunManifest is a JSON file that fully describes a Swarm-Tune training
campaign. Two participants with the same manifest file and different
node_index values will train on non-overlapping data shards of the same
model with zero manual configuration.

Run ID format: <model>-<dataset>-<NNN>
  Examples:
    gpt2-wikitrain-001     — GPT-2 data-parallel on WikiText-103
    gpt2-competition-001   — competition run with teams

Manifest files live in the top-level runs/ directory and are checked into
the repository so participants can reference them by run_id without needing
a central server.

Typical flow:
    python scripts/join.py --run-id gpt2-wikitrain-001 --node-index 2

This:
  1. Loads  runs/gpt2-wikitrain-001.json
  2. Produces an .env file with SWARM_DATA_SHARD_INDEX=2, SWARM_DATA_SHARD_TOTAL=4, ...
  3. Prints the docker run command (or starts the node directly)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Sentinel: default search path relative to repo root.
_DEFAULT_RUNS_DIR = Path(__file__).parent.parent.parent.parent / "runs"


class RunManifest(BaseModel):
    """
    Definition of a Swarm-Tune training campaign.

    All fields map directly to SWARM_ environment variables via to_env().
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    run_id: str = Field(description="Unique identifier, e.g. 'gpt2-wikitrain-001'.")
    version: int = Field(default=1, description="Manifest schema version.")
    description: str = Field(default="", description="Human-readable campaign description.")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_name: str = Field(
        default="gpt2",
        description="HuggingFace model name or 'mlp' for the toy simulation model.",
    )
    model_shard_total: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of model shards (model parallelism). "
            "1 = data-parallel only (all nodes train the same layers). "
            ">1 = each node trains layer_i where i % model_shard_total == shard_index."
        ),
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_name: str = Field(
        default="wikitext",
        description="HuggingFace dataset name (e.g. 'wikitext', 'openwebtext').",
    )
    dataset_config: str = Field(
        default="wikitext-103-raw-v1",
        description="HuggingFace dataset config / subset (e.g. 'wikitext-103-raw-v1').",
    )
    max_seq_len: int = Field(
        default=512,
        ge=1,
        description="Tokenized sequence length per training sample.",
    )

    # ------------------------------------------------------------------
    # Swarm topology
    # ------------------------------------------------------------------
    num_shards: int = Field(
        default=4,
        ge=1,
        description=(
            "Expected number of data-parallel participants. "
            "Each node trains on 1/num_shards of the dataset. "
            "More participants = more unique training data per round."
        ),
    )
    min_peers: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum peer responses required to commit a round. "
            "If fewer peers respond within aggregation_timeout_secs, the round is deferred."
        ),
    )
    bootstrap_peers: list[str] = Field(
        default_factory=list,
        description=(
            "Multiaddresses of bootstrap peers for this run. "
            "Empty = node_index=0 acts as bootstrap; other nodes must set SWARM_BOOTSTRAP_PEERS "
            "to node_0's advertised address after it starts. "
            "For production runs, populate with a stable relay or bootstrap node address."
        ),
    )

    # ------------------------------------------------------------------
    # Training hyperparameters
    # ------------------------------------------------------------------
    num_rounds: int = Field(default=100, ge=1, description="Total training rounds per node.")
    learning_rate: float = Field(default=1e-4, gt=0.0, description="AdamW learning rate.")
    batch_size: int = Field(default=8, ge=1, description="Local mini-batch size per round.")
    aggregation_timeout_secs: float = Field(
        default=30.0,
        gt=0.0,
        description="Seconds to collect peer gradients before partial aggregation.",
    )

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------
    compression: Literal["none", "topk"] = Field(
        default="none",
        description=(
            "'none' = no compression (good for < 20 nodes on fast connections). "
            "'topk' = Top-K sparsification (~50x bandwidth reduction at k=0.01). "
            "Swap to 'topk' when bandwidth is the bottleneck."
        ),
    )
    topk_ratio: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Fraction of gradient elements kept when compression='topk'.",
    )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    checkpoint_every_n_rounds: int = Field(
        default=10,
        ge=0,
        description="Save a checkpoint every N rounds. 0 = final checkpoint only.",
    )

    # ------------------------------------------------------------------
    # Competition fields (populated only in competition manifests)
    # ------------------------------------------------------------------
    competition_id: str = Field(
        default="",
        description="Competition identifier (empty for non-competition runs).",
    )
    team_id: str = Field(
        default="",
        description="Team identifier within the competition (empty for solo runs).",
    )

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> RunManifest:
        """Load and validate a RunManifest from a JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        data: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Manifest {path} must be a JSON object, got {type(data).__name__}")
        return cls(**data)

    @classmethod
    def load_by_id(cls, run_id: str, runs_dir: Path | None = None) -> RunManifest:
        """
        Load a manifest by run_id from the runs/ directory.

        Args:
            run_id:   e.g. 'gpt2-wikitrain-001'
            runs_dir: directory containing manifest JSON files.
                      Defaults to <repo_root>/runs/.

        Raises:
            FileNotFoundError: if no manifest exists for the given run_id.
        """
        search_dir = runs_dir if runs_dir is not None else _DEFAULT_RUNS_DIR
        path = search_dir / f"{run_id}.json"

        if not path.exists():
            available = [p.stem for p in search_dir.glob("*.json")] if search_dir.exists() else []
            available_str = ", ".join(sorted(available)) if available else "(none found)"
            raise FileNotFoundError(
                f"No manifest found for run_id '{run_id}'.\n"
                f"Expected: {path}\n"
                f"Available runs: {available_str}"
            )

        return cls.load(path)

    # ------------------------------------------------------------------
    # Env generation
    # ------------------------------------------------------------------
    def to_env(self, node_index: int, port: int = 9000) -> dict[str, str]:
        """
        Generate SWARM_ environment variables for a specific participant.

        Args:
            node_index: which data shard this node trains on (0-based, unique per node).
                        Must be in range [0, num_shards).
            port:       TCP port for the libp2p listener (default 9000).

        Returns:
            Mapping of env var name → string value, ready for use as an .env file.
        """
        if not (0 <= node_index < self.num_shards):
            raise ValueError(
                f"node_index={node_index} is out of range for run '{self.run_id}' "
                f"which has num_shards={self.num_shards}. "
                f"Valid range: 0 to {self.num_shards - 1}."
            )

        env: dict[str, str] = {
            # Identity
            "SWARM_NODE_ID": f"node_{node_index}",
            "SWARM_PORT": str(port),
            # Model
            "SWARM_MODEL_NAME": self.model_name,
            "SWARM_MODEL_SHARD_INDEX": str(node_index % self.model_shard_total),
            "SWARM_MODEL_SHARD_TOTAL": str(self.model_shard_total),
            # Dataset
            "SWARM_DATASET_NAME": self.dataset_name,
            "SWARM_DATASET_CONFIG": self.dataset_config,
            "SWARM_MAX_SEQ_LEN": str(self.max_seq_len),
            # Data sharding (unique per participant)
            "SWARM_DATA_SHARD_INDEX": str(node_index),
            "SWARM_DATA_SHARD_TOTAL": str(self.num_shards),
            # Training hyperparameters
            "SWARM_NUM_ROUNDS": str(self.num_rounds),
            "SWARM_LEARNING_RATE": str(self.learning_rate),
            "SWARM_BATCH_SIZE": str(self.batch_size),
            "SWARM_MIN_PEERS_FOR_ROUND": str(self.min_peers),
            "SWARM_AGGREGATION_TIMEOUT_SECS": str(self.aggregation_timeout_secs),
            # Compression
            "SWARM_COMPRESSION": self.compression,
            "SWARM_TOPK_RATIO": str(self.topk_ratio),
            # Checkpointing
            "SWARM_CHECKPOINT_EVERY_N_ROUNDS": str(self.checkpoint_every_n_rounds),
            # Logging (JSON is production-default; scripts/join.py lets the user override)
            "SWARM_LOG_FORMAT": "json",
            "SWARM_LOG_LEVEL": "INFO",
        }

        # Bootstrap peers: if the manifest specifies them, encode as JSON list.
        # If empty, node_index=0 has no bootstrap (it IS the bootstrap);
        # all other nodes must have their bootstrap set separately.
        if self.bootstrap_peers:
            env["SWARM_BOOTSTRAP_PEERS"] = json.dumps(self.bootstrap_peers)

        return env

    def write_env_file(
        self,
        path: Path,
        node_index: int,
        port: int = 9000,
        extra: dict[str, str] | None = None,
    ) -> None:
        """
        Write a .env file for the given node_index.

        The file format is:
            KEY=value
            # comment lines preserved at the top

        Args:
            path:       destination file path (e.g. Path("my.env")).
            node_index: which data shard this node trains on.
            port:       TCP port for libp2p.
            extra:      additional env vars to append (e.g. SWARM_DEVICE=mps).
        """
        env = self.to_env(node_index, port=port)
        if extra:
            env.update(extra)

        lines: list[str] = [
            f"# Swarm-Tune — auto-generated from run manifest '{self.run_id}'",
            f"# Node index: {node_index} / {self.num_shards}",
            f"# Description: {self.description}",
            "#",
            "# DO NOT share this file — it contains your node_index assignment.",
            "# Other participants must run scripts/join.py with their own node_index.",
            "#",
        ]
        for key, value in env.items():
            lines.append(f"{key}={value}")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
