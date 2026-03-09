"""
Central configuration for a Swarm-Tune node.

All values are read from environment variables (with SWARM_ prefix)
and can be overridden via a YAML config file. Pydantic validates
and coerces every field at startup — bad config fails loudly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class NodeSettings(BaseSettings):
    """Runtime configuration for a single Swarm-Tune node."""

    model_config = SettingsConfigDict(
        env_prefix="SWARM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_ignore_empty=True,  # "" in env var == not set; avoids JSON-decode crash on list fields
    )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    node_id: str = Field(
        default="",
        description="Unique human-readable node identifier. Auto-generated if empty.",
    )
    node_key_seed: str = Field(
        default="",
        repr=False,  # Never include in str()/repr() — prevents accidental log exposure.
        description=(
            "If set, derive the Ed25519 key pair deterministically from this seed. "
            "Produces a stable peer ID across restarts — required when other nodes "
            "need to hard-code this node's bootstrap address. "
            "Leave empty for a random key pair (default for non-bootstrap nodes)."
        ),
    )

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------
    host: str = Field(default="0.0.0.0", description="Interface to bind the libp2p listener.")
    port: int = Field(default=9000, ge=1024, le=65535, description="TCP port for libp2p.")
    bootstrap_peers: list[str] = Field(
        default_factory=list,
        description="Multiaddresses of bootstrap peers. Empty for the first bootstrap node.",
    )

    # ------------------------------------------------------------------
    # Data & Model
    # ------------------------------------------------------------------
    data_shard_path: Path = Field(
        default=Path("data/shards/shard_0.pt"),
        description=(
            "Path to this node's local .pt training data shard. "
            "Used only when dataset_name is empty."
        ),
    )
    model_name: str = Field(
        default="gpt2",
        description=(
            "HuggingFace model name or local path to load. "
            "Use 'mlp' for the toy MLP (unit tests / Phase 1-4 simulation)."
        ),
    )
    model_shard_index: int = Field(
        default=0,
        ge=0,
        description="Index of the model shard this node is responsible for.",
    )
    model_shard_total: int = Field(
        default=1,
        ge=1,
        description="Total number of model shards across the swarm.",
    )

    # HuggingFace dataset (Phase 5+). When dataset_name is empty the node
    # falls back to loading from data_shard_path (.pt file).
    dataset_name: str = Field(
        default="",
        description=(
            "HuggingFace dataset name (e.g. 'wikitext'). "
            "Empty = use data_shard_path .pt file (Phase 1-4 mode)."
        ),
    )
    dataset_config: str = Field(
        default="wikitext-103-raw-v1",
        description="HuggingFace dataset config / subset name.",
    )
    max_seq_len: int = Field(
        default=512,
        ge=1,
        description="Tokenized sequence length per sample for HF datasets.",
    )
    # Data shard assignment — SEPARATE from model shard assignment.
    # Each node in the swarm must have a unique data_shard_index so nodes
    # train on non-overlapping subsets of the dataset. This is independent
    # of model_shard_index (which controls layer parallelism).
    # Bug fix: using model_shard_index for data sharding caused all nodes
    # with the default model_shard_total=1 to load identical data (index=0).
    data_shard_index: int = Field(
        default=0,
        ge=0,
        description=(
            "Which data shard this node trains on (0-based). "
            "Must be unique per node in the swarm. "
            "Independent of model_shard_index (layer parallelism)."
        ),
    )
    data_shard_total: int = Field(
        default=1,
        ge=1,
        description=(
            "Total number of data shards (= number of nodes training in parallel). "
            "Set to the swarm size so each node gets a non-overlapping data slice."
        ),
    )

    # ------------------------------------------------------------------
    # Training hyperparameters
    # ------------------------------------------------------------------
    learning_rate: float = Field(default=1e-4, gt=0, description="SGD / Adam learning rate.")
    batch_size: int = Field(default=8, ge=1, description="Local mini-batch size per round.")
    num_rounds: int = Field(default=100, ge=1, description="Total number of training rounds.")

    # ------------------------------------------------------------------
    # Gradient validation
    # ------------------------------------------------------------------
    gradient_max_norm_rms: float = Field(
        default=10.0,
        gt=0.0,
        description=(
            "Per-element RMS norm threshold for gradient validation. "
            "Computed as tensor.norm() / sqrt(tensor.numel()). "
            "For standard-normal initialised models, this is ~1.0 at the start. "
            "10.0 allows healthy fine-tuning dynamics while rejecting poisoned gradients. "
            "Replaces the old absolute L2 norm threshold which incorrectly rejected "
            "legitimate large-parameter-count tensors (e.g. LLaMA embedding tables)."
        ),
    )

    # ------------------------------------------------------------------
    # Aggregation / fault tolerance
    # ------------------------------------------------------------------
    aggregation_timeout_secs: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Seconds to wait for peer gradients before performing partial aggregation. "
            "A node that misses this window is skipped — it NEVER blocks the swarm."
        ),
    )
    min_peers_for_round: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum number of peer gradient responses required to commit a round. "
            "If fewer peers respond within the timeout, the round is deferred."
        ),
    )

    # ------------------------------------------------------------------
    # Sybil resistance (Phase 5+ — required before public internet deployment)
    # ------------------------------------------------------------------
    enable_sybil_resistance: bool = Field(
        default=False,
        description=(
            "Enable Sybil resistance: subnet contribution cap and per-peer rejection "
            "rate tracking. Required before opening to public internet participants."
        ),
    )
    sybil_subnet_mask: int = Field(
        default=24,
        ge=8,
        le=32,
        description=(
            "CIDR prefix length for subnet grouping. Peers in the same /N subnet "
            "are treated as a single logical contributor in FedAvg. Default /24."
        ),
    )
    sybil_max_subnet_weight: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Maximum total FedAvg weight (in dataset-size units) any single /N subnet "
            "can contribute. Default 1.0 = one node's worth regardless of how many "
            "share the subnet."
        ),
    )
    rejection_ban_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of rounds rejected that triggers a temporary ban. "
            "E.g. 0.5 = ban a peer that has >50%% of its gradients rejected."
        ),
    )
    rejection_ban_duration_secs: float = Field(
        default=600.0,
        gt=0.0,
        description="Seconds a peer stays banned after exceeding rejection_ban_threshold.",
    )

    # ------------------------------------------------------------------
    # Cluster / scaling (20 nodes: all in cluster 0; 100 nodes: 10 clusters of 10)
    # ------------------------------------------------------------------
    cluster_id: int = Field(
        default=0,
        ge=0,
        description=(
            "Cluster this node belongs to. At 20 nodes all nodes are cluster 0. "
            "At 100 nodes the operator assigns 10 clusters of 10. "
            "Controls ClusterPeerSelector and HierarchicalAggregation."
        ),
    )
    cluster_size: int = Field(
        default=1,
        ge=1,
        description="Expected number of nodes in this cluster. Used by HierarchicalAggregation.",
    )
    aggregation_strategy: Literal["flat", "hierarchical"] = Field(
        default="flat",
        description=(
            "'flat' = single-level FedAvg, correct for ≤ ~30 nodes. "
            "'hierarchical' = two-level cluster averaging, for 100+ nodes."
        ),
    )
    compression: Literal["none", "topk"] = Field(
        default="none",
        description=(
            "'none' = IdentityCompressor (no-op, default for Phases 1-4). "
            "'topk' = Top-K sparsification (swap in when bandwidth is the bottleneck). "
            "TopK uses a self-describing sparse encoding — actual wire savings scale "
            "linearly with (1 - k). At k=0.01, bandwidth is reduced ~50x."
        ),
    )
    topk_ratio: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Fraction of gradient elements to keep when compression='topk'. Default 1%.",
    )

    # ------------------------------------------------------------------
    # NAT traversal (Phase 5+)
    # ------------------------------------------------------------------
    relay_addrs: list[str] = Field(
        default_factory=list,
        description=(
            "Multiaddresses of libp2p circuit-relay nodes. "
            "Nodes behind NAT connect to relays so peers can reach them. "
            "Example: '/ip4/1.2.3.4/tcp/4001/p2p/12D3KooW...'"
        ),
    )
    enable_relay: bool = Field(
        default=False,
        description=(
            "Enable libp2p circuit-relay for NAT traversal. "
            "Set True for internet deployment; False for local Docker sim."
        ),
    )
    enable_hole_punching: bool = Field(
        default=False,
        description=(
            "Enable dcutr hole-punching for direct connections between NAT'd peers. "
            "Falls back to circuit-relay when hole-punch fails."
        ),
    )

    # ------------------------------------------------------------------
    # Relay-only mode
    # ------------------------------------------------------------------
    relay_mode: bool = Field(
        default=False,
        description=(
            "Relay-only mode: run the P2P stack (discovery + heartbeat) without loading "
            "a model or training. Use this for the public bootstrap/relay VPS node that "
            "helps participants discover each other. Training nodes connect to it but it "
            "does not participate in gradient averaging. "
            "Set SWARM_RELAY_MODE=true on the VPS; leave False on all participant nodes."
        ),
    )

    # ------------------------------------------------------------------
    # Chaos / adversarial testing
    # ------------------------------------------------------------------
    adversarial: bool = Field(
        default=False,
        description=(
            "If True, this node broadcasts NaN-filled gradient payloads instead of "
            "real gradients. Used for Phase 4 adversarial chaos testing: other nodes "
            "must detect and reject the poisoned gradients via GradientExtractor.validate() "
            "and continue training. The node still participates normally in heartbeats "
            "and peer discovery — only the broadcast gradient payload is poisoned."
        ),
    )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    checkpoint_dir: Path = Field(
        default=Path("checkpoints"),
        description="Directory to save model checkpoints.",
    )
    checkpoint_every_n_rounds: int = Field(
        default=10,
        ge=0,
        description=(
            "Save a checkpoint every N rounds. 0 = only on clean shutdown. "
            "Checkpoints are written atomically (temp file → rename) so a crash "
            "mid-write cannot produce a corrupt checkpoint."
        ),
    )
    keep_n_checkpoints: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of rolling checkpoints to retain. Older checkpoints are deleted "
            "after each save to prevent unbounded disk growth. The final shutdown "
            "checkpoint is always kept regardless of this limit."
        ),
    )

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------
    heartbeat_eviction_secs: float = Field(
        default=60.0,
        gt=0.0,
        description=(
            "Seconds of missed heartbeats before a peer is evicted. "
            "Default 60 s = 2x the default aggregation timeout (30 s), giving nodes "
            "a full training round of slack before eviction. "
            "Increase when training rounds routinely take > 30 s "
            "(e.g. large model on CPU)."
        ),
    )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="PyTorch compute device. Use 'mps' on Apple Silicon, 'cuda' on NVIDIA.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="'json' for production/Docker, 'console' for local dev.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("checkpoint_dir", mode="before")
    @classmethod
    def validate_checkpoint_dir(cls, v: object) -> object:
        """
        Reject checkpoint directories that point at system-critical paths.

        Without this check, SWARM_CHECKPOINT_DIR=/etc or /proc would cause
        torch.save() to attempt writing model weights into system directories,
        failing at runtime with a cryptic permission error rather than a clear
        startup message.
        """
        from pathlib import Path as _Path

        p = _Path(str(v))
        # Check if the path resolves inside known system directories.
        # We only block obviously dangerous absolute paths — relative paths
        # (the default "checkpoints") are always safe.
        if p.is_absolute():
            _BLOCKED_PREFIXES = (
                "/etc",
                "/sys",
                "/proc",
                "/dev",
                "/boot",
                "/bin",
                "/sbin",
                "/usr/bin",
                "/usr/sbin",
            )
            for blocked in _BLOCKED_PREFIXES:
                if str(p).startswith(blocked):
                    raise ValueError(
                        f"checkpoint_dir={p!r} points at a system directory. "
                        "Choose a writable path such as 'checkpoints' or '/tmp/swarm-checkpoints'."
                    )
        return v

    @field_validator("bootstrap_peers", "relay_addrs", mode="before")
    @classmethod
    def parse_peers(cls, v: str | list[str]) -> list[str]:
        """Accept either a comma-separated string or a list."""
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v

    @field_validator("node_id", mode="before")
    @classmethod
    def default_node_id(cls, v: str) -> str:
        """
        Auto-generate a node_id if empty; sanitize user-provided values.

        Security: node_id is used in checkpoint filenames. Without sanitization,
        a value like '../../etc/cron.d/exploit' would write outside the checkpoint
        directory (path traversal). Strip all path-separator characters and other
        filesystem-unsafe chars, keeping only alphanumeric, hyphens, and underscores.
        """
        if not v:
            import uuid

            return f"node_{uuid.uuid4().hex[:8]}"
        # Strip path separators and other filesystem-unsafe characters.
        sanitized = re.sub(r"[^\w\-]", "_", v)
        if sanitized != v:
            import warnings

            warnings.warn(
                f"node_id {v!r} contained unsafe characters; sanitized to {sanitized!r}",
                stacklevel=2,
            )
        return sanitized

    @model_validator(mode="after")
    def validate_port_range(self) -> NodeSettings:
        """
        Ensure the metrics sidecar port (node_port + 100) is a valid port number.

        Bug fix: without this check, SWARM_PORT > 65435 produces a metrics
        sidecar port > 65535 which causes OSError on startup.
        """
        if self.port + 100 > 65535:
            raise ValueError(
                f"SWARM_PORT={self.port}: metrics sidecar needs port+100={self.port + 100} "
                f"which exceeds the maximum valid port (65535). "
                f"Set SWARM_PORT to 65435 or lower."
            )
        return self
