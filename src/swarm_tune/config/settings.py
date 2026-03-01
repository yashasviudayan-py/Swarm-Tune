"""
Central configuration for a Swarm-Tune node.

All values are read from environment variables (with SWARM_ prefix)
and can be overridden via a YAML config file. Pydantic validates
and coerces every field at startup — bad config fails loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class NodeSettings(BaseSettings):
    """Runtime configuration for a single Swarm-Tune node."""

    model_config = SettingsConfigDict(
        env_prefix="SWARM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    node_id: str = Field(
        default="",
        description="Unique human-readable node identifier. Auto-generated if empty.",
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
        description="Path to this node's local training data shard.",
    )
    model_name: str = Field(
        default="gpt2",
        description="HuggingFace model name or local path to load.",
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

    # ------------------------------------------------------------------
    # Training hyperparameters
    # ------------------------------------------------------------------
    learning_rate: float = Field(default=1e-4, gt=0, description="SGD / Adam learning rate.")
    batch_size: int = Field(default=8, ge=1, description="Local mini-batch size per round.")
    num_rounds: int = Field(default=100, ge=1, description="Total number of training rounds.")

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
    @field_validator("bootstrap_peers", mode="before")
    @classmethod
    def parse_peers(cls, v: str | list[str]) -> list[str]:
        """Accept either a comma-separated string or a list."""
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v

    @field_validator("node_id", mode="before")
    @classmethod
    def default_node_id(cls, v: str) -> str:
        if not v:
            import uuid
            return f"node_{uuid.uuid4().hex[:8]}"
        return v
