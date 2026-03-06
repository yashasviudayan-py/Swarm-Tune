"""
Data shard loading for local training.

Two backends are provided:

DataShardLoader (Phase 1-4)
    Loads a pre-generated .pt file (produced by scripts/generate_shards.py):
        {"inputs": Tensor(N, D), "targets": Tensor(N, D), ...}
    Used for Docker simulation with synthetic data.

HFDataShardLoader (Phase 5+)
    Streams a HuggingFace dataset, tokenizes it with AutoTokenizer, and
    deterministically shards it by (shard_index, shard_total). Returns
    (input_ids, input_ids) pairs for causal language model training —
    the model shifts labels internally.

The SwarmNode picks the right loader via create_data_loader(settings).
All returned tensors are on CPU; the training loop moves them to device.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import torch

if TYPE_CHECKING:
    from swarm_tune.config.settings import NodeSettings

log: structlog.BoundLogger = structlog.get_logger(__name__)


class DataShardLoader:
    """
    Loads a local training data shard from disk and yields mini-batches.

    Security note: this class loads OUR OWN local data files, not data
    received from peers.  Therefore weights_only=False is acceptable here.
    Peer-received gradient payloads are loaded with weights_only=True in
    GradientSerializer.deserialize() — that is the security-critical path.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._inputs: torch.Tensor | None = None
        self._targets: torch.Tensor | None = None

    def load(self) -> None:
        """
        Load the shard file into memory.

        Raises:
            FileNotFoundError: if the shard file does not exist.
            KeyError: if the file is missing the required 'inputs' or 'targets' keys.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"Data shard not found: {self._path}")

        # weights_only=False is safe here — this is our own local data file.
        raw: object = torch.load(self._path, map_location="cpu", weights_only=False)

        if not isinstance(raw, dict):
            raise KeyError(f"Shard file {self._path} is not a dict.")
        if "inputs" not in raw or "targets" not in raw:
            raise KeyError(
                f"Shard file {self._path} is missing 'inputs' or 'targets' keys. "
                f"Found keys: {list(raw.keys())}"
            )

        self._inputs = raw["inputs"]
        self._targets = raw["targets"]

        log.info(
            "data shard loaded",
            path=str(self._path),
            samples=self.dataset_size,
            shape=list(self._inputs.shape),
        )

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a random mini-batch from the shard (sampling with replacement).

        Args:
            batch_size: number of samples in the mini-batch.

        Returns:
            (inputs, targets) both of shape (min(batch_size, N), *feature_dims)
            on CPU. The training loop moves them to the compute device.

        Raises:
            RuntimeError: if load() has not been called yet.
        """
        if self._inputs is None or self._targets is None:
            raise RuntimeError("Data shard not loaded. Call load() first.")

        n = len(self._inputs)
        effective_size = min(batch_size, n)
        indices = torch.randint(0, n, (effective_size,))
        return self._inputs[indices], self._targets[indices]

    @property
    def dataset_size(self) -> int:
        """
        Number of samples in this node's shard.

        Used by the aggregator for weighted FedAvg: a node with more training
        samples contributes proportionally more to the global average.

        Raises:
            RuntimeError: if load() has not been called yet.
        """
        if self._inputs is None:
            raise RuntimeError("Data shard not loaded. Call load() first.")
        return int(self._inputs.shape[0])


class HFDataShardLoader:
    """
    Loads a shard of a HuggingFace dataset and yields token-ID mini-batches.

    Sharding is deterministic: dataset.shard(num_shards=shard_total, index=shard_index)
    ensures non-overlapping splits. Each node downloads the full dataset but
    only trains on its assigned shard — no peer-to-peer data transfer required.

    get_batch() returns (input_ids, input_ids) — labels equal inputs for causal
    LM training. HuggingFace models shift labels internally during the forward
    pass when labels are provided.

    Security note: this loads a public dataset from HuggingFace Hub using the
    datasets library's standard download mechanism. No peer-received data is
    deserialized here.
    """

    def __init__(self, settings: NodeSettings) -> None:
        self._model_name = settings.model_name
        self._dataset_name = settings.dataset_name
        self._dataset_config = settings.dataset_config
        self._shard_index = settings.model_shard_index
        self._shard_total = settings.model_shard_total
        self._max_seq_len = settings.max_seq_len
        self._input_ids: torch.Tensor | None = None

    def load(self) -> None:
        """
        Download (or load from cache), shard, and tokenize the dataset.

        The tokenized sequences are stored in memory as a single (N, seq_len)
        Long tensor for fast random-access sampling during training.

        Raises:
            ImportError: if `datasets` or `transformers` are not installed.
            ValueError: if the dataset has no 'text' column.
        """
        from datasets import load_dataset
        from transformers import AutoTokenizer

        log.info(
            "loading HuggingFace dataset",
            dataset=self._dataset_name,
            config=self._dataset_config,
            shard_index=self._shard_index,
            shard_total=self._shard_total,
        )

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        raw = load_dataset(self._dataset_name, self._dataset_config, split="train")

        # Deterministic shard assignment — no overlap between nodes.
        if self._shard_total > 1:
            raw = raw.shard(num_shards=self._shard_total, index=self._shard_index)

        col_names: list[str] = list(raw.column_names)
        if "text" not in col_names:
            raise ValueError(
                f"Dataset '{self._dataset_name}' has no 'text' column. "
                f"Available columns: {col_names}"
            )

        # Filter out empty or very short entries before tokenizing.
        raw = raw.filter(lambda ex: len(ex["text"].strip()) > 10)

        def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
            encoded = tokenizer(
                batch["text"],
                truncation=True,
                max_length=self._max_seq_len,
                padding="max_length",
            )
            return {"input_ids": encoded["input_ids"]}

        tokenized = raw.map(_tokenize, batched=True, remove_columns=col_names)
        self._input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)

        log.info(
            "HuggingFace dataset loaded",
            dataset=self._dataset_name,
            samples=self.dataset_size,
            seq_len=self._max_seq_len,
        )

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a random mini-batch (sampling with replacement).

        Returns:
            (input_ids, input_ids): both Long tensors of shape (batch, seq_len).
            Labels == inputs for causal LM — the model shifts internally.

        Raises:
            RuntimeError: if load() has not been called yet.
        """
        if self._input_ids is None:
            raise RuntimeError("Data shard not loaded. Call load() first.")

        n = len(self._input_ids)
        effective_size = min(batch_size, n)
        indices = torch.randint(0, n, (effective_size,))
        ids = self._input_ids[indices]
        return ids, ids.clone()

    @property
    def dataset_size(self) -> int:
        """Number of samples in this node's shard."""
        if self._input_ids is None:
            raise RuntimeError("Data shard not loaded. Call load() first.")
        return int(self._input_ids.shape[0])


def create_data_loader(settings: NodeSettings) -> DataShardLoader | HFDataShardLoader:
    """
    Factory: return the appropriate data loader based on settings.

    - If settings.dataset_name is set → HFDataShardLoader (Phase 5+).
    - Otherwise → DataShardLoader from settings.data_shard_path (Phase 1-4).
    """
    if settings.dataset_name:
        return HFDataShardLoader(settings)
    return DataShardLoader(settings.data_shard_path)
