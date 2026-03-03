"""
Data shard loading for local training.

Each node holds a pre-split shard of the training dataset.
The shard is stored as a .pt file with the structure produced by
scripts/generate_shards.py:

    {
        "inputs":      Tensor(N, D),  # input features
        "targets":     Tensor(N, D),  # target labels
        "shard_index": int,           # which shard this is (0-based)
        "num_shards":  int,           # total number of shards in the run
    }

On startup the node calls load() once. During training, get_batch()
samples a random mini-batch each round — no epoch boundaries needed
for Phase 2 (we sample with replacement for simplicity).

All returned tensors are on CPU.  The training loop is responsible for
moving them to the compute device (ModelShard.forward() handles inputs;
the caller moves targets via targets.to(output.device)).
"""

from __future__ import annotations

from pathlib import Path

import structlog
import torch

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
