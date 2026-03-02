"""
Model shard management.

In a full deployment each node holds a vertical or horizontal slice of
the model. For Phase 2 (local simulation) every node holds the full
model but trains on a different data shard — functionally equivalent
to data-parallel training with manual gradient averaging.

Future phases will implement true pipeline / tensor parallelism.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from swarm_tune.config.settings import NodeSettings

log: structlog.BoundLogger = structlog.get_logger(__name__)


class ModelShard:
    """
    Wraps a PyTorch model (or model shard) with a clean training interface.

    Responsibilities:
      - Load the model onto the correct device (cpu / mps / cuda).
      - Expose forward() and backward() so the node's training loop
        stays device-agnostic.
      - Track the local optimizer state.
    """

    def __init__(self, settings: NodeSettings) -> None:
        self._settings = settings
        self._device = torch.device(settings.device)
        self._model: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None

    def load(self, checkpoint_path: Path | None = None) -> None:
        """
        Instantiate the model and move it to the node's device.

        Args:
            checkpoint_path: optional path to a .pt checkpoint to resume from.
        """
        log.info(
            "loading model",
            model_name=self._settings.model_name,
            device=str(self._device),
        )
        # TODO(phase-2): load from HuggingFace or local checkpoint
        # self._model = AutoModelForCausalLM.from_pretrained(self._settings.model_name)
        # Placeholder: tiny 2-layer MLP for Phase 2 integration tests
        self._model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        ).to(self._device)

        if checkpoint_path and checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state)
            log.info("resumed from checkpoint", path=str(checkpoint_path))

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._settings.learning_rate
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return the model output."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model(batch.to(self._device))

    def backward(self, loss: torch.Tensor) -> None:
        """Zero gradients, run backward pass to populate param.grad."""
        if self._optimizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        self._optimizer.zero_grad()
        loss.backward()

    def apply_averaged_gradients(self, averaged: dict[str, torch.Tensor]) -> None:
        """
        Overwrite local param.grad tensors with the swarm-averaged gradients
        and step the optimizer.

        Called by the Aggregator after collecting and averaging peer gradients.
        """
        if self._model is None or self._optimizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for name, param in self._model.named_parameters():
            if name in averaged:
                param.grad = averaged[name].to(self._device)

        self._optimizer.step()
        log.debug("optimizer stepped with averaged gradients")

    def save_checkpoint(self, path: Path) -> None:
        """Save model state dict to disk."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        log.info("checkpoint saved", path=str(path))

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        return self._model
