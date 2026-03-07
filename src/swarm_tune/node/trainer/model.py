"""
Model shard management.

Supports two model backends:
  - "mlp": a tiny 2-layer MLP (used for unit tests and Phase 1-4 simulation).
  - Any HuggingFace CausalLM model name (e.g. "gpt2", "meta-llama/Llama-3-8B").

For HuggingFace models, each node loads the full model but only the layers
assigned to its shard (shard_index % shard_total == this node) have
requires_grad=True and are covered by the optimizer. All other layers are
frozen. This gives gradient-level model parallelism without tensor passing
between nodes — gradients are averaged across nodes per-layer.

Design note on multi-shard mode
---------------------------------
When shard_total > 1, different nodes have different trainable layers and
therefore different parameter sets. GradientAverager handles this via
intersection-based averaging: only parameters present in ALL contributions
are averaged. In practice this means:
  - Nodes with the SAME shard_index exchange useful gradients (intersection
    = their full shared param set).
  - Nodes with DIFFERENT shard_index contribute zero to each other's average
    (empty intersection → they update from their own local gradient only).

For effective multi-shard collaboration, route same-shard nodes into the same
cluster (cluster_id = shard_index) so ClusterPeerSelector keeps them together.
Nodes with different shard assignments do NOT need to exchange gradients —
they train independent layer slices of the same base model.

Switching from MLP → GPT-2 → LLaMA requires only a config change
(SWARM_MODEL_NAME). No trainer logic is model-specific.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from swarm_tune.config.settings import NodeSettings

log: structlog.BoundLogger = structlog.get_logger(__name__)

_MLP_MODEL_NAME = "mlp"


def _get_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """
    Extract the list of transformer block layers from a HuggingFace model.

    Tries common attribute paths used by different architectures:
      - GPT-2: model.transformer.h
      - LLaMA/Mistral: model.model.layers
      - Falcon: model.transformer.h
      - BLOOM: model.transformer.h
    """
    # Traverse up to two levels: the model itself, then .transformer or .model
    candidates: list[nn.Module] = [model]
    for attr in ("transformer", "model"):
        child = getattr(model, attr, None)
        if isinstance(child, nn.Module):
            candidates.append(child)

    for container in candidates:
        for layer_attr in ("h", "layers", "blocks", "layer"):
            layers = getattr(container, layer_attr, None)
            if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
                return list(layers)

    return []


class ModelShard:
    """
    Wraps a PyTorch model (or model shard) with a clean training interface.

    Responsibilities:
      - Load the model onto the correct device (cpu / mps / cuda).
      - For HuggingFace CausalLM: freeze layers not assigned to this shard.
      - Expose compute_loss() for the training loop (handles both MLP and HF).
      - Expose forward() for direct use in tests (MLP only).
      - Track the local optimizer state (covers only trainable parameters).
    """

    def __init__(self, settings: NodeSettings) -> None:
        self._settings = settings
        self._device = torch.device(settings.device)
        self._model: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._is_causal_lm: bool = False

    @property
    def is_causal_lm(self) -> bool:
        """True when the model is a HuggingFace CausalLM; False for MLP."""
        return self._is_causal_lm

    def load(self, checkpoint_path: Path | None = None) -> None:
        """
        Instantiate the model and move it to the node's device.

        For HuggingFace models, layers not belonging to this shard are frozen
        so the optimizer only covers this node's assigned parameters.

        Args:
            checkpoint_path: optional path to a .pt checkpoint to resume from.
        """
        model_name = self._settings.model_name
        log.info("loading model", model_name=model_name, device=str(self._device))

        if model_name == _MLP_MODEL_NAME:
            self._model = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            ).to(self._device)
            self._is_causal_lm = False
        else:
            from transformers import AutoModelForCausalLM

            hf_model: nn.Module = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
            hf_model = hf_model.to(self._device)

            # Apply layer-level sharding: freeze all layers not in this shard.
            # Layer i belongs to shard: i % shard_total == shard_index.
            shard_total = self._settings.model_shard_total
            shard_index = self._settings.model_shard_index
            if shard_total > 1:
                layers = _get_transformer_layers(hf_model)
                if layers:
                    for i, layer in enumerate(layers):
                        if i % shard_total != shard_index:
                            for param in layer.parameters():
                                param.requires_grad = False
                    active = [i for i in range(len(layers)) if i % shard_total == shard_index]
                    log.info(
                        "layer sharding applied",
                        shard_index=shard_index,
                        shard_total=shard_total,
                        total_layers=len(layers),
                        active_layers=active,
                    )
                    if shard_total > 1:
                        log.info(
                            "multi-shard mode: this node trains a subset of layers. "
                            "For effective collaboration, set cluster_id=shard_index so "
                            "nodes with the same shard exchange gradients via "
                            "ClusterPeerSelector.",
                        )

            self._model = hf_model
            self._is_causal_lm = True

        if checkpoint_path and checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state)
            log.info("resumed from checkpoint", path=str(checkpoint_path))

        # Optimizer only covers trainable parameters (respects frozen layers).
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(trainable, lr=self._settings.learning_rate)

        log.info(
            "model ready",
            model_name=model_name,
            is_causal_lm=self._is_causal_lm,
            trainable_params=sum(p.numel() for p in trainable),
            total_params=sum(p.numel() for p in self._model.parameters()),
        )

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and compute the training loss.

        For MLP (model_name="mlp"):
            inputs/targets are float tensors — MSE loss is used.

        For HuggingFace CausalLM:
            inputs/targets are Long (int64) token-ID tensors.
            The model's built-in cross-entropy loss is returned via
            outputs.loss (HF models compute this when labels are passed).

        This is the preferred API for the training loop. It encapsulates the
        loss function so SwarmNode._training_round stays model-agnostic.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self._is_causal_lm:
            outputs: Any = self._model(
                input_ids=inputs.to(self._device),
                labels=targets.to(self._device),
            )
            return cast(torch.Tensor, outputs.loss)
        else:
            output = cast(torch.Tensor, self._model(inputs.to(self._device)))
            return torch.nn.functional.mse_loss(output, targets.to(self._device))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return the raw model output.

        Primarily used by tests and by the MLP path. For HuggingFace models,
        prefer compute_loss() which passes labels and returns outputs.loss.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return cast(torch.Tensor, self._model(batch.to(self._device)))

    def backward(self, loss: torch.Tensor) -> None:
        """Zero gradients, run backward pass to populate param.grad."""
        if self._optimizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        self._optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]

    def apply_averaged_gradients(self, averaged: dict[str, torch.Tensor]) -> None:
        """
        Overwrite local param.grad tensors with the swarm-averaged gradients
        and step the optimizer.

        Only applies gradients to parameters that require grad (unfrozen).
        Called by the training loop after collecting and averaging peer gradients.
        """
        if self._model is None or self._optimizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for name, param in self._model.named_parameters():
            if name in averaged and param.requires_grad:
                param.grad = averaged[name].to(self._device)

        self._optimizer.step()
        log.debug("optimizer stepped with averaged gradients")

    def save_checkpoint(self, path: Path) -> None:
        """
        Save model state dict to disk atomically.

        Writes to a temporary file first, then performs an atomic rename.
        This guarantees that a crash mid-write cannot corrupt an existing
        checkpoint: the previous checkpoint remains intact until the new one
        is fully flushed to disk and renamed into place.

        On POSIX systems (Linux, macOS), os.replace() is atomic when the
        source and destination are on the same filesystem — which they always
        are here since both are in checkpoint_dir.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        try:
            torch.save(self._model.state_dict(), tmp_path)
            os.replace(tmp_path, path)  # atomic on POSIX
        except Exception:
            # Clean up the temp file if something went wrong before the rename.
            tmp_path.unlink(missing_ok=True)
            raise
        log.info("checkpoint saved (atomic)", path=str(path))

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model
