#!/usr/bin/env python3
"""
Swarm-Tune Checkpoint Reconstruction

Assembles a single full-model checkpoint from multiple per-node shard
checkpoint files saved during a distributed training run.

Two reconstruction strategies:
  merge    (default) — union of all state dict keys.
               Use this for model-parallel runs where different nodes
               train different layers (model_shard_total > 1). Each node's
               checkpoint contains the keys for its assigned layers; merging
               produces a complete model state dict.

  average  — element-wise average of parameters present in all checkpoints.
               Use this for data-parallel runs (model_shard_total=1) to
               produce a consensus checkpoint from all participants.
               Functionally, any single checkpoint from a converged run is
               equivalent — averaging is belt-and-suspenders.

Usage:
    # Merge shards from a model-parallel run (2 shards)
    python scripts/reconstruct_checkpoint.py \\
        --checkpoint-dir checkpoints/ \\
        --model-name gpt2 \\
        --strategy merge \\
        --output full_model.pt

    # Average checkpoints from a data-parallel run
    python scripts/reconstruct_checkpoint.py \\
        --checkpoint-dir checkpoints/ \\
        --model-name gpt2 \\
        --strategy average \\
        --output averaged_model.pt

    # Or via Makefile:
    make reconstruct CHECKPOINT_DIR=checkpoints/ MODEL=gpt2

Output:
    A single .pt file containing the merged / averaged model state dict.
    Load with:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.load_state_dict(torch.load("full_model.pt", weights_only=True))
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconstruct a full model checkpoint from per-node shard files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing per-node checkpoint .pt files.",
    )
    p.add_argument(
        "--model-name",
        default="gpt2",
        help="HuggingFace model name (used to verify the output state dict). Default: gpt2",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/full_model.pt"),
        help="Output file path for the reconstructed checkpoint. Default: checkpoints/full_model.pt",  # noqa: E501
    )
    p.add_argument(
        "--strategy",
        choices=["merge", "average"],
        default="merge",
        help=(
            "'merge': union of all state dict keys (for model-parallel runs). "
            "'average': element-wise mean over all checkpoints (for data-parallel runs). "
            "Default: merge"
        ),
    )
    p.add_argument(
        "--pattern",
        default="*_final.pt",
        help=(
            "Glob pattern to select checkpoint files. "
            "Default: '*_final.pt' (selects the final checkpoint from each node)."
        ),
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to load tensors onto. Default: cpu",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip loading the reconstructed checkpoint into the model architecture for verification.",  # noqa: E501
    )
    return p.parse_args()


def _load_checkpoints(
    checkpoint_dir: Path,
    pattern: str,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Load all matching checkpoint files, returning their state dicts."""
    files = sorted(checkpoint_dir.glob(pattern))
    if not files:
        print(
            f"error: no checkpoint files matching '{pattern}' in {checkpoint_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(files)} checkpoint(s):")
    state_dicts: list[dict[str, torch.Tensor]] = []
    for f in files:
        print(f"  {f.name}")
        raw = torch.load(f, map_location=device, weights_only=True)
        if not isinstance(raw, dict):
            print(f"  warning: {f.name} is not a state dict — skipping.", file=sys.stderr)
            continue
        # Validate all values are tensors
        validated: dict[str, torch.Tensor] = {}
        for k, v in raw.items():
            if isinstance(v, torch.Tensor):
                validated[k] = v
            else:
                print(
                    f"  warning: skipping non-tensor key '{k}' in {f.name}",
                    file=sys.stderr,
                )
        state_dicts.append(validated)

    if not state_dicts:
        print("error: no valid state dicts loaded.", file=sys.stderr)
        sys.exit(1)

    return state_dicts


def _merge(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Union of all keys. Later dictionaries override earlier ones for duplicate keys.

    This is correct for model-parallel runs: each node's checkpoint has the
    trainable parameters for its assigned layers. The union produces the
    complete model state dict.

    If two nodes have the same key with different values (unexpected in a correct
    model-parallel setup), the last-loaded value wins and a warning is printed.
    """
    merged: dict[str, torch.Tensor] = {}
    for sd in state_dicts:
        for key, tensor in sd.items():
            if key in merged:
                if not torch.allclose(merged[key], tensor, atol=1e-6):
                    print(
                        f"  warning: key '{key}' differs between checkpoints — "
                        f"using later value (expected only in model-parallel runs where "
                        f"different shards train the same non-layer parameters).",
                        file=sys.stderr,
                    )
            merged[key] = tensor
    return merged


def _average(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Element-wise average over checkpoints, restricted to keys present in ALL dicts.

    Keys absent from any checkpoint are excluded with a warning — this handles
    the case where nodes had different trainable sets (e.g. some layers frozen).
    """
    all_keys: set[str] = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        all_keys &= set(sd.keys())

    total_keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        total_keys |= set(sd.keys())

    excluded = total_keys - all_keys
    if excluded:
        print(
            f"  warning: {len(excluded)} key(s) excluded from averaging (not in all checkpoints): "
            + ", ".join(sorted(excluded)[:5])
            + (" ..." if len(excluded) > 5 else ""),
            file=sys.stderr,
        )

    averaged: dict[str, torch.Tensor] = {}
    for key in sorted(all_keys):
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        averaged[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)

    return averaged


def _verify(state_dict: dict[str, torch.Tensor], model_name: str) -> None:
    """Load the reconstructed state dict into the model architecture to verify it."""
    print(f"\nVerifying: loading reconstructed weights into {model_name}...")
    try:
        from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            suffix = " ..." if len(missing) > 5 else ""
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{suffix}")
        if unexpected:
            suffix = " ..." if len(unexpected) > 5 else ""
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{suffix}")
        if not missing and not unexpected:
            print("  Verification PASSED — all keys match.")
        else:
            print("  Verification PARTIAL — see above. This is expected for model-parallel runs.")
    except Exception as exc:
        print(f"  Verification skipped: {exc}", file=sys.stderr)


def main() -> None:
    args = _parse_args()
    checkpoint_dir: Path = args.checkpoint_dir

    if not checkpoint_dir.is_dir():
        print(f"error: checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    state_dicts = _load_checkpoints(checkpoint_dir, args.pattern, device)
    print(f"\nStrategy: {args.strategy}")

    if args.strategy == "merge":
        result = _merge(state_dicts)
    else:
        result = _average(state_dicts)

    param_count = sum(t.numel() for t in result.values())
    print(f"Reconstructed: {len(result)} parameter tensors, {param_count:,} total elements")

    if not args.skip_verify and args.model_name != "mlp":
        _verify(result, args.model_name)

    # Atomic write
    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    try:
        torch.save(result, tmp)
        import os
        os.replace(tmp, output)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    print(f"\nSaved: {output}")
    print("\nTo benchmark:")
    print(f"  make benchmark CHECKPOINT={output} MODEL={args.model_name}")


if __name__ == "__main__":
    main()
