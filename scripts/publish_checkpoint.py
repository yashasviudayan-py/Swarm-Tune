#!/usr/bin/env python3
"""
Swarm-Tune Checkpoint Publisher

Publishes a trained model checkpoint to the HuggingFace Hub with a model card
describing the training run. This enables:
  1. Independent perplexity verification by any third party.
  2. Competition result publication — teams post their checkpoint; anyone can verify.
  3. Community model sharing — others can fine-tune from a swarm-trained checkpoint.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   (or set HF_TOKEN env var)

Usage:
    python scripts/publish_checkpoint.py \\
        --checkpoint checkpoints/full_model.pt \\
        --model-name gpt2 \\
        --repo-id your-username/gpt2-swarmtune-001 \\
        --run-id gpt2-wikitrain-001 \\
        --perplexity 42.5

    # Competition submission
    python scripts/publish_checkpoint.py \\
        --checkpoint checkpoints/full_model.pt \\
        --model-name gpt2 \\
        --repo-id your-team/gpt2-competition-001 \\
        --run-id gpt2-competition-001 \\
        --perplexity 39.8 \\
        --team-id team-alpha

After publishing:
    Others can verify your score:
        python scripts/benchmark.py \\
            --checkpoint <downloaded_or_local.pt> \\
            --model-name gpt2

    Anyone who gets a different perplexity should check:
      - Same model architecture (gpt2)
      - Same dataset (wikitext/wikitext-103-raw-v1)
      - Same seq-len (512)
      - Same batch-size for evaluation (4)

Security note:
    This script uploads model weights to a public HuggingFace repository.
    Only run it with checkpoints you intend to share publicly.
    Never upload .env files or private keys.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publish a Swarm-Tune checkpoint to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the reconstructed full-model .pt checkpoint.",
    )
    p.add_argument(
        "--model-name",
        default="gpt2",
        help="Base HuggingFace model architecture (e.g. 'gpt2'). Default: gpt2",
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace Hub repo (e.g. 'your-username/gpt2-swarmtune-001').",
    )
    p.add_argument(
        "--run-id",
        default="",
        help="Swarm-Tune run_id (e.g. 'gpt2-wikitrain-001') for the model card.",
    )
    p.add_argument(
        "--perplexity",
        type=float,
        default=None,
        help="WikiText-103 perplexity (from scripts/benchmark.py) to embed in the model card.",
    )
    p.add_argument(
        "--team-id",
        default="",
        help="Team name for competition submissions.",
    )
    p.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="Number of nodes that participated in training.",
    )
    p.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Number of training rounds completed.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository (default: public).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the model card without uploading.",
    )
    return p.parse_args()


def _build_model_card(args: argparse.Namespace) -> str:
    """Generate the model card Markdown for HuggingFace Hub."""
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    ppl_line = (
        f"- **WikiText-103 Perplexity:** {args.perplexity:.2f}"
        if args.perplexity is not None
        else "- **WikiText-103 Perplexity:** (not recorded — run `make benchmark` to compute)"
    )
    team_line = f"- **Team:** {args.team_id}" if args.team_id else ""
    run_line = f"- **Run ID:** `{args.run_id}`" if args.run_id else ""
    nodes_line = f"- **Nodes:** {args.num_nodes}" if args.num_nodes else ""
    rounds_line = f"- **Rounds:** {args.num_rounds}" if args.num_rounds else ""

    verify_cmd = (
        f"python scripts/benchmark.py --checkpoint <path/to/checkpoint.pt> "
        f"--model-name {args.model_name}"
    )

    return f"""---
license: mit
tags:
  - swarm-tune
  - federated-learning
  - p2p-training
  - pytorch
base_model: {args.model_name}
---

# Swarm-Tune Checkpoint — {args.repo_id.split("/")[-1]}

Fine-tuned via [Swarm-Tune](https://github.com/yashasviudayan-py/Swarm-Tune):
decentralized federated learning over a P2P network (libp2p + FedAvg).

No data center. No cloud GPUs. Trained collectively by participants pooling
commodity hardware over the internet.

## Training Details

{ppl_line}
{team_line}
{run_line}
{nodes_line}
{rounds_line}
- **Base model:** `{args.model_name}`
- **Dataset:** WikiText-103
- **Published:** {now}

## Independent Verification

Anyone can reproduce the perplexity score:

```bash
# Install
pip install swarm-tune

# Download this checkpoint from HuggingFace Hub
# (or use a local copy)

# Evaluate
{verify_cmd}
```

The benchmark script is deterministic: the same checkpoint always produces
the same perplexity on the same machine. If you get a different score,
check that you are using the same `--model-name`, `--dataset`, `--seq-len`,
and `--batch-size` as the defaults in `scripts/benchmark.py`.

## How It Was Trained

Swarm-Tune distributes model fine-tuning across commodity hardware:

1. Each node holds a shard of the dataset and trains locally.
2. After each round, nodes extract raw `param.grad` tensors and broadcast
   them via [libp2p](https://libp2p.io/) GossipSub.
3. The swarm runs Federated Averaging (FedAvg) — weighted by dataset size.
4. Every node applies the averaged gradients and stays in sync.

Security: NaN/Inf gradient poisoning is detected and rejected before any
gradient enters the FedAvg pool. Sybil resistance limits any one subnet's
contribution. No central server. No trusted coordinator.

## Reproducibility

Base model weights are the public HuggingFace checkpoint for `{args.model_name}`.
Dataset is WikiText-103 (wikitext-103-raw-v1) from HuggingFace Datasets.
Both are deterministically reproducible. Given this checkpoint file, the
perplexity score above is independently verifiable by any third party.
"""


def main() -> None:
    args = _parse_args()
    checkpoint: Path = args.checkpoint

    if not checkpoint.exists():
        print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    model_card = _build_model_card(args)

    if args.dry_run:
        print("=== MODEL CARD (dry-run) ===")
        print(model_card)
        print("=== END MODEL CARD ===")
        print(f"\nWould upload: {checkpoint} → {args.repo_id}")
        return

    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
    except ImportError:
        print(
            "error: huggingface_hub is not installed.\n"
            "Install with: pip install huggingface_hub\n"
            "Then authenticate: huggingface-cli login",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi()

    print(f"Creating repository: {args.repo_id} (private={args.private})")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading checkpoint: {checkpoint} ({checkpoint.stat().st_size / 1e6:.1f} MB)")
    api.upload_file(
        path_or_fileobj=checkpoint,
        path_in_repo="model.pt",
        repo_id=args.repo_id,
        repo_type="model",
    )

    print("Writing model card...")
    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    repo_url = f"https://huggingface.co/{args.repo_id}"
    print(f"\nPublished: {repo_url}")
    if args.perplexity is not None:
        print(f"Perplexity: {args.perplexity:.2f}")
    print("\nVerification command for others:")
    print(
        f"  python scripts/benchmark.py "
        f"--checkpoint <local_copy.pt> "
        f"--model-name {args.model_name}"
    )


if __name__ == "__main__":
    main()
