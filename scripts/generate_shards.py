#!/usr/bin/env python3
"""
Generate synthetic training data shards for local simulation.

Creates N shards of random (input, target) pairs and saves them as
.pt files in data/shards/. Each Docker container mounts the shards
directory and loads its assigned shard.

Usage:
    python scripts/generate_shards.py
    python scripts/generate_shards.py --num-shards 5 --samples-per-shard 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def generate_shards(
    num_shards: int,
    samples_per_shard: int,
    input_dim: int,
    output_dir: Path,
    token_ids: bool = False,
    vocab_size: int = 50257,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_shards):
        if token_ids:
            # Integer token-ID shards for HuggingFace CausalLM models (e.g. GPT-2).
            # inputs  = token IDs in [0, vocab_size)
            # targets = same sequence shifted by 1 (next-token prediction)
            seq = torch.randint(0, vocab_size, (samples_per_shard, input_dim))
            shard = {
                "inputs": seq,
                "targets": seq,  # model uses labels= for causal shift internally
                "shard_index": i,
                "num_shards": num_shards,
            }
        else:
            shard = {
                "inputs": torch.randn(samples_per_shard, input_dim),
                "targets": torch.randn(samples_per_shard, input_dim),
                "shard_index": i,
                "num_shards": num_shards,
            }
        path = output_dir / f"shard_{i}.pt"
        torch.save(shard, path)
        print(f"  [+] shard_{i}.pt  ({samples_per_shard} samples, dim={input_dim}, token_ids={token_ids})")

    print(f"\nGenerated {num_shards} shards in {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data shards.")
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--samples-per-shard", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("data/shards"))
    parser.add_argument(
        "--token-ids",
        action="store_true",
        help="Generate integer token-ID shards for HuggingFace CausalLM models (e.g. GPT-2).",
    )
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size for token-ID shards.")
    args = parser.parse_args()

    print(f"Generating {args.num_shards} shards → {args.output_dir}/")
    generate_shards(
        num_shards=args.num_shards,
        samples_per_shard=args.samples_per_shard,
        input_dim=args.input_dim,
        output_dir=args.output_dir,
        token_ids=args.token_ids,
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    main()
