#!/usr/bin/env python3
"""
Swarm-Tune Benchmark Script — Perplexity Evaluation

Loads a model checkpoint and evaluates perplexity on the WikiText-103 test
split. The score is deterministic: given the same checkpoint and test set,
this script always produces the same number on any machine.

Usage:
    python scripts/benchmark.py --checkpoint checkpoints/node_0_final.pt
    make benchmark CHECKPOINT=./checkpoints/node_0_final.pt

The printed perplexity can be independently verified by any third party with
the same checkpoint file. Teams publish their scores; no central leaderboard
is needed.

Exit code 0 = success. Perplexity is printed to stdout in machine-parseable
format: "perplexity: <float>"
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate perplexity of a Swarm-Tune checkpoint.")
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the .pt checkpoint file (model state dict).",
    )
    p.add_argument(
        "--model-name",
        default="gpt2",
        help="HuggingFace model architecture (must match the checkpoint). Default: gpt2",
    )
    p.add_argument(
        "--dataset",
        default="wikitext",
        help="HuggingFace dataset name. Default: wikitext",
    )
    p.add_argument(
        "--dataset-config",
        default="wikitext-103-raw-v1",
        help="HuggingFace dataset config. Default: wikitext-103-raw-v1",
    )
    p.add_argument(
        "--seq-len",
        default=512,
        type=int,
        help="Sequence length for evaluation. Default: 512",
    )
    p.add_argument(
        "--batch-size",
        default=4,
        type=int,
        help="Evaluation batch size. Default: 4",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Compute device. Default: cpu",
    )
    p.add_argument(
        "--max-batches",
        default=None,
        type=int,
        help="Maximum number of batches to evaluate (None = full test set).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    checkpoint: Path = args.checkpoint
    if not checkpoint.exists():
        print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model_name}")
    print(f"Checkpoint:    {checkpoint}")
    print(f"Dataset:       {args.dataset} / {args.dataset_config}")
    print(f"Device:        {args.device}")

    try:
        import torch
        from datasets import load_dataset  # type: ignore[import-untyped]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
    except ImportError as e:
        print(f"error: missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install swarm-tune[dev]", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)

    # Load model architecture and inject checkpoint weights.
    print("Loading model weights from checkpoint...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test split — deterministic, never random.
    print("Loading test dataset...")
    test_data = load_dataset(args.dataset, args.dataset_config, split="test")
    test_data = test_data.filter(lambda ex: len(ex["text"].strip()) > 10)

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.seq_len,
            padding="max_length",
        )
        # Return both input_ids and attention_mask so we can exclude padding tokens
        # from the perplexity calculation (padding inflates loss artificially).
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    tokenized = test_data.map(_tokenize, batched=True, remove_columns=test_data.column_names)
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)

    print(f"Test samples:  {len(input_ids)}")

    # Evaluate perplexity: exp(mean cross-entropy loss over non-padding tokens).
    # We set padding positions in labels to -100 so HuggingFace's built-in
    # cross-entropy ignores them. Then we weight each batch's outputs.loss by
    # the number of real tokens in that batch to get a correct weighted mean.
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for start in range(0, len(input_ids), args.batch_size):
            if args.max_batches is not None and num_batches >= args.max_batches:
                break
            batch = input_ids[start : start + args.batch_size].to(device)
            mask = attention_mask[start : start + args.batch_size].to(device)

            # Set padding positions to -100 so they are excluded from the loss.
            labels = batch.clone()
            labels[mask == 0] = -100

            outputs = model(input_ids=batch, labels=labels)
            # outputs.loss is mean CE over real (non-padding) tokens only.
            # Multiply by real token count to recover the sum, then accumulate.
            n_real_tokens = int(mask.sum().item())
            if n_real_tokens == 0:
                continue
            total_loss += outputs.loss.item() * n_real_tokens
            total_tokens += n_real_tokens
            num_batches += 1

            if num_batches % 50 == 0:
                current_ppl = math.exp(total_loss / total_tokens)
                print(f"  [{num_batches} batches] running perplexity: {current_ppl:.2f}")

    if total_tokens == 0:
        print("error: no tokens evaluated", file=sys.stderr)
        sys.exit(1)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print()
    print(f"Batches evaluated: {num_batches}")
    print(f"Tokens evaluated:  {total_tokens:,}")
    print(f"Average loss:      {avg_loss:.4f}")
    print(f"perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()
