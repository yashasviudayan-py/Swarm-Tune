#!/usr/bin/env python3
"""
Parse structured JSON logs from a Swarm-Tune simulation run.

Extracts and pretty-prints:
  - Loss curves per node (rounds completed with averaging)
  - Node join/leave events (bootstrap connections, peer evictions)
  - Adversarial rejection events
  - Straggler / deferred round events

Usage:
    docker compose -f docker/docker-compose.yml logs 2>&1 | python scripts/parse_logs.py
    # or
    python scripts/parse_logs.py --file /tmp/swarm_full_run.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def _parse_json_line(line: str) -> dict | None:
    """Extract the JSON object from a Docker log line ('container | {json}')."""
    if " | " not in line:
        return None
    parts = line.split(" | ", 1)
    if len(parts) < 2:
        return None
    try:
        return json.loads(parts[1].strip())
    except json.JSONDecodeError:
        return None


def parse_logs(lines: list[str]) -> None:
    losses: dict[str, list[tuple[int, float]]] = defaultdict(list)
    join_events: list[dict] = []
    adversarial_events: list[dict] = []
    rejection_events: list[dict] = []
    deferred_events: list[dict] = []

    for line in lines:
        d = _parse_json_line(line)
        if d is None:
            continue

        event = d.get("event", "")
        node_id = d.get("node_id", "?")

        if event == "round complete (averaged)":
            losses[node_id].append((d["round"], float(d["loss"])))

        elif event in ("bootstrap peer connected", "peer evicted", "swarm node stopped"):
            join_events.append(d)

        elif event == "adversarial mode: broadcasting NaN gradient payload":
            adversarial_events.append(d)

        elif event == "rejected gradient from peer":
            rejection_events.append(d)

        elif event == "insufficient peers, applying local gradient":
            deferred_events.append(d)

    # ── Loss curves ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("LOSS CURVES (rounds completed with averaged gradients)")
    print("=" * 60)
    for node_id, rounds in sorted(losses.items()):
        if not rounds:
            continue
        round_nums = [r for r, _ in rounds]
        loss_vals = [l for _, l in rounds]
        print(f"\n  {node_id}  ({len(rounds)} rounds)")
        print(f"    first={loss_vals[0]:.4f}  last={loss_vals[-1]:.4f}  "
              f"min={min(loss_vals):.4f}  max={max(loss_vals):.4f}")
        # Print every 4th round as a mini chart
        for i, (r, l) in enumerate(rounds):
            if i % 4 == 0 or i == len(rounds) - 1:
                bar = "█" * int(l * 20)
                print(f"    round {r:2d}: {l:.4f}  {bar}")

    # ── Join / leave events ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NODE JOIN / LEAVE EVENTS")
    print("=" * 60)
    for d in join_events:
        ts = d.get("timestamp", "")[:19]
        event = d.get("event", "")
        node = d.get("node_id", "?")
        addr = d.get("addr", d.get("peer_id", ""))
        print(f"  [{ts}] {event:<30} node={node}  {addr}")

    if not join_events:
        print("  (none logged)")

    # ── Adversarial events ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ADVERSARIAL BROADCAST EVENTS")
    print("=" * 60)
    print(f"  Total NaN broadcasts from node_5_adversarial: {len(adversarial_events)}")
    if adversarial_events:
        first_ts = adversarial_events[0].get("timestamp", "")[:19]
        last_ts = adversarial_events[-1].get("timestamp", "")[:19]
        print(f"  First: {first_ts}  Last: {last_ts}")

    # ── Rejection events ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ADVERSARIAL GRADIENT REJECTIONS (Phase 4 security gate)")
    print("=" * 60)
    from collections import Counter
    rejected_by = Counter(d.get("node_id", "?") for d in rejection_events)
    rejected_peer = Counter(d.get("peer_id", "?") for d in rejection_events)
    print(f"  Total rejections: {len(rejection_events)}")
    print(f"  By peer that was rejected: {dict(rejected_peer)}")
    print(f"  By receiving node:")
    for node, count in sorted(rejected_by.items()):
        print(f"    {node}: {count}")

    # ── Deferred rounds ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DEFERRED ROUNDS (straggler tolerance)")
    print("=" * 60)
    deferred_by = Counter(d.get("node_id", "?") for d in deferred_events)
    print(f"  Total deferred rounds: {len(deferred_events)}")
    for node, count in sorted(deferred_by.items()):
        print(f"    {node}: {count} deferred")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_averaged_rounds = sum(len(v) for v in losses.values())
    print(f"  Nodes that completed rounds with averaging: {len(losses)}")
    print(f"  Total rounds completed with averaged gradients: {total_averaged_rounds}")
    print(f"  Adversarial broadcasts: {len(adversarial_events)}")
    print(f"  Gradient rejections (security gate): {len(rejection_events)}")
    print(f"  Deferred rounds (straggler tolerance): {len(deferred_events)}")
    all_nodes = set(losses.keys()) | {d.get("node_id") for d in rejection_events}
    print(f"  Unique nodes seen: {sorted(all_nodes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Swarm-Tune simulation logs.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to log file. If not provided, reads from stdin.",
    )
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    parse_logs(lines)


if __name__ == "__main__":
    main()
