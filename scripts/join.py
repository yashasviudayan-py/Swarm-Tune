#!/usr/bin/env python3
"""
Swarm-Tune Join Script — Zero-Config Node Onboarding

Reads a training campaign manifest (runs/<RUN_ID>.json) and generates the
environment file and Docker command for a specific participant.

Usage:
    python scripts/join.py --run-id gpt2-wikitrain-001 --node-index 2
    make join RUN_ID=gpt2-wikitrain-001 NODE_INDEX=2

What it does:
  1. Loads  runs/<RUN_ID>.json
  2. Validates node_index is in range [0, num_shards)
  3. Writes a .env file with all SWARM_ variables set correctly for this node
  4. Prints the docker run command to start this node

The only manual step remaining is to set SWARM_BOOTSTRAP_PEERS in your .env
after node_0 (the bootstrap) has started and advertised its address.  If the
manifest already contains bootstrap_peers, this is handled automatically.

Node assignment:
  node_index=0 — the first joiner; acts as bootstrap for the others.
                  Its advertised multiaddr must be shared with other participants
                  (copy from the startup log: "Swarm-Tune node ... is READY").
  node_index=1+ — participants; set SWARM_BOOTSTRAP_PEERS to node_0's address.

Security note:
  This script only reads local manifest files and writes a local .env file.
  It makes no network connections.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from any directory: add repo src/ to sys.path.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from swarm_tune.runs.manifest import RunManifest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate .env and Docker command for a Swarm-Tune participant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Node 0 (bootstrap) for the GPT-2 WikiText run
  python scripts/join.py --run-id gpt2-wikitrain-001 --node-index 0

  # Node 2 with a specific port and device
  python scripts/join.py --run-id gpt2-wikitrain-001 --node-index 2 \\
    --port 9002 --device mps

  # Preview without writing any files
  python scripts/join.py --run-id gpt2-wikitrain-001 --node-index 1 --dry-run

  # Use a custom runs directory
  python scripts/join.py --run-id my-run --runs-dir /path/to/runs --node-index 0
""",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="Run identifier matching a file in runs/<run-id>.json.",
    )
    p.add_argument(
        "--node-index",
        type=int,
        default=0,
        help=(
            "This participant's data shard index (0-based, unique per node). "
            "node_index=0 is the bootstrap node. Default: 0."
        ),
    )
    p.add_argument(
        "--port",
        type=int,
        default=9000,
        help="TCP port for the libp2p listener. Default: 9000.",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=Path("my.env"),
        help="Output .env file path. Default: my.env",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="PyTorch compute device. Use 'mps' on Apple Silicon, 'cuda' on NVIDIA. Default: cpu.",
    )
    p.add_argument(
        "--relay-addr",
        action="append",
        dest="relay_addrs",
        default=[],
        metavar="MULTIADDR",
        help=(
            "libp2p relay multiaddress for NAT traversal. "
            "Can be specified multiple times. Requires --enable-relay."
        ),
    )
    p.add_argument(
        "--enable-relay",
        action="store_true",
        help="Enable libp2p circuit-relay for NAT traversal (needed for home internet nodes).",
    )
    p.add_argument(
        "--bootstrap-peer",
        action="append",
        dest="extra_bootstrap_peers",
        default=[],
        metavar="MULTIADDR",
        help=(
            "Override or add bootstrap peer multiaddresses "
            "(supplements manifest bootstrap_peers). "
            "Required for node_index > 0 when the manifest has no bootstrap_peers."
        ),
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Directory containing run manifest JSON files. Default: <repo>/runs/",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without creating any files.",
    )
    return p.parse_args()


def _print_env_file(env: dict[str, str]) -> None:
    """Pretty-print env vars to stdout."""
    for key, value in env.items():
        print(f"  {key}={value}")


def _docker_command(env_file: Path, image: str = "swarmtune/node:latest") -> str:
    """Return a docker run command string for this node."""
    return (
        f"docker run --rm \\\n"
        f"  --env-file {env_file} \\\n"
        f"  -p ${{SWARM_PORT:-9000}}:9000 \\\n"
        f"  -v ./checkpoints:/app/checkpoints \\\n"
        f"  {image}"
    )


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------
    try:
        manifest = RunManifest.load_by_id(args.run_id, runs_dir=args.runs_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Swarm-Tune Join — {manifest.run_id}")
    print(f"  {manifest.description}")
    print(f"  Model: {manifest.model_name}  |  "
          f"Dataset: {manifest.dataset_name}/{manifest.dataset_config}")
    print(f"  Rounds: {manifest.num_rounds}  |  "
          f"Shards: {manifest.num_shards}  |  Min peers: {manifest.min_peers}")

    # ------------------------------------------------------------------
    # Validate node_index
    # ------------------------------------------------------------------
    node_index: int = args.node_index
    if not (0 <= node_index < manifest.num_shards):
        print(
            f"\nerror: --node-index={node_index} is out of range. "
            f"Run '{manifest.run_id}' has {manifest.num_shards} shards "
            f"(valid: 0 to {manifest.num_shards - 1}).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n  Assigning: node_index={node_index} / {manifest.num_shards}  |  port={args.port}")

    # ------------------------------------------------------------------
    # Build env dict
    # ------------------------------------------------------------------
    extra: dict[str, str] = {"SWARM_DEVICE": args.device}

    if args.enable_relay:
        extra["SWARM_ENABLE_RELAY"] = "true"
        extra["SWARM_ENABLE_HOLE_PUNCHING"] = "true"

    if args.relay_addrs:
        import json
        extra["SWARM_RELAY_ADDRS"] = json.dumps(args.relay_addrs)

    # Merge manifest bootstrap peers + any extra --bootstrap-peer flags.
    all_bootstrap = list(manifest.bootstrap_peers) + list(args.extra_bootstrap_peers)
    if all_bootstrap:
        import json
        extra["SWARM_BOOTSTRAP_PEERS"] = json.dumps(all_bootstrap)

    try:
        env = manifest.to_env(node_index, port=args.port)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    env.update(extra)

    # ------------------------------------------------------------------
    # Node-0 / non-0 guidance
    # ------------------------------------------------------------------
    if node_index == 0:
        print("\n  Role: BOOTSTRAP NODE (node_index=0)")
        print("  This node is the entry point for other participants.")
        print("  After starting, copy the multiaddr from the startup log and")
        print("  share it with other participants so they can set --bootstrap-peer.")
        print("  Look for: '[multiaddr] /ip4/<your-ip>/tcp/9000/p2p/12D3KooW...'")
    else:
        print(f"\n  Role: PARTICIPANT NODE (node_index={node_index})")
        if not all_bootstrap:
            print("\n  WARNING: No bootstrap peers configured.")
            print("  Once node_0 starts, re-run this command with:")
            print("    --bootstrap-peer /ip4/<node0-ip>/tcp/9000/p2p/<peer-id>")
            print("  Or add SWARM_BOOTSTRAP_PEERS to your .env file manually.")

    # ------------------------------------------------------------------
    # Write .env file (or dry-run)
    # ------------------------------------------------------------------
    env_file: Path = args.env_file

    if args.dry_run:
        print(f"\n  [dry-run] Would write {env_file}:")
        _print_env_file(env)
    else:
        manifest.write_env_file(env_file, node_index, port=args.port, extra=extra)
        print(f"\n  Written: {env_file}")

    # ------------------------------------------------------------------
    # Print startup instructions
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  START WITH DOCKER:")
    print("=" * 60)
    print()
    print(_docker_command(env_file))
    print()
    print("=" * 60)
    print("  OR START DIRECTLY (requires pip install swarm-tune):")
    print("=" * 60)
    print()
    print(f"  export $(grep -v '^#' {env_file} | xargs)")
    print("  swarm-tune")
    print()
    print(f"  Metrics dashboard: http://localhost:{args.port + 100}/metrics")
    print()
    print("  After training completes, benchmark your checkpoint:")
    print(f"  make benchmark CHECKPOINT=checkpoints/node_{node_index}_final.pt")
    print()


if __name__ == "__main__":
    main()
