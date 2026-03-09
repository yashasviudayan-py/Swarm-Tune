#!/usr/bin/env python3
"""
set_bootstrap.py — Bake a relay node's multiaddr into all run manifests.

After you spin up the relay VPS and note its multiaddr from the logs, run:

    python scripts/set_bootstrap.py --peer "/ip4/1.2.3.4/tcp/9000/p2p/12D3KooW..."

This updates bootstrap_peers in every runs/*.json file so that
`python scripts/join.py` works with zero extra configuration for participants.

Usage:
    python scripts/set_bootstrap.py --peer MULTIADDR [--runs-dir PATH] [--dry-run]

Options:
    --peer MULTIADDR      libp2p multiaddr of the relay/bootstrap node (required)
    --runs-dir PATH       Directory containing *.json manifests (default: runs/)
    --dry-run             Print what would change without writing files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_RUNS_DIR = _REPO_ROOT / "runs"

# Multiaddr must start with /ip4 or /ip6 and contain /p2p/
_MULTIADDR_PREFIX = ("/ip4/", "/ip6/", "/dns4/", "/dns6/")


def _validate_multiaddr(peer: str) -> None:
    if not any(peer.startswith(p) for p in _MULTIADDR_PREFIX):
        print(
            f"error: --peer does not look like a multiaddr.\n"
            f"  Expected something like: /ip4/1.2.3.4/tcp/9000/p2p/12D3KooW...\n"
            f"  Got: {peer!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    if "/p2p/" not in peer:
        print(
            f"error: --peer is missing the /p2p/<peer-id> component.\n"
            f"  Copy the full multiaddr from the relay startup log.\n"
            f"  Got: {peer!r}",
            file=sys.stderr,
        )
        sys.exit(1)


def _update_manifest(path: Path, peer: str, dry_run: bool) -> bool:
    """
    Set bootstrap_peers to [peer] in a single manifest JSON file.
    Returns True if the file was changed (or would be changed in dry-run).
    """
    with path.open() as f:
        data = json.load(f)

    current = data.get("bootstrap_peers", [])
    if current == [peer]:
        print(f"  {path.name}  (already set, skipped)")
        return False

    data["bootstrap_peers"] = [peer]

    if dry_run:
        print(f"  [dry-run] {path.name}  bootstrap_peers: {current!r}  →  {[peer]!r}")
    else:
        # Write atomically: temp file → rename
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        tmp.rename(path)
        print(f"  {path.name}  updated  bootstrap_peers → [{peer!r}]")

    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bake a relay node multiaddr into all run manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--peer",
        required=True,
        metavar="MULTIADDR",
        help="Full libp2p multiaddr of the relay/bootstrap node.",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=_DEFAULT_RUNS_DIR,
        help="Directory containing run manifest JSON files. Default: runs/",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing any files.",
    )
    args = p.parse_args()

    _validate_multiaddr(args.peer)

    runs_dir: Path = args.runs_dir
    if not runs_dir.is_dir():
        print(f"error: runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    manifests = sorted(runs_dir.glob("*.json"))
    if not manifests:
        print(f"error: no *.json files found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Setting bootstrap_peers to:\n    {args.peer}\n")
    print(f"  Manifests in {runs_dir}/:")

    changed = 0
    for manifest_path in manifests:
        try:
            if _update_manifest(manifest_path, args.peer, args.dry_run):
                changed += 1
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            print(f"  WARNING: could not update {manifest_path.name}: {exc}")

    action = "Would update" if args.dry_run else "Updated"
    print(f"\n  {action} {changed}/{len(manifests)} manifest(s).")

    if not args.dry_run and changed:
        print("\n  Next steps:")
        print("    git add runs/")
        print('    git commit -m "chore: set bootstrap peer"')
        print("    git push")
        print(
            "\n  Participants can now join with zero extra flags:\n"
            "    python scripts/join.py --run-id <run-id> --node-index N"
        )


if __name__ == "__main__":
    main()
