#!/usr/bin/env bash
# deploy_relay_fly.sh — One-shot fly.io relay deployment for Swarm-Tune
#
# Usage:  bash scripts/deploy_relay_fly.sh [--app <name>]
#
# Prerequisites:
#   1. flyctl installed:  curl -L https://fly.io/install.sh | sh
#   2. fly account:       fly auth login  (free signup at fly.io)
#
# What this does:
#   1. Creates the fly.io app (idempotent — safe to re-run)
#   2. Generates a random Ed25519 seed and stores it as a fly secret
#   3. Deploys the relay container (relay mode, no GPU, no model loading)
#   4. Waits for the node to print its multiaddr
#   5. Optionally bakes the multiaddr into all run manifests
set -euo pipefail

APP="${1:-swarm-tune-relay}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─── helpers ──────────────────────────────────────────────────────────────────

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

require_cmd() {
    if ! command -v "$1" &>/dev/null; then
        red "Error: '$1' is not installed."
        echo "  Install flyctl: curl -L https://fly.io/install.sh | sh"
        exit 1
    fi
}

require_cmd fly
require_cmd openssl

# ─── 1. Create the app (idempotent) ───────────────────────────────────────────
bold "Step 1/4 — Creating fly.io app '$APP' (skips if it already exists)"
fly apps create "$APP" 2>/dev/null && green "App created." || echo "App already exists — continuing."

# ─── 2. Set the node key seed ─────────────────────────────────────────────────
bold "Step 2/4 — Setting SWARM_NODE_KEY_SEED as a fly secret"
SEED="$(openssl rand -hex 32)"
fly secrets set "SWARM_NODE_KEY_SEED=$SEED" -a "$APP"
green "Secret set. The relay will have a stable peer ID across redeploys."

# ─── 3. Deploy ────────────────────────────────────────────────────────────────
bold "Step 3/4 — Deploying relay container (this builds the Docker image — ~2 min)"
cd "$REPO_ROOT"
fly deploy --config fly.toml --app "$APP" --remote-only

# ─── 4. Extract the multiaddr ─────────────────────────────────────────────────
bold "Step 4/4 — Waiting for relay to print its multiaddr (up to 60 s)"
MULTIADDR=""
for i in $(seq 1 12); do
    MULTIADDR=$(fly logs -a "$APP" 2>/dev/null | grep -oP '/ip4/[^"]+' | tail -1 || true)
    if [[ -n "$MULTIADDR" ]]; then break; fi
    echo "  Waiting ($i/12)..."
    sleep 5
done

if [[ -z "$MULTIADDR" ]]; then
    red "Could not extract multiaddr from logs. Check 'fly logs -a $APP' manually."
    red "Then run:  make set-bootstrap PEER=\"/ip4/<ip>/tcp/9000/p2p/12D3KooW...\""
    exit 0
fi

echo ""
green "Relay is live!"
echo ""
bold "  Bootstrap multiaddr:"
echo "  $MULTIADDR"
echo ""

# ─── 5. Optionally bake into manifests ────────────────────────────────────────
read -r -p "Bake this address into runs/*.json and commit? [y/N] " CONFIRM
if [[ "${CONFIRM,,}" == "y" ]]; then
    cd "$REPO_ROOT"
    if [[ -f scripts/set_bootstrap.py ]]; then
        python scripts/set_bootstrap.py --peer "$MULTIADDR"
        git add runs/
        git commit -m "chore: set fly.io relay bootstrap peer ($APP)"
        green "Committed. Push with: git push"
    else
        red "scripts/set_bootstrap.py not found — update runs/*.json manually."
    fi
fi

echo ""
bold "Done. Share this multiaddr with swarm participants:"
echo "  $MULTIADDR"
echo ""
echo "Participants set it in my.env as:"
echo "  SWARM_BOOTSTRAP_PEERS=$MULTIADDR"
