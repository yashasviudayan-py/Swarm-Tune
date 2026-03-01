#!/usr/bin/env bash
# =============================================================
# Developer bootstrap script.
# Idempotent — safe to run multiple times.
# =============================================================
set -euo pipefail

echo "==> Checking Python version..."
python3 --version
PYTHON_MIN="3.12"
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$(printf '%s\n' "$PYTHON_MIN" "$PYTHON_VER" | sort -V | head -n1)" != "$PYTHON_MIN" ]]; then
    echo "ERROR: Python $PYTHON_MIN+ required, found $PYTHON_VER" >&2
    exit 1
fi

echo "==> Creating virtual environment..."
python3 -m venv .venv

echo "==> Upgrading pip..."
.venv/bin/pip install --quiet --upgrade pip

echo "==> Installing project with dev dependencies..."
.venv/bin/pip install --quiet -e ".[dev]"

echo "==> Installing pre-commit hooks..."
.venv/bin/pre-commit install

echo "==> Generating data shards for simulation..."
.venv/bin/python scripts/generate_shards.py

echo ""
echo "Bootstrap complete!"
echo "Activate environment: source .venv/bin/activate"
echo "Run tests:            make test"
echo "Start simulation:     make sim-up"
