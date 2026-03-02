.DEFAULT_GOAL := help
SHELL := /bin/bash
PYTHON := python3.12
VENV := .venv
PIP := $(VENV)/bin/pip
BIN := $(VENV)/bin

# ============================================================
# Help
# ============================================================
.PHONY: help
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ============================================================
# Environment setup
# ============================================================
.PHONY: venv
venv:  ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: install
install: venv  ## Install all dependencies (including dev)
	$(PIP) install -e ".[dev]"

.PHONY: install-hooks
install-hooks: install  ## Install pre-commit hooks
	$(BIN)/pre-commit install

.PHONY: bootstrap
bootstrap: install install-hooks  ## Full developer bootstrap (venv + deps + hooks)
	@echo "Bootstrap complete. Activate with: source $(VENV)/bin/activate"

# ============================================================
# Code quality
# ============================================================
.PHONY: lint
lint:  ## Run ruff linter
	$(BIN)/ruff check src/ tests/

.PHONY: format
format:  ## Format code with ruff
	$(BIN)/ruff format src/ tests/

.PHONY: format-check
format-check:  ## Check formatting without modifying files (mirrors CI)
	$(BIN)/ruff format --check src/ tests/

.PHONY: typecheck
typecheck:  ## Run mypy type checker
	$(BIN)/mypy src/swarm_tune

.PHONY: check
check: lint format-check typecheck  ## Run all static checks (lint + format + types)

# ============================================================
# Testing
# ============================================================
.PHONY: test
test:  ## Run unit and integration tests
	$(BIN)/pytest -m "unit or integration" -v

.PHONY: test-unit
test-unit:  ## Run unit tests only
	$(BIN)/pytest -m unit -v

.PHONY: test-integration
test-integration:  ## Run integration tests only
	$(BIN)/pytest -m integration -v

.PHONY: test-chaos
test-chaos:  ## Run chaos / fault-injection tests (slow)
	$(BIN)/pytest -m chaos -v --timeout=120

.PHONY: test-all
test-all:  ## Run the full test suite including chaos tests
	$(BIN)/pytest -v --timeout=120

.PHONY: coverage
coverage:  ## Generate HTML coverage report
	$(BIN)/pytest --cov=src/swarm_tune --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ============================================================
# Docker simulation
# ============================================================
.PHONY: sim-up
sim-up:  ## Start the 5-node swarm simulation
	docker compose -f docker/docker-compose.yml up --build

.PHONY: sim-down
sim-down:  ## Stop the swarm simulation and remove containers
	docker compose -f docker/docker-compose.yml down -v

.PHONY: sim-logs
sim-logs:  ## Tail logs from all simulated nodes
	docker compose -f docker/docker-compose.yml logs -f

.PHONY: sim-kill-node
sim-kill-node:  ## Kill a random node (chaos test trigger). Usage: make sim-kill-node NODE=node_2
	docker compose -f docker/docker-compose.yml stop $(NODE)

# ============================================================
# Data
# ============================================================
.PHONY: shards
shards:  ## Generate synthetic training data shards for simulation
	$(BIN)/python scripts/generate_shards.py

# ============================================================
# Cleanup
# ============================================================
.PHONY: clean
clean:  ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ .coverage

.PHONY: clean-all
clean-all: clean sim-down  ## Remove build artifacts, caches, and Docker volumes
	rm -rf $(VENV)
