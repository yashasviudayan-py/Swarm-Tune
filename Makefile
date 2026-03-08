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
sim-up:  ## Start the 5-node swarm simulation (generates shards if missing)
	@test -f data/shards/shard_0.pt || python scripts/generate_shards.py
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
# Benchmark
# ============================================================
CHECKPOINT ?= checkpoints/node_0_final.pt
MODEL ?= gpt2

.PHONY: benchmark
benchmark:  ## Evaluate checkpoint perplexity. Usage: make benchmark CHECKPOINT=path/to/ckpt.pt
	$(BIN)/python scripts/benchmark.py --checkpoint $(CHECKPOINT) --model-name $(MODEL)

# ============================================================
# Phase 7 — Join / Distribute / Publish
# ============================================================
RUN_ID ?= gpt2-wikitrain-001
NODE_INDEX ?= 0
NODE_PORT ?= 9000
ENV_FILE ?= my.env
CHECKPOINT_DIR ?= checkpoints/
OUTPUT ?= checkpoints/full_model.pt
STRATEGY ?= merge

.PHONY: join
join:  ## Join a training run. Usage: make join RUN_ID=gpt2-wikitrain-001 NODE_INDEX=0
	$(BIN)/python scripts/join.py \
		--run-id $(RUN_ID) \
		--node-index $(NODE_INDEX) \
		--port $(NODE_PORT) \
		--env-file $(ENV_FILE)

.PHONY: reconstruct
reconstruct:  ## Merge shard checkpoints into a full model. Usage: make reconstruct CHECKPOINT_DIR=checkpoints/ MODEL=gpt2
	$(BIN)/python scripts/reconstruct_checkpoint.py \
		--checkpoint-dir $(CHECKPOINT_DIR) \
		--model-name $(MODEL) \
		--strategy $(STRATEGY) \
		--output $(OUTPUT)

.PHONY: publish
publish:  ## Publish checkpoint to HuggingFace Hub. Usage: make publish CHECKPOINT=full_model.pt REPO_ID=user/repo
	$(BIN)/python scripts/publish_checkpoint.py \
		--checkpoint $(CHECKPOINT) \
		--model-name $(MODEL) \
		--repo-id $(REPO_ID) \
		--run-id $(RUN_ID)

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
