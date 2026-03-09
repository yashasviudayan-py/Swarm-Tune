<p align="center">
  <img src="assets/logo.svg" alt="Swarm-Tune" width="140"/>
</p>

<h1 align="center">Swarm-Tune</h1>

<p align="center">
  <strong>The BitTorrent of AI training.</strong><br>
  Pool commodity GPUs over the internet to fine-tune models that don't fit on any single machine.
</p>

<p align="center">
  <a href="https://github.com/yashasviudayan-py/Swarm-Tune/actions/workflows/ci.yml">
    <img src="https://github.com/yashasviudayan-py/Swarm-Tune/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://github.com/yashasviudayan-py/Swarm-Tune/actions/workflows/chaos.yml">
    <img src="https://github.com/yashasviudayan-py/Swarm-Tune/actions/workflows/chaos.yml/badge.svg" alt="Chaos Tests">
  </a>
  <a href="https://hub.docker.com/r/yashasviudayan/swarm-tune">
    <img src="https://img.shields.io/docker/v/yashasviudayan/swarm-tune?label=docker&logo=docker&logoColor=white&color=2496ED" alt="Docker Hub">
  </a>
  <a href="https://github.com/yashasviudayan-py/Swarm-Tune/releases/tag/v1.0.0">
    <img src="https://img.shields.io/badge/release-v1.0.0-success" alt="v1.0.0">
  </a>
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://libp2p.io/">
    <img src="https://img.shields.io/badge/libp2p-0.6.0-blueviolet" alt="libp2p">
  </a>
  <a href="tests/">
    <img src="https://img.shields.io/badge/tests-110%20passing-brightgreen" alt="110 tests">
  </a>
  <a href="https://mypy.readthedocs.io/">
    <img src="https://img.shields.io/badge/mypy-strict-2a6db5" alt="mypy strict">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="ruff">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
  </a>
</p>

---

## Overview

Swarm-Tune lets a group of people pool their gaming GPUs over the internet to collaboratively fine-tune a large language model — with no data center, no cloud bill, and no central authority.

Each participant runs one node. Every node holds a shard of the model and a shard of the dataset, trains locally, extracts raw gradients, and broadcasts them to peers over a libp2p P2P network. The swarm runs Federated Averaging every round; all nodes stay in sync.

```
20 participants × RTX 3090 (24 GB VRAM) = 480 GB pooled VRAM
LLaMA 3 70B requires ~140 GB → fits in the swarm, not on any single machine
```

The system is **fully decentralized** — no master node, no tracker, no coordinator. Nodes discover each other via Kademlia DHT, tolerate stragglers and failures with timeout-based partial aggregation, and defend against adversarial participants with gradient validation and Sybil resistance. **Competing swarms** can race on the same model and dataset; winner is determined by perplexity, publicly verifiable by anyone.

---

## Quick Start

**Join an existing training run** (requires Docker):

```bash
# Step 1 — Generate your .env and startup command
python scripts/join.py --run-id gpt2-wikitrain-001 --node-index <N>

# Step 2 — Start your node
docker run --rm --env-file my.env \
  -p 9000:9000 \
  -v ./checkpoints:/app/checkpoints \
  yashasviudayan/swarm-tune:latest
```

**Run the 6-node local simulation** (no internet required):

```bash
git clone https://github.com/yashasviudayan-py/Swarm-Tune
cd Swarm-Tune
make sim-up        # starts 5 honest + 1 adversarial node in Docker
make sim-logs      # stream structured JSON logs
make sim-kill-node NODE=swarm_node_2   # chaos: kill a node mid-training
```

**Install the Python package:**

```bash
pip install swarm-tune
swarm-tune --help
```

---

## How It Works

Each training round:

```
Each node independently:
  1. Sample a mini-batch from its local data shard
  2. Forward pass → compute loss → loss.backward()
  3. Extract param.grad tensors, validate (NaN/Inf/norm bounds)
  4. Compress → serialize (SWRM wire format) → chunk into ≤60 KB frames
  5. Broadcast chunks over libp2p FloodSub

Simultaneously, receive from peers:
  6. Reassemble chunks → deserialize (weights_only=True) → decompress
  7. Validate each peer gradient (reject NaN/Inf/outliers/wrong shape)
  8. Submit to TimeoutAggregator (hard 30s window)

After timeout or quorum reached:
  9. Weighted FedAvg (weighted by dataset_size per peer)
  10. Apply averaged gradients → optimizer.step()

Straggler handling:
  - ≥ min_peers respond  → commit round
  - < min_peers respond  → fall back to local gradient, no round wasted
  - Dead nodes evicted via heartbeat after 60s, welcomed back on rejoin
```

### Why Not PyTorch DDP?

`DistributedDataParallel` assumes microsecond data-center latency, homogeneous always-on nodes, and NCCL/Gloo transport. The internet is none of those things. This project builds the gradient exchange layer from scratch: custom wire protocol, chunked framing (Noise protocol 65 KB frame limit), timeout aggregation, and federated averaging — all of which DDP abstracts away, but which you need to understand to do distributed training in the wild.

---

## Features

| Capability | Detail |
|---|---|
| **Any HuggingFace model** | `SWARM_MODEL_NAME=gpt2` or `llama3` — config-only, no code change |
| **Any HuggingFace dataset** | Deterministic sharding via `dataset.shard()` |
| **NAT traversal** | libp2p circuit-relay + dcutr hole-punching for home internet nodes |
| **Straggler tolerance** | Timeout-based partial aggregation — one dead node never blocks the swarm |
| **Gradient poisoning defense** | NaN/Inf/RMS-norm validation before any peer gradient reaches the averager |
| **Sybil resistance** | Subnet contribution cap (`/24`) + per-peer rejection rate ban |
| **Transparent chunking** | Noise protocol 65 KB frame limit handled automatically |
| **Live dashboard** | Static HTML, no build step — force-directed peer graph, loss curves, bytes throughput |
| **Zero-config join** | `python scripts/join.py --run-id X --node-index N` generates your `.env` |
| **Competition mode** | Two swarms race on the same model/dataset; winner by perplexity, publicly verifiable |
| **Relay node** | `SWARM_RELAY_MODE=true` — run a bootstrap VPS with no training, stable peer ID |
| **Checkpoint tools** | Reconstruct full model from shards; publish to HuggingFace Hub with model card |
| **110 tests** | Unit, integration, chaos (fault injection, adversarial rejection, node drop/rejoin) |

---

## Architecture

```
src/swarm_tune/
├── config/settings.py          NodeSettings — all SWARM_ env vars, pydantic validators
├── runs/manifest.py            RunManifest — training campaign definition, .env generator
└── node/
    ├── main.py                 SwarmNode training loop + RelayNode + CLI entrypoint
    ├── metrics.py              MetricsStore + anyio TCP /metrics sidecar
    ├── p2p/
    │   ├── discovery.py        libp2p host, Ed25519 keys, mDNS, Kademlia DHT, relay dialing
    │   ├── gossip.py           GossipProtocol — FloodSub, chunked framing, eviction loop
    │   ├── heartbeat.py        Liveness signals + stale peer eviction
    │   └── peer_selector.py    AllPeersSelector + BanList (rejection rate tracking)
    ├── trainer/
    │   ├── model.py            ModelShard — HF AutoModelForCausalLM or toy MLP + layer sharding
    │   ├── data.py             DataShardLoader (.pt) + HFDataShardLoader (HF datasets)
    │   ├── gradient.py         GradientExtractor — extract + validate param.grad tensors
    │   ├── serializer.py       GradientSerializer — SWRM wire format, weights_only=True
    │   └── compressor.py       Compressor protocol: Identity (now) → TopK (bandwidth scale-up)
    └── aggregator/
        ├── averaging.py        GradientAverager — weighted FedAvg + Sybil subnet cap
        ├── timeout.py          TimeoutAggregator — partial aggregation + rate limiting
        └── strategy.py         AggregationStrategy: Flat (now) → Hierarchical (100+ nodes)

runs/
├── gpt2-wikitrain-001.json     4-node data-parallel GPT-2/WikiText-103 run
├── gpt2-competition-001.json   50-round competition manifest
└── gpt2-competition-2v2.json   2-node-per-team competition manifest

scripts/
├── join.py                     Zero-config participant onboarding
├── run_competition.py          Orchestrate two-team competition, write JSON result
├── set_bootstrap.py            Bake relay VPS multiaddr into all manifests
├── reconstruct_checkpoint.py   Merge/average shard checkpoints → full model
├── publish_checkpoint.py       Push checkpoint + model card to HuggingFace Hub
├── benchmark.py                Perplexity evaluation on WikiText-103 test split
└── generate_shards.py          Generate synthetic .pt shards for local simulation

docker/
├── Dockerfile                  Multi-stage build, non-root user, dynamic health check
├── docker-compose.yml          6-node simulation (5 honest + 1 adversarial)
└── docker-compose.relay.yml    Single relay/bootstrap node for VPS deployment

dashboard/index.html            Static vanilla-JS dashboard — no build step
```

### Extensibility Abstractions

Three `Protocol` interfaces designed for zero-friction scale-up. The training loop never changes — only the implementation behind the protocol.

| Protocol | Now (≤30 nodes) | Scale-up (100+ nodes) | How to swap |
|---|---|---|---|
| `Compressor` | `IdentityCompressor` (no-op) | `TopKCompressor` (~50× bandwidth reduction at 1%) | `SWARM_COMPRESSION=topk` |
| `PeerSelector` | `AllPeersSelector` + `BanList` | `ClusterPeerSelector` | `SWARM_AGGREGATION_STRATEGY=hierarchical` |
| `AggregationStrategy` | `FlatAggregation` | `HierarchicalAggregation` | `SWARM_AGGREGATION_STRATEGY=hierarchical` |

---

## Public Deployment

### Hosting a relay/bootstrap node (VPS — $5/month)

A relay node gives participants a stable public multiaddr to connect to. It runs P2P only — no model loading, no training.

```bash
# On your VPS
echo "SWARM_NODE_KEY_SEED=your_long_secret_here" > relay.env
make relay-up                                  # docker compose -f docker/docker-compose.relay.yml up -d
make relay-logs | grep Multiaddr               # copy /ip4/<vps-ip>/tcp/9000/p2p/12D3KooW...
```

Once you have the multiaddr, bake it into all run manifests so `join.py` works with zero extra flags:

```bash
make set-bootstrap PEER="/ip4/<vps-ip>/tcp/9000/p2p/12D3KooW..."
git commit -am "chore: set bootstrap peer" && git push
```

After that, any participant worldwide can join with a single command — no `--bootstrap-peer` flag needed.

### Participant onboarding (their machine)

```bash
git clone https://github.com/yashasviudayan-py/Swarm-Tune
python scripts/join.py --run-id gpt2-wikitrain-001 --node-index 2 --device cuda
# → writes my.env
# → prints: docker run --env-file my.env -p 9000:9000 yashasviudayan/swarm-tune:latest
```

### Running a competition

```bash
make competition \
  COMPETITION_ID=gpt2-comp-001 \
  TEAM_A_ID=team-alpha  TEAM_A_CHECKPOINT=ckpts/alpha.pt \
  TEAM_B_ID=team-beta   TEAM_B_CHECKPOINT=ckpts/beta.pt
```

Results are written to `results/competition_result.json`. Anyone can independently verify by running `make benchmark CHECKPOINT=<downloaded-checkpoint>` against a published HuggingFace Hub checkpoint.

---

## Development

**Prerequisites:** Python 3.12+, Docker, `brew install gmp` (macOS) or `apt install libgmp-dev` (Linux)

```bash
git clone https://github.com/yashasviudayan-py/Swarm-Tune
cd Swarm-Tune
make bootstrap          # venv + deps + pre-commit hooks

make check              # ruff lint + format check + mypy --strict
make test               # 110 unit + integration tests
make test-chaos         # 10 fault injection tests (node drop, adversarial, etc.)
make sim-up             # 6-node Docker simulation
```

### All Makefile targets

```bash
# Code quality
make check              # lint + format + types
make format             # auto-format with ruff

# Testing
make test               # unit + integration (fast)
make test-chaos         # fault injection (slow, real timeouts)
make test-all           # everything
make coverage           # HTML coverage report → htmlcov/

# Simulation
make sim-up             # 6-node Docker swarm
make sim-down           # stop and remove containers
make sim-logs           # tail all container logs
make sim-kill-node NODE=swarm_node_2

# Training runs
make join RUN_ID=gpt2-wikitrain-001 NODE_INDEX=0
make reconstruct CHECKPOINT_DIR=checkpoints/ MODEL=gpt2
make publish CHECKPOINT=checkpoints/full_model.pt REPO_ID=user/model
make benchmark CHECKPOINT=checkpoints/node_0_final.pt

# Competition
make competition COMPETITION_ID=... TEAM_A_ID=... TEAM_A_CHECKPOINT=... \
                 TEAM_B_ID=... TEAM_B_CHECKPOINT=...

# Public deployment
make relay-up           # start relay node (requires relay.env)
make relay-logs         # tail relay logs, copy the multiaddr
make set-bootstrap PEER="/ip4/.../tcp/9000/p2p/12D3KooW..."
```

---

## Project Status

All 8 phases complete. Production-audited. v1.0.0 released.

| Phase | Description | Status |
|---|---|---|
| 1 | P2P Network — libp2p, Ed25519, mDNS, Kademlia, heartbeat, peer eviction | ✅ |
| 2 | Gradient Extraction — SWRM protocol, `weights_only=True`, FedAvg | ✅ |
| 3 | Gradient Sync — FloodSub, chunked framing, `TimeoutAggregator` | ✅ |
| 4 | Docker Simulation — 6-node sim, chaos tests, adversarial rejection | ✅ |
| 5 | Internet Deployment — HF models/datasets, NAT traversal, Sybil resistance, /metrics | ✅ |
| 6 | Live Dashboard — force-directed graph, persistent loss curves, bytes tracking | ✅ |
| 7 | Distribution — `RunManifest`, `join.py`, checkpoint reconstruction, HF Hub publish | ✅ |
| 8 | Competition — `run_competition.py`, 2v2 manifest, `make competition`, relay node | ✅ |

**Test coverage:** 110 tests (unit + integration + chaos). `mypy --strict` clean. `ruff` clean.

**Docker Hub:** [`yashasviudayan/swarm-tune`](https://hub.docker.com/r/yashasviudayan/swarm-tune) — `latest` and `1.0.0` tags available for `linux/amd64` and `linux/arm64`.

---

## Security Properties

| Property | Mechanism |
|---|---|
| Cryptographic peer identity | Ed25519 key pairs via libp2p — spoofing requires breaking the key |
| No pickle deserialization | `torch.load(..., weights_only=True)` on all peer data |
| Wire format validation | SWRM magic bytes checked before any deserialization |
| Gradient poisoning defense | NaN/Inf/RMS-norm bounds enforced; poisoned gradients rejected before the averager |
| Sybil resistance | `/24` subnet contribution cap in FedAvg; configurable prefix |
| Reputation system | Per-peer rejection rate tracking; temporary ban on threshold exceeded |
| Rate limiting | One gradient submission per peer per round |
| Path traversal prevention | `node_id` sanitized via regex; `checkpoint_dir` rejects system paths at startup |
| Atomic checkpoints | `.tmp` + `os.replace()` — crash cannot corrupt an existing checkpoint |
| SIGTERM-safe shutdown | Final checkpoint wrapped in `CancelScope(shield=True)` |
| Bootstrap dial timeout | 10s per-peer via `anyio.move_on_after()` — unresponsive relay can't block startup |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Networking | `libp2p` 0.6.0 — Ed25519 peer IDs, FloodSub, mDNS, Kademlia DHT |
| Deep learning | `PyTorch` ≥2.3 — direct `param.grad` access, MPS on Apple Silicon |
| Models | HuggingFace `transformers` ≥4.40 — any `AutoModelForCausalLM` |
| Datasets | HuggingFace `datasets` ≥2.20 — deterministic sharding, streaming |
| Async runtime | `anyio` + `trio` — libp2p requires trio; anyio keeps the rest backend-agnostic |
| Config | `pydantic-settings` — fail loudly at startup, not at runtime |
| Logging | `structlog` — JSON in Docker, human-readable console locally |
| Orchestration | Docker + docker-compose — reproducible multi-node simulation |
| Language | Python 3.12 — `mypy --strict` throughout |

---

## Contributing

Read [`CLAUDE.md`](CLAUDE.md) before contributing — it is the source of truth for every architecture and security decision in this codebase.

Three rules that are non-negotiable:
1. **No central server.** Nodes coordinate only via libp2p gossip.
2. **No standard DDP.** Gradients are extracted from `param.grad` and exchanged manually.
3. **Never `pickle.loads()` peer data.** Always `weights_only=True`.

---

## License

MIT. See [`LICENSE`](LICENSE).
