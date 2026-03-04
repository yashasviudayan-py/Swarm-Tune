# Swarm-Tune

> **The BitTorrent of AI Training.**
> 20 gaming PCs. 480 GB of pooled VRAM. One model that nobody could train alone.

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://libp2p.io/"><img src="https://img.shields.io/badge/libp2p-0.6.0-blueviolet" alt="libp2p"></a>
  <a href="https://anyio.readthedocs.io/"><img src="https://img.shields.io/badge/async-anyio%20%7C%20trio-009688" alt="anyio + trio"></a>
  <a href="https://docs.pydantic.dev/latest/concepts/pydantic_settings/"><img src="https://img.shields.io/badge/config-pydantic--settings-E92063?logo=pydantic&logoColor=white" alt="pydantic-settings"></a>
  <a href="https://www.structlog.org/"><img src="https://img.shields.io/badge/logging-structlog-555555" alt="structlog"></a>
  <br>
  <a href="https://mypy.readthedocs.io/"><img src="https://img.shields.io/badge/mypy-strict-2a6db5" alt="mypy strict"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="ruff"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/tests-54%20passing-brightgreen" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

---

## The Problem

Training a large language model or fine-tuning a foundation model requires VRAM that almost nobody has:

| Model | VRAM Required | Who can afford it? |
|---|---|---|
| LLaMA 3 8B (fp16) | ~16 GB | A few enthusiasts |
| LLaMA 3 70B (fp16) | ~140 GB | Almost nobody |
| Mixtral 8x22B (fp16) | ~280 GB | Nobody |
| GPT-4 class models | 1+ TB | OpenAI, Google, Microsoft |

The hardware to train at scale costs hundreds of thousands of dollars. Cloud GPU rentals cost $3–$8/hour per H100, and you need dozens of them. The result: AI training is a monopoly held by a handful of corporations.

**Swarm-Tune breaks that monopoly.**

---

## The Idea

What if you could do for AI training what BitTorrent did for file sharing?

In BitTorrent, no single machine needs to store or serve an entire file. Every peer holds a shard, and the swarm collectively delivers the whole. The protocol is resilient: peers join, leave, go offline, come back — and the download continues.

Swarm-Tune applies this exact philosophy to gradient descent.

- **20 people** with RTX 3090s or RTX 4090s (24 GB VRAM each)
- Each loads **a shard of the model** — the part that fits in their 24 GB
- Each trains on **a shard of the data** locally
- Each computes **local gradients** and broadcasts them to the swarm
- The swarm **averages the gradients** and every node updates its weights
- The result: a model that required **480 GB of pooled VRAM** gets trained on commodity hardware, over the internet, with no central authority

No one needs a data center. No one pays $8/hour for H100s. The swarm IS the cluster.

---

## Progress

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | P2P Network Initialization | ✅ Complete |
| **Phase 2** | Local Gradient Extraction | ✅ Complete |
| **Phase 3** | Gradient Synchronization over libp2p GossipSub | ✅ Complete |
| **Phase 4** | Docker Simulation & Chaos Testing | 🔧 In Progress |
| **Phase 5** | Internet Deployment | ⬜ Future |

---

## What's Working Now

### Phase 1 — P2P Network ✅

Every node is a cryptographic identity. The swarm is fully decentralised with no central server.

- **libp2p host** with Ed25519 key pairs — peer IDs are cryptographic, not strings. Spoofing requires breaking the key.
- **mDNS discovery** for local simulation; Kademlia DHT for internet deployment.
- **Bootstrap peer** — the first known address. Not a master, just a door.
- **Heartbeat protocol** — every node publishes a liveness signal over FloodSub every few seconds. Dead nodes are evicted from the active peer set after 20 s. Rejoining nodes are welcomed on the next heartbeat with no special handling.
- **Straggler tolerance** — `TimeoutAggregator` gives each round a hard deadline. If ≥ `min_peers` respond within the window → commit the round. If not → defer. A dead node never blocks the swarm.

**Verified by:** a two-node integration test with real TCP connections, real Ed25519 keys, and real FloodSub pubsub message exchange.

---

### Phase 2 — Local Gradient Extraction ✅

Each node independently computes, serializes, and logs its gradient payload.

- **`DataShardLoader`** — loads `.pt` shard files (`inputs`, `targets`), samples random mini-batches each round, tracks `dataset_size` for weighted FedAvg.
- **`ModelShard`** — wraps a PyTorch model (currently a small MLP; will graduate to a real transformer). Handles forward, backward, and applying averaged gradients via `AdamW`. Checkpoint save/resume with `weights_only=True`.
- **`GradientExtractor`** — extracts `param.grad` tensors after `loss.backward()`, moves them to CPU for serialization. Skips frozen layers.
- **`GradientSerializer`** — SWRM wire format: `[4B magic][4B version][N bytes torch-serialized dict]`. Deserializes with `weights_only=True` (prevents arbitrary code execution). Validates magic, version, and dict structure before returning.
- **`IdentityCompressor`** — no-op compressor (Phases 1–4). `TopKCompressor` is built and tested (ships in Phase 5+ for bandwidth reduction). Swapping compression strategy requires one config value change.
- **`GradientAverager`** — weighted FedAvg: each peer's contribution is weighted by its local dataset size. Peers with more samples contribute proportionally more to the global average.

**Verified by:** a full pipeline integration test — shard load → forward → backward → extract → compress → serialize → deserialize → decompress → validate → apply — plus a 10-round convergence test that asserts loss decreases.

---

### Phase 3 — Gradient Synchronization ✅

Nodes exchange real gradients over libp2p FloodSub. The swarm trains collectively.

- **`GossipProtocol`** — full FloodSub wiring replacing the prior placeholder. Subscribes to the gradient topic, exposes `broadcast_gradient()` and `run_receiver()`.
- **Transparent chunked framing** — libp2p's Noise protocol has a hard 65,535-byte per-frame limit. Gradient payloads (~264 KB for a small MLP) exceed this. The gossip layer silently splits every broadcast into ≤60 KB frames, each tagged with a `transfer_id` + `chunk_index` + `total_chunks` header. The receiver reassembles before dispatch — the training loop sees only complete messages.
- **`GradientMessage` wire format** — `struct`-packed inner header: `sender_id_len (uint32) | round_number (int32) | dataset_size (int64)` + raw sender string + gradient payload bytes.
- **Stale transfer eviction** — partial transfers that never complete (e.g., sender dropped mid-message) are evicted after 60 s, preventing unbounded memory growth.
- **`_on_peer_gradient` pipeline** — for every received message: deserialize → decompress → validate (NaN/Inf/norm) → submit to `TimeoutAggregator`. Any validation failure logs a warning and skips that peer — the round continues.
- **FedAvg representation consistency** — the local gradient is submitted as `decompress(compress(raw_grad))`, matching the representation of every peer gradient that has gone through the full compress→serialize→deserialize→decompress path. Avoids biased FedAvg averages when using `TopKCompressor`.

**Verified by:** a two-node integration test with real TCP connections, real FloodSub pubsub, and real gradient tensor exchange — receiver reconstructs the exact sender tensors.

---

### Phase 4 — Docker Simulation & Chaos Testing 🔧

Infrastructure complete. End-to-end Docker run pending verification.

- **Docker Compose simulation** — 5 honest peer nodes + 1 adversarial node, each in its own container. `make sim-up` auto-generates synthetic data shards if missing, then builds and starts all containers.
- **Deterministic bootstrap address** — `node_0` uses `SWARM_NODE_KEY_SEED=swarm_bootstrap_node_0` to produce a stable peer ID (`12D3KooWJWTRCtfVVBtPkDSjL8iy1ysM5WoQRdd5vLWVMrccePHU`) across restarts. The bootstrap multiaddress includes the `/p2p/<PEER_ID>` suffix required by the Noise handshake.
- **Adversarial node** — `node_5_adversarial` runs with `SWARM_ADVERSARIAL=true`. It trains normally locally but replaces every gradient broadcast with NaN-filled tensors, simulating a gradient-poisoning attack.
- **Poisoning defence** — receiving nodes pass every peer gradient through `GradientExtractor.validate()`. NaN/Inf values and out-of-bounds L2 norms are caught before the gradient reaches the aggregator. The swarm continues training on honest peers' contributions.

**Chaos tests verified:**

| Scenario | Expected Behaviour | Status |
|---|---|---|
| 1 of 3 peers drops mid-round | Partial aggregation on 2 peers, round commits | ✅ |
| All peers drop (total partition) | Round deferred, no crash | ✅ |
| Late peer rejoins next round | Welcomed back, contributes normally | ✅ |
| Peer submits gradient twice (network retry) | Deduplicated — counted once | ✅ |
| Adversarial peer sends NaN gradients | Rejected by validator, swarm continues | ✅ |
| Adversarial peer sends Inf gradients | Rejected by validator, swarm continues | ✅ |
| Adversarial peer sends out-of-bounds norm | Rejected by validator, swarm continues | ✅ |
| Peer sends malformed bytes | Rejected at deserialization, swarm continues | ✅ |
| Peer sends wrong SWRM magic header | Rejected before `torch.load`, swarm continues | ✅ |
| End-to-end: 3 honest + 1 adversarial | 3 contributions averaged; result is finite | ✅ |

---

### Security Gates Implemented

| Gate | Implementation | Status |
|---|---|---|
| Cryptographic peer IDs | Ed25519 via libp2p | ✅ Phase 1 |
| No pickle from peers | `weights_only=True` in `GradientSerializer.deserialize()` | ✅ Phase 2 |
| SWRM magic + version header | Validated before any deserialization | ✅ Phase 2 |
| Payload type validation | Dict structure checked after `torch.load` | ✅ Phase 2 |
| NaN / Inf rejection | `GradientExtractor.validate()` | ✅ Phase 2 |
| L2 norm bounds | Threshold `1e4` per tensor, configurable | ✅ Phase 2 |
| FedAvg representation consistency | Local grad compress→decompress before submission | ✅ Phase 3 |
| Chunked frame re-assembly safety | Stale partial transfers evicted after 60 s | ✅ Phase 3 |
| Adversarial gradient rejection (end-to-end) | NaN/Inf/norm → reject + log, round continues | ✅ Phase 4 |
| Sybil resistance | Limit contribution per IP subnet | ⬜ Phase 5 |
| Eclipse resistance | Maintain diverse-IP peer connections | ⬜ Phase 5 |
| Reputation / temporary bans | Per-peer rejection rate tracking | ⬜ Phase 5 |

---

## How It Works

### The Core Loop

```
┌─────────────────────────────────────────────────────────────┐
│                        SWARM-TUNE LOOP                       │
│                                                              │
│   Node A            Node B            Node C    ...Node N   │
│   ┌──────┐          ┌──────┐          ┌──────┐              │
│   │Shard │          │Shard │          │Shard │              │
│   │  of  │          │  of  │          │  of  │              │
│   │Model │          │Model │          │Model │              │
│   └──┬───┘          └──┬───┘          └──┬───┘              │
│      │                 │                 │                  │
│   Forward           Forward           Forward               │
│   + Backward        + Backward        + Backward            │
│      │                 │                 │                  │
│   Gradients         Gradients         Gradients             │
│      │                 │                 │                  │
│      └────────────────►│◄────────────────┘                  │
│                        │                                    │
│                   P2P Gossip                                 │
│                  (libp2p swarm)                              │
│                        │                                    │
│              Averaged Gradients                              │
│              broadcast to all peers                          │
│                        │                                    │
│   ┌──────┐          ┌──────┐          ┌──────┐              │
│   │Update│          │Update│          │Update│              │
│   │Weights          │Weights          │Weights              │
│   └──────┘          └──────┘          └──────┘              │
│                                                              │
│              All nodes stay in sync. Repeat.                 │
└─────────────────────────────────────────────────────────────┘
```

### Why This is Fundamentally Different from PyTorch DDP

PyTorch's `DistributedDataParallel` is designed for **data center interconnects**: 100 Gbps InfiniBand, microsecond latency, nodes that never go offline. It assumes a controlled, homogeneous, always-on environment.

The real world is none of those things.

Swarm-Tune is built for **the internet**: variable latency, asymmetric bandwidth, nodes that drop offline mid-round, machines with different specs, participants in different countries. The gradient synchronization protocol must handle this gracefully.

### The Straggler Problem (and How We Solve It)

In a naive all-reduce approach, one slow node blocks everyone. If Node 7 goes offline, training halts.

Swarm-Tune uses **timeout-based partial aggregation**:

```
Round N begins. Timeout window: T seconds (default 30s).

Nodes that respond within T  →  their gradients are included.
Nodes that miss the window   →  skipped this round, catch up next.

If ≥ min_peers respond  →  round is valid, average and proceed.
If < min_peers respond  →  fall back to local gradient, no round wasted.

A dead node NEVER blocks the swarm.
```

This mirrors how BitTorrent handles slow seeders: you download from whoever is available, not whoever is slowest.

### The Noise Frame Problem (and How We Solve It)

libp2p's Noise protocol encrypts traffic in frames capped at 65,535 bytes. A gradient payload for even a small MLP (~264 KB) exceeds this limit, causing `NoiseInvalidMessage` errors at the transport layer.

The gossip layer solves this with transparent chunking:

```
broadcast_gradient(264 KB payload)
  │
  ├── frame 0: [transfer_id=X | chunk=0/4 | 60 KB]
  ├── frame 1: [transfer_id=X | chunk=1/4 | 60 KB]
  ├── frame 2: [transfer_id=X | chunk=2/4 | 60 KB]
  └── frame 3: [transfer_id=X | chunk=3/4 | 24 KB]

receiver: accumulate chunks by transfer_id → reassemble → dispatch
```

The training loop sees only complete gradient messages. The chunking is invisible.

---

## Architecture

```
src/swarm_tune/
├── config/
│   └── settings.py          # NodeSettings — pydantic-settings, SWARM_ env vars
├── node/
│   ├── main.py              # SwarmNode orchestrator + CLI entrypoint
│   ├── p2p/
│   │   ├── discovery.py     # libp2p host, Ed25519 keys, mDNS, peer table
│   │   ├── gossip.py        # GossipProtocol — FloodSub broadcast + chunked framing
│   │   ├── heartbeat.py     # Liveness signals + stale peer eviction
│   │   └── peer_selector.py # PeerSelector protocol (AllPeers → ClusterPeers at 100 nodes)
│   ├── trainer/
│   │   ├── model.py         # ModelShard — PyTorch model + optimizer + checkpointing
│   │   ├── data.py          # DataShardLoader — load .pt shards, get_batch()
│   │   ├── gradient.py      # GradientExtractor — extract + validate param.grad tensors
│   │   ├── serializer.py    # GradientSerializer — SWRM wire format, weights_only=True
│   │   └── compressor.py    # Compressor protocol (Identity → TopK at 100 nodes)
│   └── aggregator/
│       ├── averaging.py     # GradientAverager — weighted FedAvg math
│       ├── timeout.py       # TimeoutAggregator — straggler tolerance
│       └── strategy.py      # AggregationStrategy protocol (Flat → Hierarchical at 100 nodes)
scripts/
└── generate_shards.py       # Synthetic training data generation
docker/
├── Dockerfile               # Multi-stage build (builder + lean runtime, non-root)
└── docker-compose.yml       # 6-node simulation: 5 honest + 1 adversarial
tests/
├── unit/                    # Fast, isolated — GradientExtractor, Serializer, DataShard, Aggregator
├── integration/             # Multi-component — convergence, P2P heartbeat, gradient sync
└── chaos/                   # Fault injection — node drop, deferred rounds, rejoin, adversarial
```

### The Three Extensibility Abstractions

Built now because they cost nothing and save rewrites later:

| Protocol | Default (Phases 1–4) | Scale-up (Phase 5+) | Swap requires |
|---|---|---|---|
| `Compressor` | `IdentityCompressor` (no-op) | `TopKCompressor` (1% → 100× bandwidth) | One config value |
| `PeerSelector` | `AllPeersSelector` | `ClusterPeerSelector` | One config value |
| `AggregationStrategy` | `FlatAggregation` | `HierarchicalAggregation` | One config value |

Zero changes to the training loop. Only config and the new implementation file.

---

## Development Setup

**Prerequisites:** Python 3.12, `brew install gmp` (macOS) or `libgmp-dev` (Linux)

```bash
# Full developer bootstrap (venv + deps + pre-commit hooks)
make bootstrap

# Code quality (lint + format check + mypy --strict)
make check

# Run unit and integration tests
make test

# Run chaos / fault-injection tests (slow, real timeouts)
make test-chaos

# Generate synthetic data shards for simulation
make shards

# Start the 6-node Docker swarm (auto-generates shards if missing)
make sim-up

# Tail logs from all nodes
make sim-logs

# Kill a node mid-training (chaos)
make sim-kill-node NODE=node_2

# Full cleanup
make clean-all
```

### Running Tests

```bash
# Fast (unit only)
python -m pytest -m unit --no-cov -v

# All (unit + integration + chaos)
python -m pytest -m "unit or integration or chaos" --no-cov -v

# With coverage report
make coverage
```

---

## The Real-World Vision

Imagine a future where:

- A small AI lab publishes a training job to the Swarm-Tune network
- 50 hobbyists with gaming PCs opt in
- Each runs the Swarm-Tune client, which loads their shard of the model and data
- Training runs over 48 hours, across home internet connections, surviving restarts and disconnects
- The lab pays participants in compute credits or tokens
- The resulting model is open-sourced

This is **democratized AI training**. The infrastructure is the community.

---

## Architectural Rules

These constraints are non-negotiable (see `CLAUDE.md` for full detail):

1. **No Central Server.** Nodes coordinate exclusively through libp2p gossip. No Flask. No FastAPI. No master node.
2. **No Standard DDP.** PyTorch's `DistributedDataParallel` assumes a data center. Gradients are manually extracted from `param.grad`, serialized, transmitted, and averaged.
3. **Straggler Tolerance.** A slow or dead node never blocks the swarm. Timeout-based partial aggregation, always.
4. **Code Against Abstractions.** `Compressor`, `PeerSelector`, and `AggregationStrategy` are protocols today so they can be swapped at scale without touching the training loop.
5. **Security is Architectural.** `weights_only=True` on every deserialization of peer data. NaN/Inf/norm bounds enforced before any gradient enters the aggregator. No trusted peer IDs from the wire.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Networking | `libp2p` 0.6.0 | Powers IPFS. Ed25519 peer IDs. Kademlia DHT. No central server. |
| Deep Learning | `PyTorch` ≥2.3 | Full access to `param.grad` tensors. MPS support on Apple Silicon. |
| Async runtime | `anyio` + `trio` | libp2p requires trio internally; anyio keeps the rest backend-agnostic. |
| Config | `pydantic-settings` | Fail loudly on bad config at startup, not at runtime. |
| Logging | `structlog` | JSON in Docker, human-readable console locally. |
| Orchestration | `Docker` + `docker-compose` | Simulate 6 independent nodes on one machine. |
| Language | Python 3.12 | `mypy --strict` throughout. No untyped code. |
| Linting | `ruff` | Replaces black, isort, flake8, pylint in one tool. |

---

## Contributing

Read `CLAUDE.md` before contributing. It is the source of truth for every design decision.

The architecture is intentionally kept simple and readable — this is as much an educational reference for decentralized ML as a working system.

---

## License

MIT. Build on it. Break it. Make it better.

---

*"Any sufficiently advanced distributed system is indistinguishable from a swarm."*
