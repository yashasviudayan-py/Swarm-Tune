# Swarm-Tune

> **The BitTorrent of AI Training.**
> 20 gaming PCs. 480 GB of pooled VRAM. One model that nobody could train alone.

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
Round N begins. Timeout window: T seconds.

Nodes that respond within T → their gradients are included.
Nodes that miss the window → skipped this round, catch up next.

If ≥ 50% of known peers respond → round is valid, average and proceed.
If < 50% respond → extend window or pause and wait.

A dead node NEVER blocks the swarm.
```

This mirrors how BitTorrent handles slow seeders: you download from whoever is available, not whoever is slowest.

---

## Architecture

```
swarm-tune/
├── node/
│   ├── p2p/              # libp2p peer discovery, gossip, heartbeat
│   ├── trainer/          # PyTorch model shard + local gradient extraction
│   ├── aggregator/       # Gradient averaging with straggler tolerance
│   └── main.py           # Node entrypoint
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml  # Simulates N nodes locally
├── data/
│   └── shards/           # Per-node data partitions
├── tests/
│   └── chaos/            # Kill nodes mid-training, verify convergence
└── CLAUDE.md             # Architectural rules (read before contributing)
```

### Components

#### P2P Layer (`node/p2p/`)

Built on `libp2p` (Python implementation). Handles:
- **Peer discovery**: mDNS for local simulation, DHT for internet deployment
- **Gossip protocol**: Gradient broadcast and collection across the swarm
- **Heartbeat**: Every node publishes a liveness signal; dead nodes are evicted from the active peer set

No central server. No registry. Every node is equal.

#### Trainer (`node/trainer/`)

Pure PyTorch. Handles:
- Loading a model shard into local memory (CPU or MPS on Apple Silicon)
- Running forward and backward passes on local data
- Extracting raw gradients from `param.grad` tensors
- Serializing gradients to bytes for network transmission

#### Aggregator (`node/aggregator/`)

The math layer. Handles:
- Collecting serialized gradients from peers
- Deserializing and stacking tensors
- Computing weighted averages (weighted by dataset size per node for fairness)
- Applying averaged gradients to update local weights
- Enforcing the timeout window and partial aggregation logic

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

## Current Scope (Local Simulation)

The current implementation simulates the full swarm on a single machine using Docker:

- **4-5 containers** simulate independent nodes with separate data shards
- Each container runs the Swarm-Tune node software
- Nodes discover each other via libp2p within the Docker network
- Training runs for N rounds with gradient sync each round
- Chaos tests kill random containers mid-training to prove fault tolerance

**Hardware:** Apple Silicon M4 Pro, 24 GB Unified Memory. PyTorch uses `mps` backend where supported, CPU fallback inside Docker.

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | P2P Network Initialization — nodes discover each other, maintain live peer lists | Planned |
| **Phase 2** | Local Gradient Extraction — PyTorch forward/backward, serialize gradients | Planned |
| **Phase 3** | Gradient Synchronization — average weights over libp2p gossip, straggler tolerance | Planned |
| **Phase 4** | Docker Simulation & Chaos Testing — kill nodes mid-training, verify convergence | Planned |
| **Phase 5** | Internet Deployment — real peers, real latency, real-world testing | Future |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Networking | `libp2p` (Python) | Battle-tested P2P protocol. Powers IPFS. No central server required. |
| Deep Learning | `PyTorch` | Full control over gradient tensors. Manual extraction possible. |
| Orchestration | `Docker` + `docker-compose` | Simulate N independent nodes on one machine. |
| Language | Python 3.12+ | Strict typing, modern async, broad ML ecosystem. |

---

## Architectural Rules

These constraints are non-negotiable (see `CLAUDE.md` for full detail):

1. **No Central Server.** Nodes coordinate exclusively through libp2p gossip. No Flask. No FastAPI. No master.
2. **No Standard DDP.** PyTorch's `DistributedDataParallel` assumes a data center. We do not live there. Gradients are manually extracted, serialized, transmitted, and averaged.
3. **Straggler Tolerance.** A slow or dead node never blocks the swarm. Timeout-based partial aggregation always.

---

## Contributing

This project is in early development. The architecture is intentionally kept simple and readable — this is as much an educational reference for decentralized ML as it is a working system.

Read `CLAUDE.md` before contributing. It is the source of truth for every design decision.

---

## License

MIT. Build on it. Break it. Make it better.

---

*"Any sufficiently advanced distributed system is indistinguishable from a swarm."*
