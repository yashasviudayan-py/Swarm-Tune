# CLAUDE.md — Swarm-Tune

This file is the source of truth for every architectural and security decision.
Read it before writing a single line of code.

---

## 1. Project Vision: Swarm-Tune

**The BitTorrent of AI Training.**

The concrete goal: **20 people with regular gaming PCs (e.g. RTX 3090/4090, 24 GB VRAM each) pool their hardware over standard internet connections to fine-tune a massive model that would not fit on any single machine.**

- 20 nodes × 24 GB = **480 GB of pooled VRAM**
- A 70B parameter model in fp16 requires ~140 GB. No single node has that. The swarm does.
- No data center. No cloud bill. No NVIDIA monopoly. The swarm IS the cluster.

Each node holds a shard of the model and a shard of the data. Each trains locally, extracts gradients, and broadcasts them to peers. The swarm averages the gradients. Every node updates. Repeat.

This is Federated Averaging (FedAvg) over a hostile, unreliable, decentralized network — which is a fundamentally harder problem than data-center distributed training, and the one worth solving.

**Future target: 100+ nodes.** Every design decision must keep this in mind.

---

## 2. Hardware & Environment Constraints

- **Development hardware:** Apple Silicon M4 Pro, 24 GB Unified Memory, 16-core GPU.
- **Compute backend:** PyTorch with `mps` (Metal Performance Shaders) where supported. CPU fallback is acceptable inside Docker simulation.
- **Target hardware (real deployment):** Any machine with a GPU ≥ 8 GB VRAM and a broadband internet connection. NVIDIA, AMD, Apple — all must work.
- **Cost:** $0 budget for development. No external cloud GPUs or APIs. 100% local simulation.

---

## 3. The Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Networking | `libp2p` (Python) | Powers IPFS. Battle-tested P2P. No central server. Cryptographic peer IDs. Kademlia DHT scales to millions of nodes. |
| Deep Learning | `PyTorch` | Full access to `param.grad` tensors for manual extraction. |
| Orchestration | `Docker` + `docker-compose` | Simulate 5 disparate nodes on one machine. |
| Language | Python 3.12+ strict typing | `mypy --strict` clean. No untyped code merged. |
| Config | `pydantic-settings` | Fail loudly on bad config at startup, not at runtime. |
| Logging | `structlog` | JSON logs in Docker, human-readable logs locally. |

---

## 4. Strict Architectural Rules

These rules are non-negotiable. No exceptions. No "temporary" violations.

### Rule 1: No Central Server

There is no master HTTP server (Flask, FastAPI, or otherwise) orchestrating the swarm.
Nodes discover each other and coordinate exclusively through libp2p P2P gossip.
Every node is equal. The bootstrap peer is just the first known address, not a master.

### Rule 2: No Standard DDP

Do **not** use PyTorch's `DistributedDataParallel` with `nccl` or `gloo` backends.
Those assume microsecond-latency data-center interconnects. We live on the internet.

Instead, always:
1. Extract raw gradients from `param.grad` tensors after `loss.backward()`
2. Compress gradients through the `Compressor` abstraction (no-op by default, swappable)
3. Serialize them with the `SWRM` wire protocol (`GradientSerializer`)
4. Broadcast over libp2p GossipSub to the node's cluster peers
5. Collect peer gradients with `TimeoutAggregator`
6. Average with `GradientAverager` (FedAvg, weighted by dataset size)
7. Apply averaged gradients and step the optimizer

### Rule 3: Straggler Tolerance

A slow or dead node **never** blocks the swarm.

- Every round has a hard timeout (`aggregation_timeout_secs`, default 30s).
- If ≥ `min_peers_for_round` respond within the window → commit the round.
- If < `min_peers_for_round` respond → defer the round, do not crash.
- Dead nodes are evicted from the peer table after `EVICTION_THRESHOLD_SECS` (20s).
- Rejoining nodes are welcomed back on the next heartbeat, no special handling.

### Rule 4: Code Against Abstractions, Not Implementations

Every component that will need to change at scale must be behind an interface today,
even if only one implementation exists. This costs nothing now and saves rewrites later.

The three mandatory abstractions (see Section 8):
- `Compressor` — gradient compression strategy
- `PeerSelector` — which peers to broadcast to
- `AggregationStrategy` — flat vs. hierarchical averaging

---

## 5. Security Rules (Non-Negotiable, Per Phase)

Security in a P2P system is **architectural, not cosmetic**. It is built phase-by-phase alongside features, not bolted on at the end. A gradient from an unknown peer is untrusted data. Treat it accordingly.

### Phase 1 — Identity & Transport

- **Peer IDs are cryptographic.** libp2p derives peer IDs from Ed25519 public keys. Spoofing requires breaking the key. Do not implement any alternative peer identity scheme.
- **Never trust a peer ID string that arrives over the wire.** Verify it matches the actual libp2p connection's remote peer ID.

### Phase 2 — Serialization Safety

- **Never use `pickle.loads()` on data from a peer.** Ever. It allows arbitrary code execution.
- **Always use `torch.load(..., weights_only=True)`** on the receiving end. This is enforced in `GradientSerializer.deserialize()`. Do not remove this flag.
- The `SWRM` magic + version header must be validated before any deserialization attempt.

### Phase 3 — Gradient Validation (Poisoning Defence)

Every received gradient dict must be passed through `GradientExtractor.validate()` before it is submitted to the aggregator. This enforces:
- **No NaN or Inf values.** A poisoned gradient with NaN will corrupt the model silently.
- **L2 norm bounds.** Gradients with implausibly large norms are rejected. Threshold: `max_norm=1e4` by default.
- **Shape matching.** A gradient tensor with a wrong shape for a parameter is rejected.

### Phase 4 — Chaos & Adversarial Testing

Chaos tests must include at least one adversarial scenario:
- A node that sends gradients with NaN values → must be rejected, swarm continues.
- A node that sends gradients with out-of-bounds norms → must be rejected, swarm continues.
- A node that sends malformed bytes → must be rejected at deserialization, swarm continues.

Rejection means: log a warning, skip that peer's contribution this round. Do not crash. Do not ban the peer permanently (they may have had a transient bug).

### Future Phases — Sybil & Eclipse Resistance

Not in scope for local simulation but must be considered before internet deployment:
- **Sybil resistance:** limit the fraction of the aggregation any single IP subnet can contribute.
- **Eclipse resistance:** maintain connections to peers from diverse IP ranges.
- **Reputation:** track per-peer rejection rate; repeated poisoning triggers temporary bans.

---

## 6. Execution Phases (The Roadmap)

### Phase 1: P2P Network Initialization
- Bootstrap a libp2p swarm where nodes discover each other via mDNS (local) or known bootstrap peer.
- Implement gossip-based heartbeat: every node publishes liveness, dead nodes are evicted.
- **Security gate:** cryptographic peer IDs verified on connection.
- **Deliverable:** 5 Docker containers that find each other and maintain a live, auto-updating peer list.

### Phase 2: Local Gradient Extraction
- Instantiate a model shard in PyTorch (start with a small MLP, graduate to a real transformer).
- Run forward + backward pass on the local data shard.
- Extract `param.grad` tensors, compress with `IdentityCompressor`, serialize with `SWRM` protocol.
- **Security gate:** `weights_only=True` enforced in serializer. No pickle.
- **Deliverable:** Each node independently computes, serializes, and logs its gradient payload.

### Phase 3: Gradient Synchronization
- Broadcast serialized gradients via libp2p GossipSub to cluster peers.
- Collect peer gradients with `TimeoutAggregator` (straggler tolerance active).
- Validate each received gradient before averaging.
- Compute FedAvg weighted average, apply to local model, step optimizer.
- **Security gate:** NaN/Inf/norm checks reject malicious gradients before aggregation.
- **Deliverable:** All live nodes converge to the same weights after each round.

### Phase 4: Docker Simulation & Chaos Testing
- Train for N rounds across 5 containers, verify loss convergence.
- Chaos: kill random containers mid-training, verify swarm continues.
- Adversarial chaos: inject a poisoned-gradient node, verify the swarm rejects it and continues.
- **Deliverable:** Logged training run with loss curves, node join/leave events, adversarial rejection events, and proof of convergence.

---

## 7. Scaling Architecture (20 → 100+ Nodes)

This section documents the three bottlenecks that emerge between 20 and 100 nodes,
the abstractions built now to handle them, and what the implementations look like at each scale.

### The Three Bottlenecks

#### Bottleneck 1: Bandwidth (Gradient Size)

| Nodes | Model | Gradient size | Per-round data (no compression) |
|---|---|---|---|
| 20 | 70B fp16 | ~140 GB | 2.8 TB |
| 100 | 70B fp16 | ~140 GB | **14 TB** |
| 100 | 70B fp16 | ~1.4 GB (Top-K 1%) | **140 GB** |

At 100 nodes without compression, the bandwidth requirement is physically impossible on home internet.
**Solution: `Compressor` abstraction.** Plug in Top-K sparsification or 1-bit quantization without touching the gossip or aggregation layers.

#### Bottleneck 2: Aggregation Topology (Flat vs. Hierarchical)

Flat aggregation (what we build in Phase 3):
```
All 20 nodes → single aggregation round → averaged gradient
```

This collapses at 100 nodes. One node waiting for 99 gradient responses will always hit the timeout.

Hierarchical aggregation (what we add in a future phase):
```
100 nodes → 10 clusters of 10
Each cluster aggregates locally → 10 cluster gradients
Cluster leaders aggregate globally → 1 final gradient
```

This reduces round latency from O(N) to O(√N) and bandwidth per node by ~10×.

**Solution: `AggregationStrategy` abstraction.** `FlatAggregation` ships in Phase 3. `HierarchicalAggregation` plugs in later without changing the training loop.

#### Bottleneck 3: Peer Connections (Full Mesh vs. DHT Routing)

At 20 nodes: every node can maintain connections to all 19 peers. Fine.

At 100 nodes: 99 simultaneous TCP connections per node is impractical on home internet (port limits, NAT, memory).

**Solution: `PeerSelector` abstraction.** Today it returns all known peers. At 100 nodes it returns only the node's cluster peers (10-20 connections). libp2p's Kademlia DHT already routes messages to non-connected peers — we don't need to maintain a connection to everyone.

---

### Abstractions Built Now (Even Though Only One Implementation Exists)

These three abstractions cost nothing to add now. Removing them later would require rewriting the training loop.

#### 1. `Compressor` Protocol (`node/trainer/compressor.py`)

```
Compressor (Protocol)
├── IdentityCompressor    — no-op, ships now, used in Phases 1-4
├── TopKCompressor        — future: keep top K% of gradient elements
└── QuantizedCompressor   — future: 1-bit or 8-bit quantization
```

The `GradientSerializer` wraps the compressor. Swapping compression strategy requires changing one config value.

#### 2. `PeerSelector` Protocol (`node/p2p/peer_selector.py`)

```
PeerSelector (Protocol)
├── AllPeersSelector      — returns all live peers, ships now
└── ClusterPeerSelector   — future: returns only cluster members
```

The `GossipProtocol` asks `PeerSelector` who to broadcast to. At 20 nodes it's everyone. At 100 nodes it's your cluster.

#### 3. `AggregationStrategy` Protocol (`node/aggregator/strategy.py`)

```
AggregationStrategy (Protocol)
├── FlatAggregation       — single-level FedAvg, ships now
└── HierarchicalAggregation — future: two-level cluster averaging
```

The `SwarmNode` training loop calls `strategy.aggregate(contributions)`. The strategy is injected at startup from config.

---

### Configuration for Scale

`NodeSettings` already has `cluster_id` and `cluster_size` fields (both default to `0` and `1`).
At 20 nodes all nodes are in cluster 0. At 100 nodes the operator sets cluster assignments.
The code never needs to change — only config.

---

## 8. What "Done" Looks Like

The project is complete when a single `make sim-up` command:
1. Spins up 5 Docker containers
2. Nodes discover each other with no manual intervention
3. Training runs for N rounds with gradient sync each round
4. Loss decreases monotonically (or near-monotonically)
5. Killing 2 containers does not stop training
6. Bringing them back does not corrupt the model
7. A poisoned-gradient container is rejected and logged
8. All of the above is captured in structured JSON logs

The architecture is "done" when replacing `IdentityCompressor` with `TopKCompressor`,
`AllPeersSelector` with `ClusterPeerSelector`, and `FlatAggregation` with `HierarchicalAggregation`
requires **zero changes to the training loop** — only config and the new implementation files.
