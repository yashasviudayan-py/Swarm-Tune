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

**The competition angle:** groups of friends can run competing swarms — same base model, same dataset, different participants — and compare final benchmark scores. No central authority needed. Anyone can verify results by running the benchmark locally on the published checkpoint.

---

## 2. Hardware & Environment Constraints

- **Development hardware:** Apple Silicon M4 Pro, 24 GB Unified Memory, 16-core GPU.
- **Compute backend:** PyTorch with `mps` (Metal Performance Shaders) where supported. CPU fallback is acceptable inside Docker simulation.
- **Target hardware (real deployment):** Any machine with a GPU ≥ 8 GB VRAM and a broadband internet connection. NVIDIA, AMD, Apple — all must work.
- **Cost:** $0 budget for development. No external cloud GPUs or APIs. 100% local simulation.
- **Public relay node:** A single $5/month VPS is the only infrastructure cost for internet deployment. It acts as a bootstrap + circuit-relay node only — it holds no model weights, no gradients, no training state.

---

## 3. The Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Networking | `libp2p` (Python) | Powers IPFS. Battle-tested P2P. No central server. Cryptographic peer IDs. Kademlia DHT scales to millions of nodes. |
| Deep Learning | `PyTorch` | Full access to `param.grad` tensors for manual extraction. |
| Models | HuggingFace `transformers` | GPT-2 → LLaMA. Standard checkpointing. `from_pretrained()` for easy shard loading. |
| Orchestration | `Docker` + `docker-compose` | Simulate 6 disparate nodes on one machine. |
| Language | Python 3.12+ strict typing | `mypy --strict` clean. No untyped code merged. |
| Config | `pydantic-settings` | Fail loudly on bad config at startup, not at runtime. |
| Logging | `structlog` | JSON logs in Docker, human-readable logs locally. |
| Dashboard | Per-node HTTP metrics endpoint + static HTML/JS | Zero central server. Each node serves its own metrics on a sidecar port. |
| Distribution | `pip install swarm-tune` + Docker Hub image | One-command install for participants. |

---

## 4. Strict Architectural Rules

These rules are non-negotiable. No exceptions. No "temporary" violations.

### Rule 1: No Central Server

There is no master HTTP server (Flask, FastAPI, or otherwise) orchestrating the swarm.
Nodes discover each other and coordinate exclusively through libp2p P2P gossip.
Every node is equal. The bootstrap peer is just the first known address, not a master.

**Dashboard exception:** each node may expose a local HTTP metrics endpoint (default port `node_port + 100`, e.g. `9100` for a node on `9000`). This serves only that node's own data. There is no aggregation backend. The dashboard HTML/JS page is a static file that the user opens locally and points at whatever node endpoints they choose.

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

### Rule 5: Real Models Must Be Shardable Without Code Changes

The `ModelShard` abstraction must support loading any HuggingFace model by name.
Switching from the toy MLP to GPT-2 to LLaMA must require only a config change
(`SWARM_MODEL_NAME`, `SWARM_MODEL_SHARD_INDEX`, `SWARM_MODEL_SHARD_TOTAL`).
No trainer logic should be model-specific.

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
- **Local gradients are also validated.** Own NaN/Inf from an exploding loss must not enter the FedAvg pool.

### Phase 4 — Chaos & Adversarial Testing ✅ COMPLETE

- Adversarial node broadcasting NaN gradients → rejected, swarm continues.
- Out-of-bounds norm gradients → rejected, swarm continues.
- Malformed bytes → rejected at deserialization, swarm continues.
- Wrong SWRM magic → rejected before `torch.load`, swarm continues.
- Stale-round gradients → rejected by round_number check, not entered into aggregator.
- Chunk index bounds → validated; `chunk_idx >= total_chunks` raises ValueError before storage.

### Phase 5 — Sybil & Eclipse Resistance ✅ COMPLETE

Implemented as part of Phase 5 internet deployment:
- **Sybil resistance:** subnet contribution cap (`/N` prefix, configurable) applied in `GradientAverager` before FedAvg weight computation. Default /24.
- **Reputation:** `BanList` in `peer_selector.py` tracks per-peer rejection rate across rounds; peers exceeding `rejection_ban_threshold` (default 50%) are temporarily banned for `rejection_ban_duration_secs` (default 600s).
- **Rate limiting:** one gradient submission per peer per round enforced in `TimeoutAggregator`.
- **Eclipse resistance:** maintain connections to peers from diverse IP ranges (future hardening).

---

## 6. Execution Phases (The Roadmap)

### Phase 1: P2P Network Initialization ✅ COMPLETE
- Bootstrap a libp2p swarm where nodes discover each other via mDNS (local) or known bootstrap peer.
- Implement gossip-based heartbeat: every node publishes liveness, dead nodes are evicted.
- **Security gate:** cryptographic peer IDs verified on connection.
- **Deliverable:** 5 Docker containers that find each other and maintain a live, auto-updating peer list.

### Phase 2: Local Gradient Extraction ✅ COMPLETE
- Instantiate a model shard in PyTorch (start with a small MLP, graduate to a real transformer).
- Run forward + backward pass on the local data shard.
- Extract `param.grad` tensors, compress with `IdentityCompressor`, serialize with `SWRM` protocol.
- **Security gate:** `weights_only=True` enforced in serializer. No pickle.
- **Deliverable:** Each node independently computes, serializes, and logs its gradient payload.

### Phase 3: Gradient Synchronization ✅ COMPLETE
- Broadcast serialized gradients via libp2p GossipSub to cluster peers.
- Collect peer gradients with `TimeoutAggregator` (straggler tolerance active).
- Validate each received gradient before averaging.
- Compute FedAvg weighted average, apply to local model, step optimizer.
- **Security gate:** NaN/Inf/norm checks reject malicious gradients before aggregation.
- **Deliverable:** All live nodes converge to the same weights after each round.

### Phase 4: Docker Simulation & Chaos Testing ✅ COMPLETE
- Train for 20 rounds across 6 containers (5 honest + 1 adversarial), verify loss convergence.
- Chaos: kill random containers mid-training, verify swarm continues.
- Adversarial chaos: inject a poisoned-gradient node, verify the swarm rejects it and continues.
- Post-audit: patched 4 production bugs (stale-round contamination, local gradient validation,
  wrong fallback gradient, chunk bounds crash).
- **Deliverable:** Logged training run with loss curves, node join/leave events, adversarial
  rejection events, and proof of convergence. `scripts/parse_logs.py` for evidence capture.

### Phase 5: Internet Deployment & NAT Traversal ✅ COMPLETE
- **Step 1 — Real model loading:** `ModelShard` now calls `AutoModelForCausalLM.from_pretrained()`
  for any HuggingFace model. Layer sharding by `shard_index % shard_total`. Loss switched from
  MSE → cross-entropy via `outputs.loss`. MLP path preserved for local simulation / unit tests.
- **Step 2 — Real data pipeline:** `HFDataShardLoader` streams HuggingFace datasets, tokenizes
  with `AutoTokenizer`, shards deterministically by `data_shard_index / data_shard_total`.
  `DataShardLoader` retained for backward-compatible `.pt` simulation mode.
- **Step 3 — NAT traversal:** `relay_addrs`, `enable_relay`, `enable_hole_punching` added to
  `NodeSettings`. `PeerDiscovery.connect_bootstrap()` dials relay peers first when enabled.
- **Step 4 — Easy install:** `.github/workflows/publish.yml` (PyPI + Docker Hub on tag),
  `JOIN.md` (5-minute onboarding), `my.env.template` (all vars documented). Node startup
  prints a human-readable summary line.
- **Step 5 — `/metrics` sidecar + dashboard:** `MetricsStore` dataclass + `run_metrics_server()`
  (aiohttp, port+100). `dashboard/index.html` (vanilla JS, no build step) polls node endpoints,
  renders live loss curves, peer count, rejection/deferred counters.
- **Step 6 — Checkpoint save + benchmark:** auto-save every N rounds + final on shutdown.
  `scripts/benchmark.py` computes deterministic perplexity over WikiText-103 test split.
  `make benchmark CHECKPOINT=<path>` target added.
- **Step 7 — Sybil resistance + rate limiting:** `_apply_subnet_cap()` in `GradientAverager`,
  `BanList` in `peer_selector.py`, one-gradient-per-peer-per-round in `TimeoutAggregator`.
- Post-audit: patched 5 production bugs (data/model shard index confusion, metrics server
  OSError crash, port+100 overflow, subnet cap `int()` truncation, IPv6 subnet key).
- **Deliverable:** All 7 steps implemented and production-audited. `dashboard/index.html` live,
  `scripts/benchmark.py` deterministic, Sybil resistance enabled. Ready for internet deployment
  once a public relay VPS is provisioned.

### Phase 6: Dashboard 🔲 NEXT
- Each `SwarmNode` exposes a `/metrics` JSON endpoint on a sidecar port (`node_port + 100`).
  Uses a lightweight HTTP server (no FastAPI — use `http.server` or `aiohttp`) running
  alongside the libp2p listener. Does not share any state with the P2P stack.
- Metrics include: current round, loss history, peer count, gradient rejection count,
  deferred rounds, bytes sent/received, node uptime.
- A single static `dashboard/index.html` file (vanilla JS, no build step, no npm) polls
  the metrics endpoints of whatever nodes the user configures.
- No backend. No central server. The dashboard is a passive observer only.
- **Deliverable:** Open `dashboard/index.html` in a browser → live loss curves, peer network
  graph, adversarial rejection counter, round progress for all configured nodes.

### Phase 7: Data & Model Distribution 🔲
- **Model distribution:** participants download base model weights from HuggingFace Hub
  (`from_pretrained()`). The swarm does not transfer model weights — only gradients.
- **Data distribution:** define a canonical dataset split (deterministic by `shard_index`
  and `num_shards`). Participants download the full dataset and the node slices its own
  shard automatically. No peer-to-peer data transfer.
- **Checkpoint distribution:** after training, the organiser publishes layer-shard checkpoints.
  Any node can reconstruct the full model by collecting all shards (like BitTorrent).
- **Deliverable:** `make join RUN_ID=gpt2-wikitrain-001` downloads the right data shard,
  loads the right model shard, and joins the swarm automatically.

### Phase 8: Competition & Leaderboard 🔲
- **Competition format:** two or more swarms train the same base model on the same dataset
  for a fixed number of rounds or wall-clock time. Winner is determined by perplexity on
  a shared held-out test set.
- **No central leaderboard server.** Each team publishes their final checkpoint (e.g. to
  HuggingFace Hub or IPFS). Anyone can download it, run `make benchmark`, and verify the
  score. Results are posted publicly (GitHub Discussions, HuggingFace model card, etc.).
- **Verification:** the benchmark script is deterministic and reproducible. Given a
  checkpoint and the test set, it always produces the same perplexity score.
- **Deliverable:** `make benchmark CHECKPOINT=path/to/checkpoint` prints a verifiable
  perplexity score that any third party can reproduce.

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
├── TopKCompressor        — keep top K% of gradient elements (Phase 5+)
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

### Phases 1–4 ✅ (Local Simulation — COMPLETE)

A single `make sim-up` command:
1. Spins up 6 Docker containers (5 honest + 1 adversarial)
2. Nodes discover each other with no manual intervention
3. Training runs for N rounds with gradient sync each round
4. Loss decreases monotonically (or near-monotonically)
5. Killing containers does not stop training
6. Bringing them back does not corrupt the model
7. A poisoned-gradient container is rejected and logged
8. All of the above is captured in structured JSON logs parseable by `scripts/parse_logs.py`

### Phase 5 ✅ (Internet Deployment — COMPLETE)

All infrastructure implemented and audited:
1. `AutoModelForCausalLM` loads any HuggingFace model; layer sharding by config only
2. `HFDataShardLoader` streams WikiText-103, tokenizes, shards deterministically
3. Circuit-relay NAT traversal wired into `PeerDiscovery`; `enable_relay` config flag
4. `JOIN.md` + `my.env.template` + Docker Hub publish workflow ready
5. `/metrics` sidecar (aiohttp) + `dashboard/index.html` (vanilla JS) live
6. Auto-checkpoint + `scripts/benchmark.py` perplexity evaluation
7. Sybil resistance (subnet cap + ban list + rate limit) enabled
- Pending: provision public relay VPS + push `swarmtune/node` tag to trigger publish workflow

### Phase 6 🔲 NEXT (Dashboard)

`open dashboard/index.html` in any browser:
1. Live loss curves for all configured nodes
2. Peer network graph (who is connected to whom)
3. Round counter, rejection events, straggler events
4. Works with zero backend — polls node `/metrics` endpoints directly

### Phase 7 🔲 (Distribution)

`make join RUN_ID=gpt2-run-001`:
1. Downloads the correct data shard automatically
2. Loads the correct model shard from HuggingFace
3. Joins the swarm and starts contributing gradients

### Phase 8 🔲 (Competition)

`make benchmark CHECKPOINT=./checkpoints/final`:
1. Runs perplexity evaluation on the shared test set
2. Prints a reproducible score any third party can verify
3. Teams publish scores publicly; no central leaderboard needed

---

## 9. The Three Real-World Blockers (Phase 5 Priority)

These are the specific technical problems that must be solved before any non-developer
can participate in a real training run.

### Blocker 1: NAT Traversal

**Problem:** most home users are behind NAT. Their machine has no public IP. libp2p's
default TCP transport fails when both peers are behind NAT simultaneously.

**Solution (in order of implementation):**
1. Deploy a public circuit-relay node (libp2p `relay` protocol). Cost: one $5/month VPS.
   Nodes behind NAT connect to the relay; the relay forwards traffic between them.
2. Enable libp2p AutoRelay: nodes automatically discover and use relay nodes from the DHT.
3. Enable libp2p hole-punching (`dcutr` protocol): attempt direct connection first,
   fall back to relay only if hole-punching fails. Reduces relay load at scale.

**Config additions needed in `NodeSettings`:**
- `relay_addrs: list[str]` — multiaddresses of known relay nodes
- `enable_relay: bool` — default `True` for internet deployment
- `enable_hole_punching: bool` — default `True`

**Files to modify:** `node/p2p/discovery.py` (add relay dialing), `config/settings.py`.

### Blocker 2: Real Models + Loss Function

**Problem:** the current `ModelShard` wraps a hand-written MLP. It cannot load GPT-2,
LLaMA, or any HuggingFace model without code changes. Additionally, `main.py` computes
MSE loss (`torch.nn.functional.mse_loss`) — this is wrong for language model training,
which requires cross-entropy over token logits.

**Solution:**
- `ModelShard` must call `AutoModelForCausalLM.from_pretrained(model_name)` and load
  only the layers assigned to this shard (`shard_index` out of `shard_total`).
- Layer assignment is deterministic: `layers[i]` belongs to shard `i % shard_total`.
- The optimizer must only cover parameters in the assigned layers.
- Gradient extraction already works on any `nn.Module` — no changes needed there.
- `SwarmNode._compute_loss()` must be replaced: call `model(input_ids, labels=targets)`
  and return `outputs.loss` (the built-in cross-entropy from HuggingFace models).

**Config already supports this:**
- `SWARM_MODEL_NAME` (default `"gpt2"`) — any HuggingFace model ID
- `SWARM_MODEL_SHARD_INDEX` — this node's shard
- `SWARM_MODEL_SHARD_TOTAL` — total shards in the swarm

**Files to modify:** `node/trainer/model.py`, `node/node/main.py` (`_compute_loss`).

**Graduation path:** MLP (done) → GPT-2 small (117M, Phase 5) → GPT-2 medium (345M) →
LLaMA 3 8B (Phase 6+).

### Blocker 3: Easy Install

**Problem:** setup currently requires Python 3.12, `libgmp` system library, Docker,
and manual config. This excludes non-developers.

**Solution:**
- Publish `swarm-tune` package to PyPI (`pip install swarm-tune`).
- Publish `swarmtune/node` image to Docker Hub with all dependencies pre-installed.
- Provide a one-page `JOIN.md` that takes a participant from zero to running in under
  5 minutes: download Docker → `docker run swarmtune/node --env-file my.env`.
- Provide a `my.env.template` with every required variable documented.
- The node must print a human-readable startup summary (not just JSON logs) so
  participants can confirm they are connected without parsing logs.

**Files to add:** `.github/workflows/publish.yml` (PyPI + Docker Hub on tag),
`JOIN.md`, `my.env.template`.

### Blocker 4: Real Training Data Pipeline

**Problem:** `DataShardLoader` currently loads hand-crafted `.pt` files generated by
`scripts/generate_shards.py` (synthetic random tensors). No real text dataset is wired
in. This means the model trains on noise and cannot produce a meaningful checkpoint.
Data distribution is listed as Phase 7 but it is actually a Phase 5 blocker — you cannot
do real internet training without real data.

**Solution:**
- Replace `DataShardLoader` with a HuggingFace `datasets`-backed loader.
- Use a public dataset (WikiText-103 or OpenWebText).
- Sharding is deterministic: `dataset.shard(num_shards=shard_total, index=shard_index)`.
- Tokenize with `AutoTokenizer.from_pretrained(model_name)`, returning `input_ids` tensors.
- The node slices its own shard automatically — no peer-to-peer data transfer.

**Config additions needed in `NodeSettings`:**
- `dataset_name: str` — default `"wikitext"`, any HuggingFace dataset ID
- `dataset_config: str` — default `"wikitext-103-raw-v1"` (the specific subset)
- `max_seq_len: int` — default `512`, tokenized sequence length per sample

**Files to modify:** `node/trainer/data.py`. Training loop unchanged — `get_batch()`
still returns `(inputs, targets)`, but now as tokenized `input_ids` tensors.

### Blocker 5: Sybil Resistance (Required Before Public Deployment)

**Problem:** currently any number of nodes from the same IP or operator can join the
swarm and dominate FedAvg. A single attacker running 19 of 20 nodes controls the result.
This is a critical security gap before opening the swarm to strangers on the internet.

**Solution (minimum viable):**
- Cap the fraction of the aggregation any single IP subnet (/24) can contribute.
  If 5 nodes share the same /24 subnet, their combined FedAvg weight is capped at 1×
  (as if they were one node), regardless of dataset size.
- Track per-peer gradient rejection rate across rounds. Peers that exceed a rejection
  threshold (e.g. >50% of rounds rejected) are temporarily banned (e.g. 10 minutes).
- Rate limit: reject a second gradient submission from the same peer in the same round.

**Files to modify:** `node/aggregator/averaging.py` (subnet cap + rate limit),
`node/p2p/peer_selector.py` (ban list), `config/settings.py` (new thresholds).

**This is the Phase 5 security gate.** Do not open to public internet participants
without it.

---

## 10. Implementation Order of Attack

This is the exact sequence to go from "impressive local demo" to "real people training
AI together over the internet." Each step unblocks the next.

### Step 1 — Real Model Loading (`node/trainer/model.py`) ✅ DONE
`AutoModelForCausalLM.from_pretrained(model_name)` with layer sharding by `shard_index % shard_total`.
`_compute_loss()` removed; `ModelShard.compute_loss()` returns `outputs.loss` (cross-entropy).
MLP path preserved for unit tests via `model_name="mlp"` constant.

### Step 2 — Real Data Pipeline (`node/trainer/data.py`) ✅ DONE
`HFDataShardLoader` streams any HuggingFace dataset, tokenizes with `AutoTokenizer`,
shards deterministically via `dataset.shard(num_shards, index)`. Added independent
`data_shard_index` / `data_shard_total` settings (decoupled from model shard assignment).
`DataShardLoader` retained for `.pt` simulation mode backward-compat.

### Step 3 — NAT Traversal (`node/p2p/discovery.py`, `config/settings.py`) ✅ DONE
`relay_addrs`, `enable_relay`, `enable_hole_punching` added to `NodeSettings`.
`PeerDiscovery._connect_to_relay_peers()` dials relay multiaddrs when `enable_relay=True`.
**Remaining:** provision the actual $5/month VPS and configure its multiaddr.

### Step 4 — Docker Hub Image + Easy Install ✅ DONE
`.github/workflows/publish.yml` (PyPI OIDC + Docker Hub multi-arch). `JOIN.md` 5-minute
guide. `my.env.template` fully documented. Node prints human-readable startup summary.
**Remaining:** push a `v*.*.*` tag to trigger the publish workflow.

### Step 5 — `/metrics` Sidecar + Dashboard ✅ DONE
`MetricsStore` + `run_metrics_server()` (aiohttp, `port+100`, `/metrics` + `/health`).
`dashboard/index.html` (vanilla JS, localStorage node list, live loss curves on `<canvas>`).

### Step 6 — Checkpoint Save + Benchmark Script ✅ DONE
`SwarmNode.run()` saves checkpoints every `checkpoint_every_n_rounds` and on clean shutdown.
`scripts/benchmark.py` computes `exp(cross-entropy)` over WikiText-103 test split.
`make benchmark CHECKPOINT=<path>` target in `Makefile`.

### Step 7 — Sybil Resistance + Rate Limiting ✅ DONE
`_apply_subnet_cap()` in `GradientAverager` (configurable /N prefix, `round()` not `int()`).
`BanList` in `peer_selector.py` (per-peer rejection rate, configurable threshold + duration).
One-gradient-per-peer-per-round enforced in `TimeoutAggregator`.
**Gate:** chaos test — 10 nodes from the same /24 subnet contribute no more than 1× weight in FedAvg.

### Step 8 — Run ID System + `make join` · ~2 days
Define a training campaign manifest (model name, dataset, num rounds, shard count) addressable
by a `RUN_ID` string. `make join RUN_ID=gpt2-run-001` downloads the right data shard config,
sets `SWARM_MODEL_NAME`, `SWARM_MODEL_SHARD_INDEX`, and starts the node automatically.
**Gate:** `make join RUN_ID=gpt2-wikitrain-001` on two machines → both nodes train on the
correct non-overlapping data shards with no manual config beyond the `.env` file.

### Step 9 — Competition Mode · ~2 days
Two swarms, same base model, same dataset, different participants, fixed number of rounds.
Winner determined by `make benchmark`. Each team publishes checkpoint to HuggingFace Hub or
IPFS. Anyone can verify by running `make benchmark CHECKPOINT=<path>`.
**Gate:** end-to-end competition run — two teams of 2+ nodes each, 50 rounds, benchmark scores
published and independently verified.

---

### Production Readiness Milestones

| After Step | What becomes possible |
|---|---|
| 1–2 | Real model training locally (GPT-2 on WikiText) |
| 3–4 | Two strangers on home internet can train together |
| 5–6 | Training is observable and results are verifiable |
| 7 | Safe to advertise to the public |
| 8–9 | The competitive training angle is live |
