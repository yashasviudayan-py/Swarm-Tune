# Swarm-Tune Architecture Document

**Version:** 1.0.0
**Last Updated:** 2026-03-09
**Status:** Production-ready (Phases 1–8 complete)

---

## 1. System Overview

Swarm-Tune is a decentralized federated learning system that enables groups of consumer GPU owners to collaboratively fine-tune large language models without a central server or cloud infrastructure. It implements Federated Averaging (FedAvg) over a hostile peer-to-peer network using libp2p for transport and PyTorch for gradient computation.

**Core thesis:** 20 nodes × 24 GB VRAM = 480 GB pooled memory. A 70B fp16 model needs ~140 GB — no single node has it, but the swarm does.

**Scale target:** 100+ nodes over the public internet.

---

## 2. Component Breakdown (C4 — Container Level)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        SwarmNode (main.py)                          │
│   Top-level orchestrator. Wires subsystems, runs training loop.     │
├────────────────┬──────────────────────┬──────────────────────────────┤
│   P2P Layer    │   Trainer Layer      │   Aggregator Layer           │
│                │                      │                              │
│ ┌────────────┐ │ ┌──────────────────┐ │ ┌──────────────────────────┐ │
│ │ Discovery  │ │ │ ModelShard       │ │ │ TimeoutAggregator        │ │
│ │ (libp2p    │ │ │ (MLP / HF       │ │ │ (deadline-based partial  │ │
│ │  host,     │ │ │  CausalLM)      │ │ │  gradient collection)    │ │
│ │  mDNS,     │ │ ├──────────────────┤ │ ├──────────────────────────┤ │
│ │  bootstrap)│ │ │ GradientExtract  │ │ │ GradientAverager         │ │
│ ├────────────┤ │ │ (extract +       │ │ │ (FedAvg, subnet cap,     │ │
│ │ Gossip     │ │ │  validate)       │ │ │  streaming memory)       │ │
│ │ (FloodSub, │ │ ├──────────────────┤ │ ├──────────────────────────┤ │
│ │  chunking, │ │ │ Serializer       │ │ │ AggregationStrategy      │ │
│ │  reassembly│ │ │ (SWRM wire fmt)  │ │ │ (Flat / Hierarchical)    │ │
│ ├────────────┤ │ ├──────────────────┤ │ └──────────────────────────┘ │
│ │ Heartbeat  │ │ │ Compressor       │ │                              │
│ │ (liveness, │ │ │ (Identity/TopK)  │ │ ┌──────────────────────────┐ │
│ │  eviction) │ │ ├──────────────────┤ │ │ MetricsStore + HTTP      │ │
│ ├────────────┤ │ │ DataShardLoader  │ │ │ (per-node /metrics       │ │
│ │ PeerSelect │ │ │ / HFDataShard   │ │ │  sidecar on port+100)    │ │
│ │ + BanList  │ │ │                  │ │ └──────────────────────────┘ │
│ └────────────┘ │ └──────────────────┘ │                              │
└────────────────┴──────────────────────┴──────────────────────────────┘
         ▲                                          ▲
         │ libp2p FloodSub (gossip)                 │ Static HTML
         ▼                                          ▼
┌──────────────────┐                    ┌──────────────────────┐
│  Other Peers     │                    │  dashboard/index.html│
│  (N SwarmNodes)  │                    │  (browser polls      │
│                  │                    │   /metrics endpoints) │
└──────────────────┘                    └──────────────────────┘
```

### 2.1 P2P Layer (`node/p2p/`)

| Component | Responsibility |
|-----------|---------------|
| `PeerDiscovery` | Creates Ed25519 identity, binds TCP listener, manages peer table, resolves multiaddrs |
| `GossipProtocol` | FloodSub gradient broadcast with transparent chunking (≤60 KB frames) and reassembly |
| `Heartbeat` | Periodic liveness pings (5s interval), stale peer eviction (configurable, default 60s) |
| `PeerSelector` | Abstraction for broadcast target selection (`AllPeers` / `ClusterPeer`) |
| `BanList` | Temporary peer bans based on gradient rejection rate (Sybil resistance) |

### 2.2 Trainer Layer (`node/trainer/`)

| Component | Responsibility |
|-----------|---------------|
| `ModelShard` | Loads MLP or HF CausalLM, layer-level sharding, optimizer management, atomic checkpoints |
| `GradientExtractor` | Extracts `.grad` tensors post-backward, validates (NaN/Inf, RMS norm threshold) |
| `GradientSerializer` | SWRM wire protocol: `[4B magic][4B version][torch payload]`, `weights_only=True` on deserialize |
| `Compressor` | `IdentityCompressor` (no-op) or `TopKCompressor` (sparse encoding, ~50x reduction at k=1%) |
| `DataShardLoader` / `HFDataShardLoader` | Local .pt files or HuggingFace datasets with deterministic sharding |

### 2.3 Aggregator Layer (`node/aggregator/`)

| Component | Responsibility |
|-----------|---------------|
| `TimeoutAggregator` | Deadline-based collection: waits for `min_peers` or `timeout_secs`, whichever comes first |
| `GradientAverager` | Dataset-size-weighted FedAvg with intersection-based parameter matching and streaming memory |
| `AggregationStrategy` | `FlatAggregation` (all peers) or `HierarchicalAggregation` (cluster-based, stub) |

### 2.4 Support Components

| Component | Responsibility |
|-----------|---------------|
| `MetricsStore` + HTTP server | Per-node JSON metrics via raw TCP (anyio, trio-compatible) |
| `RunManifest` | Training campaign definitions (JSON), env file generation, `join.py` onboarding |
| `competition.py` | Perplexity comparison between team checkpoints |

---

## 3. Data Flow

### 3.1 Training Round (per node)

```
1. FORWARD/BACKWARD
   DataShard.get_batch() → ModelShard.compute_loss() → ModelShard.backward()
                                                              │
2. EXTRACT & COMPRESS                                         ▼
   GradientExtractor.extract(model) → Compressor.compress() → validate()
                                                              │
3. SELF-SUBMIT                                                ▼
   Compressor.decompress(compressed) → TimeoutAggregator.submit(local_grad)
                                                              │
4. BROADCAST                                                  ▼
   GradientSerializer.serialize() → GossipProtocol.broadcast_gradient()
        │                                                     │
        ▼                                                     │
   Chunking (≤60KB per FloodSub msg) → libp2p FloodSub       │
                                                              │
5. RECEIVE (from peers)                                       │
   FloodSub → reassemble chunks → deserialize → decompress   │
   → validate (NaN/Inf, RMS norm) → ban-list check           │
   → TimeoutAggregator.submit(peer_grad)                      │
                                                              │
6. AGGREGATE & APPLY                                          ▼
   TimeoutAggregator.wait() → GradientAverager.average()
   → ModelShard.apply_averaged_gradients() → optimizer.step()
```

### 3.2 Wire Protocol

```
Outer (FloodSub frame — per chunk):
  [8B transfer_id (uint64)] [4B chunk_idx (uint32)] [4B total_chunks (uint32)] [N bytes data]

Inner (reassembled GradientMessage):
  [4B sender_id_len (uint32)] [4B round_number (int32)] [8B dataset_size (int64)]
  [sender_id bytes (UTF-8)] [SWRM payload]

SWRM Payload:
  [4B "SWRM" magic] [4B version (uint32)] [torch.save dict]
```

---

## 4. Security Posture

### 4.1 Identity & Transport
- **Ed25519 key pairs** → cryptographic peer IDs; spoofing requires breaking the key
- **Noise handshake** verifies identity before any application data flows
- **Authenticated peer ID** (`msg.from_id`) used for ban lists, not wire-level `sender_id`
- **SSRF protection**: resolved IPs checked against link-local (169.254.x.x) before dialing

### 4.2 Deserialization
- `torch.load(..., weights_only=True)` on all peer-received payloads
- SWRM magic bytes validated before any torch deserialization
- Deserialized dict validated: all keys must be `str`, all values must be `torch.Tensor`
- Local data files use `weights_only=False` (trusted, not from network)

### 4.3 Gradient Validation
- NaN/Inf rejection on every received gradient
- Per-element RMS norm threshold (default 10.0) — model-size-agnostic
- Shape validation implicit via parameter intersection in FedAvg

### 4.4 Sybil Resistance
- Subnet contribution cap: /24 subnet peers treated as single contributor
- Rolling rejection rate tracking → temporary bans (default 600s after >50% rejection)
- Rate limiting: one gradient per peer per round (TimeoutAggregator dedup)

### 4.5 Checkpoint Safety
- System path guard (`/etc`, `/sys`, `/proc`, etc.) on `checkpoint_dir`
- `node_id` sanitized to `[\w\-]` only — prevents path traversal
- Atomic write (temp file → `os.replace`)
- `CancelScope(shield=True)` protects final checkpoint from SIGTERM

### 4.6 HTTP Metrics Server
- GET-only (405 for other methods)
- Security headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Cache-Control: no-store`
- `Access-Control-Allow-Origin: *` (intentional — dashboard polls from browser)
- Request size capped at 8192 bytes

---

## 5. Deployment Strategy

### 5.1 Local Development
```bash
make check          # ruff + mypy
make test           # 110 tests (unit + integration)
make sim-up         # 6-node Docker simulation
```

### 5.2 Docker Simulation (6 Nodes)
- `docker-compose.yml`: 5 honest nodes + 1 adversarial
- Node 0 = bootstrap (deterministic key seed → stable peer ID)
- Shared Docker bridge network for mDNS discovery
- Auto-generated synthetic shards

### 5.3 Internet Deployment
1. **Relay VPS**: `docker run --net host swarmtune/node` with `SWARM_RELAY_MODE=true`
2. **Participants**: `python scripts/join.py --run-id <manifest> --node-index N`
3. **CI/CD**: `.github/workflows/publish.yml` → PyPI + Docker Hub on version tag

### 5.4 Competition Mode
```bash
make competition \
  TEAM_A_CHECKPOINT=ckpts/alpha.pt \
  TEAM_B_CHECKPOINT=ckpts/beta.pt
```
Teams publish checkpoints to HuggingFace Hub. Anyone verifies with `make benchmark`.

---

## 6. Constraints & Trade-offs

| Constraint | Impact | Mitigation |
|-----------|--------|-----------|
| py-libp2p uses **trio** internally | Must use `anyio.run(backend="trio")` everywhere | No asyncio libraries (aiohttp removed, raw TCP metrics server) |
| FloodSub is O(N) per message | Scales poorly beyond ~30 nodes | GossipSub migration planned (requires go-libp2p or rust-libp2p) |
| Noise frame limit: 65,535 bytes | Gradients must be chunked | Transparent chunking with TTL-based stale transfer eviction |
| Internet latency (100ms+) | Cannot use standard DDP | Custom FedAvg with timeout-based partial aggregation |
| Consumer hardware (variable) | Stragglers are common | Hard deadline per round; dead nodes evicted, not waited for |
| Hostile network | Gradient poisoning, Sybil attacks | NaN/Inf rejection, RMS norm, subnet cap, ban list |

---

## 7. Audit Findings & Fixes

See `AUDIT.md` for the comprehensive security, logic, and performance audit
with root-cause fixes and regression tests.
