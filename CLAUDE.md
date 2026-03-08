# CLAUDE.md — Swarm-Tune

Source of truth for architecture, security, and implementation decisions.
Read this before writing any code.

---

## 1. What This Project Is

**The BitTorrent of AI training.**

20 people with gaming PCs (RTX 3090/4090, 24 GB VRAM) pool hardware over the internet to fine-tune a model that doesn't fit on any single machine.

- 20 nodes × 24 GB = 480 GB pooled VRAM
- A 70B fp16 model needs ~140 GB — no single node has it, the swarm does
- No data center, no cloud bill

Each node holds a shard of the model and a shard of the data. Trains locally, extracts gradients, broadcasts to peers, swarm averages. Repeat. This is **FedAvg over a hostile decentralized network**.

**Competition angle:** groups run competing swarms (same base model, same dataset, different participants), compare perplexity scores. No central authority. Anyone can verify by running `make benchmark` on a published checkpoint.

**Scale target: 100+ nodes.** Every design decision must account for this.

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Networking | `libp2p` (Python) — Ed25519 peer IDs, FloodSub, mDNS, Kademlia DHT |
| Deep Learning | `PyTorch` — direct access to `param.grad` tensors |
| Models | HuggingFace `transformers` — `AutoModelForCausalLM.from_pretrained()` |
| Data | HuggingFace `datasets` — deterministic sharding, `AutoTokenizer` |
| Orchestration | Docker + docker-compose — 6-node local simulation |
| Language | Python 3.12+, `mypy --strict` clean |
| Config | `pydantic-settings` — fail loudly at startup, not at runtime |
| Logging | `structlog` — JSON in Docker, console locally |
| Dashboard | Static `dashboard/index.html` + per-node aiohttp metrics sidecar |
| Distribution | `pip install swarm-tune` + Docker Hub image |

---

## 3. Non-Negotiable Architectural Rules

### No Central Server
No Flask/FastAPI orchestrating the swarm. Nodes coordinate exclusively via libp2p gossip. Every node is equal. The bootstrap peer is just the first known address.

**Exception:** each node exposes a local metrics endpoint at `port + 100` (e.g. `9100` for a node on `9000`). This serves only that node's own data. The dashboard is a static HTML file the user opens locally.

### No Standard DDP
Do NOT use `DistributedDataParallel` with nccl/gloo. Those assume microsecond data-center latency. We live on the internet.

Instead:
1. Extract raw gradients from `param.grad` after `loss.backward()`
2. Compress via the `Compressor` abstraction (no-op by default)
3. Serialize with `SWRM` wire protocol (`GradientSerializer`)
4. Broadcast over libp2p FloodSub to cluster peers
5. Collect with `TimeoutAggregator`
6. Average with `GradientAverager` (FedAvg, weighted by dataset size)
7. Apply averaged gradients, step optimizer

### Straggler Tolerance
A slow or dead node **never** blocks the swarm.
- Hard timeout per round: `aggregation_timeout_secs` (default 30s)
- If ≥ `min_peers_for_round` respond → commit
- If fewer → defer, do not crash
- Dead nodes evicted after `heartbeat_eviction_secs` (default 60s)
- Rejoining nodes welcomed back on next heartbeat, no special handling

### Code Against Abstractions
Three mandatory abstractions (one implementation today, swappable at scale):
- `Compressor` — `IdentityCompressor` (now), `TopKCompressor` (active), `QuantizedCompressor` (future)
- `PeerSelector` — `AllPeersSelector` (now), `ClusterPeerSelector` (future)
- `AggregationStrategy` — `FlatAggregation` (now), `HierarchicalAggregation` (future)

### Real Models Are Config-Only
Switching from MLP → GPT-2 → LLaMA requires only `SWARM_MODEL_NAME` config change. No trainer logic is model-specific.

---

## 4. Security Rules

### Identity & Transport (Phase 1)
- Peer IDs are Ed25519 public key hashes — spoofing requires breaking the key
- Never trust a peer ID string from the wire; libp2p verifies during Noise handshake

### Serialization (Phase 2)
- **Never `pickle.loads()` on peer data** — arbitrary code execution
- Always `torch.load(..., weights_only=True)` in `GradientSerializer.deserialize()`
- Validate `SWRM` magic bytes before any deserialization

### Gradient Validation (Phase 3)
Every received gradient must pass `GradientExtractor.validate()`:
- No NaN or Inf values
- Per-element RMS norm ≤ `gradient_max_norm_rms` (default 10.0; RMS = `norm / sqrt(numel)`)
- Shape must match local model parameter shapes
- Local gradients are also validated before entering FedAvg

### Sybil Resistance (Phase 5)
- Subnet contribution cap: peers sharing a /N subnet (default /24) are treated as one contributor
- `BanList`: peers exceeding `rejection_ban_threshold` (default 50%) get temp-banned for `rejection_ban_duration_secs` (default 600s)
- Rate limiting: one gradient submission per peer per round in `TimeoutAggregator`

### Checkpoint Safety (Audit)
- `checkpoint_dir` validator rejects writes to system paths (`/etc`, `/sys`, `/proc`, etc.)
- `node_id` is sanitized to `[\w\-]` only — prevents path traversal in checkpoint filenames
- Final checkpoint wrapped in `anyio.CancelScope(shield=True)` — survives SIGTERM without corruption

---

## 5. Implementation Details That Bit Us

**libp2p internals:**
- py-libp2p 0.6.0 uses **trio** internally — always `anyio.run(backend="trio")`
- `Pubsub` is a `Service` — use `background_trio_service(pubsub)`, NOT `tg.start_soon(pubsub.run)`
- Bootstrap must connect AFTER pubsub starts — zero-capacity trio channel deadlocks otherwise
- `pubsub.subscribe()` and `pubsub.publish()` are both async
- `FloodSub([...])` expects `Sequence[TProtocol]` — cast: `TProtocol(FLOODSUB_PROTOCOL_ID)`

**Chunking:**
- Noise protocol hard limit: 65,535 bytes per frame
- MLP gradient (~264 KB) exceeds this — transparent chunking required
- Chunk header: `struct.Struct(">Q I I")` — uint64 transfer_id, uint32 chunk_idx, uint32 total_chunks
- Stale transfer eviction: `_TRANSFER_TTL_SECS = 60.0`, timer-driven `_eviction_loop()` runs every 30s

**FedAvg consistency:**
- Local gradient submitted as `decompress(compress(raw_grad))` to match peer gradient representation

**Bootstrap dial timeout:**
- `host.connect()` has no built-in timeout — wrap with `anyio.move_on_after(10.0)`

**Docker non-root HuggingFace cache:**
- Container runs as UID 1001; `~/.cache` → `/root/.cache` is inaccessible
- Fix: `ENV HF_HOME=/app/.cache/huggingface` in Dockerfile

**Loss history memory:**
- `deque(maxlen=1000)` in `MetricsStore.loss_history` — O(1) append, bounded JSON size

**Exception handling in gradient pipeline:**
- `except (ValueError, RuntimeError)` → `log.warning` (expected operational failures)
- `except Exception` → `log.error` (programming bugs — should be investigated)

---

## 6. Phase Status

| Phase | Status | What was built |
|---|---|---|
| 1 — P2P Network | ✅ Complete | libp2p swarm, mDNS discovery, heartbeat, peer eviction |
| 2 — Gradient Extraction | ✅ Complete | `GradientExtractor`, `GradientSerializer` (SWRM protocol), `ModelShard` |
| 3 — Gradient Sync | ✅ Complete | FloodSub broadcast, chunking, `TimeoutAggregator`, FedAvg |
| 4 — Docker Simulation | ✅ Complete | 6-container sim (5 honest + 1 adversarial), chaos tests, 102 rounds |
| 5 — Internet Deployment | ✅ Code complete | HF models, HF datasets, NAT traversal, `/metrics` sidecar, checkpoints, Sybil resistance. **Deployment deferred until after Phase 8.** |
| 6 — Dashboard | ✅ Complete | Status table, force-directed peer graph, persistent loss history, bytes tracking |
| 7 — Distribution | ✅ Complete | `RunManifest`, `scripts/join.py`, `scripts/reconstruct_checkpoint.py`, `scripts/publish_checkpoint.py` |
| 8 — Competition | 🔲 Next | Two competing swarms, `make benchmark` determines winner, checkpoint publishing |

---

## 7. Key File Map

```
src/swarm_tune/
  config/settings.py          — NodeSettings (all env vars, validators)
  node/main.py                — SwarmNode training loop
  node/p2p/
    discovery.py              — PeerDiscovery, libp2p host, bootstrap/relay dialing
    gossip.py                 — GossipProtocol, chunking, receiver loop, eviction loop
    heartbeat.py              — liveness broadcast + peer eviction
    peer_selector.py          — AllPeersSelector, BanList
  node/trainer/
    model.py                  — ModelShard (MLP or HF AutoModelForCausalLM)
    data.py                   — DataShardLoader (.pt), HFDataShardLoader (HF datasets)
    gradient.py               — GradientExtractor (extract + validate)
    serializer.py             — GradientSerializer (SWRM wire protocol)
    compressor.py             — IdentityCompressor, TopKCompressor
  node/aggregator/
    aggregator.py             — TimeoutAggregator (straggler tolerance)
    averaging.py              — GradientAverager (FedAvg + subnet cap)
    strategy.py               — FlatAggregation, HierarchicalAggregation stub
  node/metrics.py             — MetricsStore, run_metrics_server() (aiohttp)
  runs/manifest.py            — RunManifest (training campaign definition)

runs/
  gpt2-wikitrain-001.json     — canonical 4-node GPT-2/WikiText training run
  gpt2-competition-001.json   — 50-round competition manifest (Phase 8 seed)

scripts/
  join.py                     — `python scripts/join.py --run-id X --node-index N`
  reconstruct_checkpoint.py   — merge or average sharded checkpoints
  publish_checkpoint.py       — upload to HuggingFace Hub with model card
  benchmark.py                — perplexity evaluation on WikiText-103 test split
  generate_shards.py          — generate synthetic .pt shards for local sim

docker/
  Dockerfile                  — 2-stage build; non-root user; dynamic HEALTHCHECK
  docker-compose.yml          — 6-node sim (node_0 bootstrap, node_5 adversarial)

dashboard/index.html          — static vanilla-JS dashboard, no build step
```

---

## 8. CI / Dev Commands

```bash
make check          # ruff lint + format check + mypy (mirrors CI)
make test           # pytest -m "unit or integration" --no-cov
make sim-up         # spin up 6-node Docker simulation
make benchmark CHECKPOINT=path/to/ckpt.pt
make join RUN_ID=gpt2-wikitrain-001 NODE_INDEX=0
make reconstruct CHECKPOINT_DIR=checkpoints/ MODEL=gpt2
```

- Coverage threshold: 60% (P2P network code is not coverable without live stack)
- `anyio_backend` fixture in `conftest.py` returns `"trio"` — required for all async tests
- Use `@pytest.mark.anyio` not `@pytest.mark.asyncio`
- 53 tests total

---

## 9. Phase 8 — What to Build Next

**Goal:** two teams run competing swarms (same base model, same dataset, fixed rounds). Winner determined by `make benchmark` perplexity. Results published publicly; anyone can verify.

**What's needed:**
1. Competition manifest fields already exist (`competition_id`, `team_id` in `RunManifest`)
2. `scripts/benchmark.py` already computes deterministic perplexity
3. `scripts/publish_checkpoint.py` already uploads to HuggingFace Hub
4. Need: end-to-end competition run with 2+ teams, 2+ nodes each, 50 rounds, scores published and independently verified

**No central leaderboard.** Each team publishes their checkpoint to HuggingFace Hub. Anyone verifies with `make benchmark CHECKPOINT=<path>`.

---

## 10. Phase 5 Deployment (When Ready)

Code is complete. Three steps to go live:
1. Provision a $5/month VPS, run: `docker run --net host swarmtune/node --env SWARM_NODE_KEY_SEED=relay_seed`
2. Update `runs/gpt2-wikitrain-001.json` with the relay VPS multiaddr in `bootstrap_peers`
3. Push a `v1.0.0` git tag → triggers `.github/workflows/publish.yml` (PyPI + Docker Hub)
