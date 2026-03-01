# CLAUDE.md — Swarm-Tune

## 1. Project Vision: Swarm-Tune

**The BitTorrent of AI Training.**

Swarm-Tune is a decentralized, peer-to-peer orchestration layer that allows multiple mid-range PCs to pool their VRAM over standard internet connections to fine-tune a machine learning model. No data center. No NVIDIA cluster monopoly. Just a swarm of commodity hardware collaborating to train models that would otherwise require infrastructure nobody can afford.

## 2. Hardware & Environment Constraints

- **Hardware:** Apple Silicon M4 Pro, 24 GB Unified Memory, 16-core GPU.
- **Compute Backend:** PyTorch optimized for `mps` (Metal Performance Shaders) where possible. CPU fallback is acceptable inside the Docker simulation.
- **Cost:** $0 budget. No external cloud GPUs or APIs. 100% local simulation.

## 3. The Tech Stack

| Layer | Technology |
|---|---|
| Networking | `libp2p` (Python implementation) — decentralized node discovery and gossip |
| Deep Learning | `PyTorch` — model instantiation and manual gradient extraction |
| Orchestration | `Docker` + `docker-compose` — simulate 4-5 disparate nodes locally |
| Language | Python 3.12+ with strict typing (`mypy`-clean) |

## 4. Strict Architectural Rules

These rules are non-negotiable. Follow them without exception.

### Rule 1: No Central Server

There is no master HTTP server (Flask, FastAPI, or otherwise) orchestrating the swarm. Nodes discover each other and coordinate exclusively through P2P gossip protocols via libp2p. Every node is equal.

### Rule 2: No Standard DDP

Do **not** rely on PyTorch's native `DistributedDataParallel` with standard backends (`nccl`, `gloo`) — those assume low-latency data-center interconnects. Instead, manually:

1. Extract gradients from the local backward pass.
2. Serialize them for network transport.
3. Transmit them to peers over libp2p.
4. Average the received gradients across participating nodes.

### Rule 3: Straggler Tolerance

The aggregation math must account for nodes that drop offline, rejoin, or suffer high latency. A slow or dead node must never block the swarm. Use timeout-based partial aggregation: if only 3 of 5 nodes report gradients within the window, average those 3 and move on.

## 5. Execution Phases (The Roadmap)

### Phase 1: P2P Network Initialization

- Bootstrap a libp2p swarm where nodes discover each other via mDNS or a known bootstrap peer.
- Implement a gossip-based heartbeat so every node knows who is alive.
- Deliverable: N Docker containers that find each other and maintain a live peer list.

### Phase 2: Local Gradient Extraction

- Instantiate a small model (e.g., a 2-layer MLP or tiny transformer) in PyTorch.
- Run a forward + backward pass on a local data shard.
- Extract raw gradients from `param.grad` tensors and serialize them (e.g., `torch.save` to bytes).
- Deliverable: Each node independently computes and serializes gradients.

### Phase 3: Gradient Synchronization

- Broadcast serialized gradients to all peers over the libp2p gossip channel.
- Each node collects gradients from peers, deserializes them, and computes a weighted average.
- Apply the averaged gradients to update the local model.
- Handle partial aggregation when not all peers respond within the timeout window.
- Deliverable: All live nodes converge to the same model weights after each round.

### Phase 4: Docker Simulation & Chaos Testing

- Use `docker-compose` to spin up 4-5 nodes, each with its own data shard.
- Train for multiple rounds and verify loss convergence across the swarm.
- Chaos testing: randomly kill containers mid-training, then bring them back. Prove the swarm recovers and continues training without corruption.
- Deliverable: Logged training run showing loss curves, node join/leave events, and successful convergence.
