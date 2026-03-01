# Architecture Deep-Dive

## Overview

Swarm-Tune is structured as three loosely-coupled subsystems wired together by the `SwarmNode` orchestrator:

```
SwarmNode
├── P2P Layer      ← Who's alive? How do we talk to them?
├── Trainer        ← What's our local gradient?
└── Aggregator     ← What does the swarm agree on?
```

## Data Flow per Training Round

```
1. open_round(N)
        │
        ▼
2. local forward() + backward()
        │
        ▼
3. GradientExtractor.extract(model)
        │   → {param_name: tensor} on CPU
        ▼
4. GradientSerializer.serialize(gradients)
        │   → bytes with SWRM header
        ▼
5. GossipProtocol.broadcast_gradient(message)
        │   → sent to all peers via libp2p GossipSub
        │
        │   ← peers broadcast their own gradients simultaneously
        ▼
6. TimeoutAggregator.wait()
        │   → collects up to aggregation_timeout_secs
        │   → partial result is OK if >= min_peers respond
        ▼
7. GradientAverager.average(contributions)
        │   → FedAvg weighted by dataset_size
        ▼
8. ModelShard.apply_averaged_gradients(averaged)
        │   → overwrite param.grad, step optimizer
        ▼
9. Repeat for round N+1
```

## Module Dependency Graph

```
config.settings
       │
       ├──► node.p2p.discovery
       │         └──► node.p2p.heartbeat
       │
       ├──► node.p2p.gossip
       │
       ├──► node.trainer.model
       │         └──► node.trainer.gradient
       │         └──► node.trainer.serializer
       │
       └──► node.aggregator.timeout
                 └──► node.aggregator.averaging
```

No circular imports. Each module has a single clear responsibility.

## Wire Protocol (v1)

Gradient messages use a simple framed binary format:

```
┌──────────┬───────────┬──────────────────────────┐
│ 4 bytes  │  4 bytes  │       N bytes            │
│  "SWRM"  │  version  │  torch.save() payload    │
│  (magic) │  (uint32) │  (weights_only=True)     │
└──────────┴───────────┴──────────────────────────┘
```

The magic + version header allows protocol evolution while keeping backward compatibility on the gossip topic.

## Fault Tolerance Model

| Scenario | Behaviour |
|---|---|
| Node goes offline mid-round | Evicted from peer table after 20s. Missing from aggregation. |
| Node rejoins | Heartbeat re-registers it. Included in next round. |
| < min_peers respond | Round deferred. Retry next cycle. |
| >= min_peers respond | Partial aggregation proceeds. Missing nodes skipped. |
| Malicious gradient (NaN/Inf/large norm) | Rejected at deserialization. Peer not penalised (single incident). |

## Scaling Considerations

The current implementation uses GossipSub which is O(log N) in broadcast cost. At 20 nodes with ~24 GB of VRAM each:

- A 7B parameter model in fp16 = ~14 GB of weights → gradient size ≈ 14 GB per round
- At 10 Mbps uplink per node, this is ~190 minutes per gradient exchange

**Practical optimisations for future phases:**
1. **Gradient compression**: Top-K sparsification or 1-bit quantisation can reduce gradient size by 100–1000x.
2. **Asynchronous SGD**: Nodes don't wait for peers at all — they apply whatever gradients arrive. Convergence is slower but throughput is higher.
3. **Layer-wise pipelining**: Broadcast gradients layer by layer as they're computed, overlapping compute and communication.
