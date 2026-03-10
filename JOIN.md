# Join a Swarm-Tune Training Run

This guide takes you from zero to a running Swarm-Tune node in under 5 minutes.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- A GPU with ≥ 8 GB VRAM (optional but recommended — CPU works for small models)
- A broadband internet connection
- The **bootstrap peer address** and **run ID** from the swarm organizer

---

## Step 1 — Get your configuration file

Download the template and fill in your details:

```bash
curl -O https://raw.githubusercontent.com/yashasviudayan-py/Swarm-Tune/main/my.env.template
cp my.env.template my.env
```

Open `my.env` in any text editor and set at minimum:

| Variable | What to set |
|---|---|
| `SWARM_NODE_ID` | Any unique name (e.g. `alice-rtx4090`) |
| `SWARM_BOOTSTRAP_PEERS` | Address from the organizer, e.g. `/ip4/1.2.3.4/tcp/9000/p2p/12D3Koo...` |
| `SWARM_MODEL_NAME` | Model name from the organizer (e.g. `gpt2`) |
| `SWARM_DATASET_NAME` | Dataset name (e.g. `wikitext`) |
| `SWARM_DATASET_CONFIG` | Dataset subset (e.g. `wikitext-103-raw-v1`) |
| `SWARM_DEVICE` | `cuda` for NVIDIA, `mps` for Apple Silicon, `cpu` for CPU-only |
| `SWARM_ENABLE_RELAY` | `true` if you are behind a home router (NAT) |
| `SWARM_BOOTSTRAP_PEERS` | Full multiaddr from the relay operator (includes peer ID) |

---

## Step 2 — Start your node

```bash
docker run --rm \
  --env-file my.env \
  --name swarm-tune-node \
  swarmtune/node:latest
```

**With NVIDIA GPU:**
```bash
docker run --rm --gpus all \
  --env-file my.env \
  --name swarm-tune-node \
  swarmtune/node:latest
```

**With Apple Silicon (MPS):**
MPS is not available inside Docker. Run natively instead:

```bash
pip install swarm-tune
swarm-tune --env-file my.env
```

---

## Step 3 — Verify you are connected

Watch the startup logs. Within 30 seconds you should see:

```
swarm node ready  device=cuda  peers=3  round=0
```

If you see `peers=0` after 60 seconds:
- Check that `SWARM_BOOTSTRAP_PEERS` is set correctly
- Make sure port `9000` (or your `SWARM_PORT`) is not blocked by a firewall
- Set `SWARM_ENABLE_RELAY=true` if you are behind NAT

---

## Step 4 — Let it run

The node will train for `SWARM_NUM_ROUNDS` rounds, averaging gradients with
peers after each round. You can watch progress via the metrics endpoint:

```
http://localhost:9100/metrics
```

(Port = `SWARM_PORT + 100`. Open `dashboard/index.html` in a browser for live charts.)

---

## Stopping gracefully

```bash
docker stop swarm-tune-node
```

The node saves a checkpoint before exiting. Checkpoints are stored in `./checkpoints/`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: libgmp` | Install GMP: `brew install gmp` (macOS) or `apt install libgmp-dev` (Linux) |
| `ConnectionRefusedError` on bootstrap | Verify the bootstrap address and that the organizer's node is running |
| `CUDA out of memory` | Reduce `SWARM_BATCH_SIZE` or switch to `SWARM_DEVICE=cpu` |
| `peers=0` after 60s | Enable relay: `SWARM_ENABLE_RELAY=true` |

---

## Running your own free relay (no VPS cost)

If you are organising a swarm and need a stable public bootstrap address, you can deploy
one for free on [fly.io](https://fly.io) — their free tier gives you 3 shared VMs +
160 GB/month bandwidth, which is more than enough for a Swarm-Tune relay.

```bash
# 1. Install flyctl (one-time)
curl -L https://fly.io/install.sh | sh

# 2. Log in (free account, no credit card for the free tier)
fly auth login

# 3. Deploy the relay — takes ~2 minutes
make relay-fly
```

The script creates the app, generates a stable Ed25519 key, deploys the container,
and prints the bootstrap multiaddr. Share that address with participants as
`SWARM_BOOTSTRAP_PEERS` in their `my.env` file.

**Alternatives if you prefer not to use fly.io:**

| Option | How it works | Best for |
|---|---|---|
| **fly.io** (recommended) | Permanent free VM | Groups training for weeks/months |
| **ngrok TCP tunnel** | `ngrok tcp 9000` on your machine | Quick demos, short sessions |
| **Tailscale** | All participants join same VPN | Trusted groups, no relay needed |
| **Oracle Cloud Free Tier** | Always-free ARM VM | Self-hosted, full control |

For ngrok: run `ngrok tcp 9000` on the bootstrap machine, use the `tcp://` address it gives
you as the bootstrap peer (wrap it in a multiaddr: `/dns4/<ngrok-host>/tcp/<port>/p2p/<peer-id>`).

---

## Native Python install (without Docker)

```bash
pip install swarm-tune
swarm-tune
```

All settings are read from environment variables or a `.env` file in the working directory.
