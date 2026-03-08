# =============================================================
# Stage 1: Builder — install Python deps into a clean layer
# =============================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only what's needed to install dependencies first
# (leverages Docker layer caching — deps rarely change)
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# =============================================================
# Stage 2: Runtime — lean image for actual node execution
# =============================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Create a non-root user for security
RUN useradd -m -u 1001 swarm && chown -R swarm:swarm /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src

# Create directories for runtime data (mounted as volumes).
# HF_HOME is set to /app/.cache/huggingface so the non-root user (swarm, UID 1001)
# can write model downloads. Without this, HuggingFace defaults to ~/.cache which
# resolves to /root/.cache — inaccessible to UID 1001.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/data/shards /app/logs /app/checkpoints /app/.cache/huggingface \
    && chown -R swarm:swarm /app

USER swarm

# Expose libp2p default port
EXPOSE 9000

# Health check: probe the metrics sidecar HTTP endpoint (port = SWARM_PORT + 100).
# Reads SWARM_PORT from the container environment so the check works for any port,
# not just the default 9100 — fixes a bug where multi-node compose stacks with
# different SWARM_PORT values all reported unhealthy because the check was hardcoded.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "\
import os, urllib.request; \
port = int(os.environ.get('SWARM_PORT', 9000)) + 100; \
urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5)" \
    || exit 1

# Default entrypoint — override NODE_ID and SWARM_PORT per container
ENTRYPOINT ["python", "-m", "swarm_tune.node.main"]
