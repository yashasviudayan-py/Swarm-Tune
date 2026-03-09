"""
Per-node metrics HTTP sidecar server.

Each node runs a lightweight HTTP server on port (node_port + 100) that
serves a single /metrics endpoint as JSON. This is a passive observer only:
it reads from a shared MetricsStore object that the training loop writes to.

There is no central aggregation backend. The dashboard/index.html polls
whatever node endpoints the user configures from their browser.

Implementation note:
  Uses anyio's TCP primitives directly instead of aiohttp. aiohttp uses
  asyncio internally and will fail under the trio backend (no asyncio event
  loop present). The metrics server only handles two trivial GET routes, so
  a minimal raw-TCP HTTP/1.1 handler is simpler and fully trio-compatible.

Rule compliance:
  - No FastAPI. No aiohttp. Uses anyio TCP (backend-agnostic).
  - Does not share state with the libp2p P2P stack.
  - No central server. Each node serves only its own data.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import anyio
import anyio.abc
import structlog

log: structlog.BoundLogger = structlog.get_logger(__name__)

# Maximum request size we'll read from a client (prevents memory exhaustion).
_MAX_REQUEST_BYTES = 8192

# Sliding window for loss history. 1000 points is enough for smooth loss curves
# on the dashboard while keeping JSON serialization cost constant across long runs.
_LOSS_HISTORY_CAP = 1000


@dataclass
class MetricsStore:
    """
    Shared state written by the training loop and read by the HTTP server.

    All writes happen from the main training coroutine (single writer).
    Reads happen from the HTTP handler coroutine. No lock needed because
    Python's GIL protects individual attribute assignments.
    """

    node_id: str = ""
    start_time: float = field(default_factory=time.time)

    # Training progress
    current_round: int = 0
    total_rounds: int = 0
    # Capped at _LOSS_HISTORY_CAP entries (sliding window) to prevent unbounded
    # memory growth and ever-increasing JSON serialization cost in long runs.
    loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=_LOSS_HISTORY_CAP))

    # Peer state
    peer_count: int = 0
    peer_ids: list[str] = field(default_factory=list)

    # Gradient health
    gradient_rejections: int = 0
    deferred_rounds: int = 0

    # Network throughput (bytes)
    bytes_sent: int = 0
    bytes_received: int = 0

    def record_round(self, round_num: int, loss: float) -> None:
        self.current_round = round_num
        self.loss_history.append(round(loss, 6))

    def record_rejection(self) -> None:
        self.gradient_rejections += 1

    def record_deferred(self) -> None:
        self.deferred_rounds += 1

    def update_peers(self, peer_ids: list[str]) -> None:
        self.peer_ids = peer_ids
        self.peer_count = len(peer_ids)

    def to_dict(self) -> dict[str, Any]:
        uptime = round(time.time() - self.start_time, 1)
        last_loss = self.loss_history[-1] if self.loss_history else None
        return {
            "node_id": self.node_id,
            "uptime_secs": uptime,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "last_loss": last_loss,
            "loss_history": list(self.loss_history),
            "peer_count": self.peer_count,
            "peer_ids": self.peer_ids,
            "gradient_rejections": self.gradient_rejections,
            "deferred_rounds": self.deferred_rounds,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
        }


def _make_response(status: str, content_type: str, body: str) -> bytes:
    body_bytes = body.encode()
    headers = (
        f"HTTP/1.1 {status}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body_bytes)}\r\n"
        f"Access-Control-Allow-Origin: *\r\n"
        f"X-Content-Type-Options: nosniff\r\n"
        f"X-Frame-Options: DENY\r\n"
        f"Cache-Control: no-store, no-cache\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    return headers.encode() + body_bytes


async def _handle_client(client: anyio.abc.ByteStream, store: MetricsStore) -> None:
    """Handle a single HTTP connection."""
    try:
        data = await client.receive(_MAX_REQUEST_BYTES)
        request = data.decode("utf-8", errors="replace")
        first_line = request.split("\r\n", 1)[0]
        parts = first_line.split(" ")
        method = parts[0] if parts else "GET"
        path = parts[1] if len(parts) >= 2 else "/"

        # Only GET is valid — reject everything else before touching any state.
        if method != "GET":
            response = _make_response("405 Method Not Allowed", "text/plain", "method not allowed")
            await client.send(response)
            return

        if path.startswith("/health"):
            response = _make_response("200 OK", "text/plain", "ok")
        elif path.startswith("/metrics"):
            body = json.dumps(store.to_dict(), indent=2)
            response = _make_response("200 OK", "application/json", body)
        else:
            response = _make_response("404 Not Found", "text/plain", "not found")

        await client.send(response)
    except (ConnectionError, BrokenPipeError, EOFError, OSError):
        # Client disconnected before we finished — expected and benign.
        pass
    except Exception:
        # Unexpected error (e.g. JSON serialization bug) — log so it's visible.
        log.warning("metrics handler error", exc_info=True)
    finally:
        await client.aclose()


async def run_metrics_server(store: MetricsStore, port: int) -> None:
    """
    Run the metrics HTTP server until cancelled.

    Binds a TCP listener on the given port and dispatches each connection
    to _handle_client in a new task. Uses anyio TCP primitives (no asyncio)
    so it is fully compatible with the trio backend used by libp2p.

    Args:
        store: shared MetricsStore updated by the training loop.
        port:  HTTP port to bind (typically node_port + 100).
    """
    try:
        listener = await anyio.create_tcp_listener(local_port=port, local_host="0.0.0.0")
    except OSError as exc:
        # Port already in use or permission denied. Disable metrics gracefully
        # rather than letting the exception propagate up to the TaskGroup and
        # cancel the entire training loop.
        log.warning(
            "metrics server failed to start — training continues without metrics",
            port=port,
            error=str(exc),
        )
        return

    log.info("metrics server started", port=port, endpoint=f"http://0.0.0.0:{port}/metrics")

    async def _serve(client: anyio.abc.ByteStream) -> None:
        await _handle_client(client, store)

    async with listener:
        await listener.serve(_serve)
