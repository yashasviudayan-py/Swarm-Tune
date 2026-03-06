"""
Per-node metrics HTTP sidecar server.

Each node runs a lightweight aiohttp server on port (node_port + 100) that
serves a single /metrics endpoint as JSON. This is a passive observer only:
it reads from a shared MetricsStore object that the training loop writes to.

There is no central aggregation backend. The dashboard/index.html polls
whatever node endpoints the user configures from their browser.

Rule compliance:
  - No FastAPI. Uses aiohttp (lightweight ASGI-free HTTP server).
  - Does not share state with the libp2p P2P stack.
  - No central server. Each node serves only its own data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

log: structlog.BoundLogger = structlog.get_logger(__name__)


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
    loss_history: list[float] = field(default_factory=list)

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
            "loss_history": self.loss_history,
            "peer_count": self.peer_count,
            "peer_ids": self.peer_ids,
            "gradient_rejections": self.gradient_rejections,
            "deferred_rounds": self.deferred_rounds,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
        }


async def run_metrics_server(store: MetricsStore, port: int) -> None:
    """
    Run the aiohttp metrics server until cancelled.

    Args:
        store: shared MetricsStore updated by the training loop.
        port: HTTP port to bind (typically node_port + 100).
    """
    try:
        from aiohttp import web
    except ImportError:
        log.warning("aiohttp not installed — metrics server disabled")
        return

    import json

    async def handle_metrics(request: web.Request) -> web.Response:
        data = json.dumps(store.to_dict(), indent=2)
        return web.Response(
            text=data,
            content_type="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    async def handle_health(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)

    try:
        await site.start()
    except OSError as exc:
        # Port already in use or permission denied. Disable metrics gracefully
        # rather than letting the exception propagate up to the TaskGroup and
        # cancel the entire training loop.
        await runner.cleanup()
        log.warning(
            "metrics server failed to start — training continues without metrics",
            port=port,
            error=str(exc),
        )
        return

    log.info("metrics server started", port=port, endpoint=f"http://0.0.0.0:{port}/metrics")

    try:
        # Run until this coroutine is cancelled by the task group.
        import anyio

        await anyio.sleep_forever()
    finally:
        await runner.cleanup()
        log.info("metrics server stopped")
