"""
Timeout-based partial aggregation — the straggler tolerance mechanism.

This is the most important component for internet-scale deployment.

The problem: in a naive synchronous all-reduce, one offline node blocks
the entire swarm indefinitely. This is fatal for internet training where
nodes drop, restart, and have variable latency.

The solution: each training round has a hard deadline (configurable, default
30 seconds). When the deadline expires:
  - If >= min_peers have submitted gradients → proceed with partial average.
  - If < min_peers have responded → defer the round and retry.

Nodes that miss the deadline are NOT penalised permanently. They may submit
gradients in future rounds normally. The swarm never waits for the slowest
machine.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from swarm_tune.node.aggregator.averaging import GradientAverager, PeerGradient

if TYPE_CHECKING:
    import torch

    from swarm_tune.config.settings import NodeSettings

log: structlog.BoundLogger = structlog.get_logger(__name__)


class TimeoutAggregator:
    """
    Collects incoming peer gradients for a given round and enforces the
    timeout window before triggering the average.

    Lifecycle:
      1. open_round(round_number) — start collecting for this round.
      2. submit(peer_gradient)    — called by gossip handler for each arrival.
      3. await wait()             — blocks until timeout or min_peers reached.
      4. get_result()             — returns the averaged gradients.
    """

    def __init__(self, settings: NodeSettings) -> None:
        self._settings = settings
        self._averager = GradientAverager()
        self._round: int = -1
        self._contributions: list[PeerGradient] = []
        self._round_start: float = 0.0
        self._ready_event: asyncio.Event = asyncio.Event()

    def open_round(self, round_number: int) -> None:
        """Reset state for a new training round."""
        self._round = round_number
        self._contributions = []
        self._round_start = time.monotonic()
        self._ready_event = asyncio.Event()
        log.info("aggregation round opened", round=round_number)

    def submit(self, peer_gradient: PeerGradient) -> None:
        """
        Accept a gradient contribution from a peer.

        Idempotent: duplicate submissions from the same peer in the same round
        are silently dropped (the first one wins).
        """
        existing_ids = {c.peer_id for c in self._contributions}
        if peer_gradient.peer_id in existing_ids:
            log.debug("duplicate submission ignored", peer_id=peer_gradient.peer_id)
            return

        self._contributions.append(peer_gradient)
        log.debug(
            "gradient received",
            peer_id=peer_gradient.peer_id,
            round=self._round,
            total_received=len(self._contributions),
        )

        # Signal immediately if we already have enough peers
        if len(self._contributions) >= self._settings.min_peers_for_round:
            self._ready_event.set()

    async def wait(self) -> list[PeerGradient]:
        """
        Wait for either:
          a) min_peers submissions arrive (fast path), or
          b) the aggregation_timeout_secs deadline expires (straggler path).

        Returns the list of valid contributions collected so far.
        """
        timeout = self._settings.aggregation_timeout_secs
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            log.info(
                "aggregation round complete (quorum reached)",
                round=self._round,
                peers=len(self._contributions),
            )
        except TimeoutError:
            elapsed = time.monotonic() - self._round_start
            log.warning(
                "aggregation timeout — proceeding with partial result",
                round=self._round,
                peers_received=len(self._contributions),
                min_required=self._settings.min_peers_for_round,
                elapsed_secs=f"{elapsed:.1f}",
            )

        return self._contributions

    def get_averaged_gradients(self) -> dict[str, torch.Tensor]:
        """
        Run the weighted average over whatever contributions arrived.

        Raises:
            ValueError: if fewer than min_peers_for_round have contributed.
        """
        if len(self._contributions) < self._settings.min_peers_for_round:
            raise ValueError(
                f"Round {self._round}: only {len(self._contributions)} peers responded, "
                f"need at least {self._settings.min_peers_for_round}. Deferring round."
            )
        return self._averager.average(self._contributions)
