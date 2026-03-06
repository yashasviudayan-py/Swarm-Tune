"""
Shared pytest fixtures for the Swarm-Tune test suite.

Fixtures here are available to all test modules without explicit import.
"""

from __future__ import annotations

import pytest
import torch

from swarm_tune.config.settings import NodeSettings
from swarm_tune.node.aggregator.averaging import PeerGradient


@pytest.fixture()
def anyio_backend() -> str:
    """All anyio-marked tests run with the trio backend (required by libp2p)."""
    return "trio"


# ---------------------------------------------------------------------------
# Settings fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_settings() -> NodeSettings:
    """Minimal node settings suitable for unit tests (no network, no disk)."""
    return NodeSettings(
        node_id="test_node",
        host="127.0.0.1",
        port=19000,
        bootstrap_peers=[],
        model_name="mlp",  # toy MLP: no HuggingFace download in unit tests
        learning_rate=1e-3,
        batch_size=4,
        num_rounds=2,
        aggregation_timeout_secs=2.0,
        min_peers_for_round=2,
        device="cpu",
        log_level="WARNING",
        log_format="console",
    )


@pytest.fixture()
def multi_node_settings() -> list[NodeSettings]:
    """Settings for a 3-node integration test swarm."""
    return [
        NodeSettings(
            node_id=f"node_{i}",
            host="127.0.0.1",
            port=19000 + i,
            bootstrap_peers=[] if i == 0 else ["/ip4/127.0.0.1/tcp/19000"],
            model_name="mlp",  # toy MLP: no HuggingFace download in integration tests
            batch_size=4,
            num_rounds=2,
            aggregation_timeout_secs=5.0,
            min_peers_for_round=2,
            device="cpu",
            log_level="WARNING",
            log_format="console",
        )
        for i in range(3)
    ]


# ---------------------------------------------------------------------------
# Gradient fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_gradients() -> dict[str, torch.Tensor]:
    """A small gradient dict that mimics a 2-layer model."""
    return {
        "0.weight": torch.randn(256, 128),
        "0.bias": torch.randn(256),
        "2.weight": torch.randn(128, 256),
        "2.bias": torch.randn(128),
    }


@pytest.fixture()
def peer_gradient(simple_gradients: dict[str, torch.Tensor]) -> PeerGradient:
    return PeerGradient(
        peer_id="test_peer",
        gradients=simple_gradients,
        dataset_size=100,
    )


@pytest.fixture()
def three_peer_gradients(simple_gradients: dict[str, torch.Tensor]) -> list[PeerGradient]:
    """Three peers with equal dataset sizes and slightly different gradients."""
    return [
        PeerGradient(
            peer_id=f"peer_{i}",
            gradients={k: v + (i * 0.01) for k, v in simple_gradients.items()},
            dataset_size=100,
        )
        for i in range(3)
    ]
