"""
Peer discovery for the Swarm-Tune P2P network.

Strategy:
  - Local simulation (Docker): mDNS-based discovery within the Docker bridge network.
  - Internet deployment: bootstrap peer + libp2p Kademlia DHT.

There is NO central tracker. A bootstrap peer is simply the first known
address — once connected, the DHT takes over and the node is fully
decentralised.

Security (Phase 1):
  Peer IDs are derived from Ed25519 public keys via libp2p's standard
  multihash encoding. Spoofing a peer ID requires breaking the key.
  We never trust a peer ID string from the wire; libp2p verifies identity
  during the Noise handshake before any application data flows.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import multiaddr as ma
import structlog
from libp2p import create_new_ed25519_key_pair, new_host  # type: ignore[attr-defined]
from libp2p.custom_types import TProtocol
from libp2p.peer.peerinfo import info_from_p2p_addr
from libp2p.pubsub.floodsub import FloodSub
from libp2p.pubsub.pubsub import Pubsub

from swarm_tune.node.p2p.gossip import CONTROL_TOPIC

if TYPE_CHECKING:
    from libp2p import IHost  # type: ignore[attr-defined]
    from libp2p.pubsub.subscription import ISubscriptionAPI  # type: ignore[attr-defined]

    from swarm_tune.config.settings import NodeSettings

log: structlog.BoundLogger = structlog.get_logger(__name__)

FLOODSUB_PROTOCOL_ID = "/floodsub/1.0.0"


@dataclass
class PeerInfo:
    """Immutable snapshot of a discovered peer."""

    peer_id: str
    multiaddr: str
    last_seen: float = field(default=0.0)


class PeerDiscovery:
    """
    Manages peer discovery and maintains the live peer table.

    The peer table is a dict[peer_id -> PeerInfo]. Peers are evicted
    when the Heartbeat component stops receiving their liveness signals.

    Lifecycle (managed via AsyncExitStack so host.run() stays open):
      await discovery.start()   # binds TCP, starts mDNS, connects bootstrap
      ...                        # node runs
      await discovery.stop()    # closes host cleanly
    """

    def __init__(self, settings: NodeSettings) -> None:
        self._settings = settings
        self._peers: dict[str, PeerInfo] = {}
        self._running = False
        self._host: IHost | None = None
        self._pubsub: Pubsub | None = None
        self._control_sub: ISubscriptionAPI | None = None
        self._own_multiaddr: str | None = None
        self._stack: contextlib.AsyncExitStack | None = None

    async def start(self) -> None:
        """
        Start the libp2p host and connect to bootstrap peers.

        Creates a fresh Ed25519 key pair (or a deterministic one if
        node_key_seed is configured) so the peer ID is cryptographically bound
        to this node's private key — Phase 1 security gate.

        Raises:
            RuntimeError: if the host cannot bind to the configured port.
        """
        # --- Identity: Ed25519 key pair → cryptographic peer ID ---
        seed_str = self._settings.node_key_seed
        if seed_str:
            seed_bytes = hashlib.sha256(seed_str.encode()).digest()
            key_pair = create_new_ed25519_key_pair(seed=seed_bytes)
            log.info("using deterministic key pair", seed_label=seed_str)
        else:
            key_pair = create_new_ed25519_key_pair()

        # --- Host: TCP transport + Noise security + mDNS discovery ---
        self._host = new_host(key_pair=key_pair, enable_mDNS=True)

        # Enter host.run() via AsyncExitStack so the TCP listener stays alive
        # for the full node lifetime without blocking here.
        self._stack = contextlib.AsyncExitStack()
        listen_addr = ma.Multiaddr(f"/ip4/{self._settings.host}/tcp/{self._settings.port}")
        await self._stack.enter_async_context(self._host.run([listen_addr]))

        # addrs already include the /p2p/PEER_ID suffix in libp2p 0.6
        addrs = self._host.get_addrs()
        self._own_multiaddr = str(addrs[0]) if addrs else None

        log.info(
            "peer discovery started",
            node_id=self._settings.node_id,
            peer_id=str(self._host.get_id()),
            own_multiaddr=self._own_multiaddr,
        )

        # --- Pubsub: FloodSub on the control topic for heartbeats ---
        router = FloodSub([TProtocol(FLOODSUB_PROTOCOL_ID)])
        self._pubsub = Pubsub(self._host, router)
        self._control_sub = await self._pubsub.subscribe(CONTROL_TOPIC)

        self._running = True

    async def connect_bootstrap(self) -> None:
        """
        Connect to bootstrap peers.

        Must be called *after* pubsub is running (background_trio_service),
        because the connection triggers notify_connected which enqueues to
        an unbuffered channel that handle_peer_queue consumes.
        """
        await self._connect_to_bootstrap_peers()

    async def stop(self) -> None:
        """Gracefully shut down the libp2p host."""
        self._running = False
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        log.info("peer discovery stopped", node_id=self._settings.node_id)

    @staticmethod
    def _resolve_multiaddr(addr_str: str) -> str:
        """
        Resolve any hostname in an /ip4/<host>/... multiaddr to its numeric IP.

        The multiaddr library only accepts dotted-decimal IPv4 strings for the
        /ip4/ protocol component.  Docker bridge networks assign container
        hostnames (e.g. 'node_0') that must be DNS-resolved to an IP before
        the Multiaddr can be constructed.  Real IPs pass through unchanged.
        """
        m = re.match(r"^(/ip4/)([^/]+)(/.+)$", addr_str)
        if not m:
            return addr_str
        host = m.group(2)
        try:
            socket.inet_pton(socket.AF_INET, host)  # already a valid IPv4
            return addr_str
        except OSError:
            pass
        try:
            ip = socket.getaddrinfo(host, None, socket.AF_INET)[0][4][0]
            return f"{m.group(1)}{ip}{m.group(3)}"
        except OSError:
            log.warning("could not resolve bootstrap hostname", hostname=host)
            return addr_str

    async def _connect_to_bootstrap_peers(self) -> None:
        for addr_str in self._settings.bootstrap_peers:
            try:
                addr_str = self._resolve_multiaddr(addr_str)
                maddr = ma.Multiaddr(addr_str)
                # Only connect if the address carries a /p2p/PEER_ID component.
                # Addresses without peer IDs cannot be verified by the Noise
                # handshake and are skipped (mDNS will handle local discovery).
                if "p2p" not in addr_str:
                    log.warning(
                        "bootstrap address has no /p2p/PEER_ID — skipping "
                        "(set SWARM_NODE_KEY_SEED on the bootstrap node to get "
                        "a stable peer ID)",
                        addr=addr_str,
                    )
                    continue

                log.info("connecting to bootstrap peer", addr=addr_str)
                peer_info = info_from_p2p_addr(maddr)
                assert self._host is not None
                await self._host.connect(peer_info)
                log.info("bootstrap peer connected", addr=addr_str)
            except Exception:
                log.warning(
                    "failed to connect to bootstrap peer",
                    addr=addr_str,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Peer table management (called by Heartbeat)
    # ------------------------------------------------------------------

    def register_peer(self, peer_id: str, multiaddr: str, timestamp: float) -> None:
        """Called by the Heartbeat when a liveness signal arrives."""
        self._peers[peer_id] = PeerInfo(peer_id=peer_id, multiaddr=multiaddr, last_seen=timestamp)

    def evict_peer(self, peer_id: str) -> None:
        """Called by the Heartbeat when a peer times out."""
        self._peers.pop(peer_id, None)
        log.info("peer evicted", peer_id=peer_id)

    def get_live_peers(self) -> list[PeerInfo]:
        """Return a snapshot of the current live peer table."""
        return list(self._peers.values())

    @property
    def peer_count(self) -> int:
        return len(self._peers)

    @property
    def own_multiaddr(self) -> str | None:
        """Full multiaddr including /p2p/PEER_ID, usable as a bootstrap address."""
        return self._own_multiaddr

    @property
    def pubsub(self) -> Pubsub | None:
        """FloodSub pubsub instance; available after start()."""
        return self._pubsub

    @property
    def control_subscription(self) -> ISubscriptionAPI | None:
        """Subscription for the control topic; available after start()."""
        return self._control_sub
