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

# Regex to extract the IPv4 address from a multiaddr like:
#   /ip4/192.168.1.1/tcp/9000/p2p/12D3KooW...
_IP4_RE = re.compile(r"^/ip4/([^/]+)/")

# Loopback and link-local prefixes to skip when choosing our own advertised address.
_SKIP_PREFIXES = ("127.", "169.254.")


@dataclass
class PeerInfo:
    """Immutable snapshot of a discovered peer."""

    peer_id: str
    multiaddr: str
    last_seen: float = field(default=0.0)
    # Cryptographic libp2p peer ID extracted from the multiaddr /p2p/... component.
    # Set when the peer includes their libp2p peer ID in their heartbeat payload.
    # Used to cross-check against the authenticated FloodSub message origin.
    libp2p_peer_id: str = field(default="")


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
        self._own_libp2p_id: str = ""
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
        self._own_libp2p_id = str(self._host.get_id())

        # Enter host.run() via AsyncExitStack so the TCP listener stays alive
        # for the full node lifetime without blocking here.
        self._stack = contextlib.AsyncExitStack()
        listen_addr = ma.Multiaddr(f"/ip4/{self._settings.host}/tcp/{self._settings.port}")
        await self._stack.enter_async_context(self._host.run([listen_addr]))

        # Choose the best advertised address: prefer the first non-loopback,
        # non-link-local address. Falling back to addrs[0] on a multi-interface
        # machine could advertise the loopback (127.x) or a Docker bridge that
        # is unreachable by external peers.
        addrs = self._host.get_addrs()
        self._own_multiaddr = self._pick_best_addr(addrs)

        log.info(
            "peer discovery started",
            node_id=self._settings.node_id,
            libp2p_peer_id=self._own_libp2p_id,
            own_multiaddr=self._own_multiaddr,
        )

        # --- Pubsub: FloodSub on the control topic for heartbeats ---
        # Note: py-libp2p implements FloodSub (simple broadcast to all
        # connected peers). GossipSub (mesh-based, O(log N) per message)
        # is available in go-libp2p and rust-libp2p but not in py-libp2p.
        # FloodSub is correct for ≤ 30 nodes; at 100+ nodes, migrating to
        # go-libp2p with GossipSub is the recommended upgrade path.
        router = FloodSub([TProtocol(FLOODSUB_PROTOCOL_ID)])
        self._pubsub = Pubsub(self._host, router)
        self._control_sub = await self._pubsub.subscribe(CONTROL_TOPIC)

        self._running = True

    @staticmethod
    def _pick_best_addr(addrs: list[object]) -> str | None:
        """
        Choose the best multiaddr to advertise to peers.

        Prefers non-loopback, non-link-local IPv4 addresses so peers on
        other machines can actually reach us. Falls back to addrs[0] if
        no better option exists (e.g. in a Docker environment where the only
        non-loopback address is the bridge IP — that's still the right choice).
        """
        if not addrs:
            return None
        addr_strs = [str(a) for a in addrs]
        for addr_str in addr_strs:
            m = _IP4_RE.match(addr_str)
            if not m:
                continue
            ip = m.group(1)
            if not any(ip.startswith(prefix) for prefix in _SKIP_PREFIXES):
                return addr_str
        # All addresses are loopback/link-local (e.g. pure IPv6 host).
        # Fall back to the first available address.
        return addr_strs[0]

    async def connect_bootstrap(self) -> None:
        """
        Connect to bootstrap and relay peers.

        Must be called *after* pubsub is running (background_trio_service),
        because the connection triggers notify_connected which enqueues to
        an unbuffered channel that handle_peer_queue consumes.

        When enable_relay is True, relay nodes are dialed first so that the
        node establishes a circuit before attempting bootstrap connections.
        This allows NAT'd nodes to be reachable before the DHT walk starts.
        """
        if self._settings.enable_relay and self._settings.relay_addrs:
            await self._connect_to_relay_peers()
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

    async def _connect_to_relay_peers(self) -> None:
        """
        Connect to circuit-relay nodes for NAT traversal (Phase 5+).

        Relay peers are dialed the same way as bootstrap peers. Once connected,
        the relay can forward inbound connections to this node so peers behind
        other NATs can reach us even without a direct route.

        Full dcutr hole-punching support depends on libp2p adding the dcutr
        protocol to py-libp2p. Until then, relay-assisted connections are the
        primary NAT traversal mechanism.
        """
        if self._host is None:
            raise RuntimeError(
                "_connect_to_relay_peers called before start() — host is not initialized."
            )
        log.info(
            "connecting to circuit-relay nodes",
            count=len(self._settings.relay_addrs),
            hole_punching=self._settings.enable_hole_punching,
        )
        connected = 0
        for addr_str in self._settings.relay_addrs:
            try:
                addr_str = self._resolve_multiaddr(addr_str)
                maddr = ma.Multiaddr(addr_str)
                if "p2p" not in addr_str:
                    log.warning(
                        "relay address has no /p2p/PEER_ID — skipping",
                        addr=addr_str,
                    )
                    continue
                log.info("connecting to relay peer", addr=addr_str)
                peer_info = info_from_p2p_addr(maddr)
                await self._host.connect(peer_info)
                log.info("relay peer connected", addr=addr_str)
                connected += 1
            except Exception:
                log.warning("failed to connect to relay peer", addr=addr_str, exc_info=True)
        if connected == 0 and self._settings.relay_addrs:
            log.error(
                "all relay connections failed — node may be unreachable behind NAT",
                attempted=len(self._settings.relay_addrs),
            )

    async def _connect_to_bootstrap_peers(self) -> None:
        if self._host is None:
            raise RuntimeError(
                "_connect_to_bootstrap_peers called before start() — host is not initialized."
            )
        connected = 0
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
                await self._host.connect(peer_info)
                log.info("bootstrap peer connected", addr=addr_str)
                connected += 1
            except Exception:
                log.warning(
                    "failed to connect to bootstrap peer",
                    addr=addr_str,
                    exc_info=True,
                )
        if connected == 0 and self._settings.bootstrap_peers:
            log.warning(
                "all bootstrap peer connections failed — relying on mDNS for local discovery",
                attempted=len(self._settings.bootstrap_peers),
            )

    # ------------------------------------------------------------------
    # Peer table management (called by Heartbeat)
    # ------------------------------------------------------------------

    def register_peer(
        self,
        peer_id: str,
        multiaddr: str,
        timestamp: float,
        libp2p_peer_id: str = "",
    ) -> None:
        """Called by the Heartbeat when a liveness signal arrives."""
        self._peers[peer_id] = PeerInfo(
            peer_id=peer_id,
            multiaddr=multiaddr,
            last_seen=timestamp,
            libp2p_peer_id=libp2p_peer_id,
        )

    def evict_peer(self, peer_id: str) -> None:
        """Called by the Heartbeat when a peer times out."""
        self._peers.pop(peer_id, None)
        log.info("peer evicted", peer_id=peer_id)

    def get_live_peers(self) -> list[PeerInfo]:
        """Return a snapshot of the current live peer table."""
        return list(self._peers.values())

    def get_peer_ip(self, node_id: str) -> str:
        """
        Extract the IPv4 address from a peer's multiaddr.

        Used to populate PeerGradient.peer_ip for Sybil resistance subnet
        capping. Returns empty string if the peer is unknown or if the
        multiaddr does not carry an /ip4/ component.

        The peer table is keyed by human-readable node_id (from the heartbeat
        payload). The multiaddr carries the peer's actual IP from the network
        layer (e.g. '/ip4/192.168.1.1/tcp/9000/p2p/12D3KooW...').
        """
        info = self._peers.get(node_id)
        if info is None:
            return ""
        m = _IP4_RE.match(info.multiaddr)
        return m.group(1) if m else ""

    @property
    def peer_count(self) -> int:
        return len(self._peers)

    @property
    def own_multiaddr(self) -> str | None:
        """Full multiaddr including /p2p/PEER_ID, usable as a bootstrap address."""
        return self._own_multiaddr

    @property
    def own_libp2p_id(self) -> str:
        """This node's cryptographic libp2p peer ID (Ed25519 hash)."""
        return self._own_libp2p_id

    @property
    def pubsub(self) -> Pubsub | None:
        """FloodSub pubsub instance; available after start()."""
        return self._pubsub

    @property
    def control_subscription(self) -> ISubscriptionAPI | None:
        """Subscription for the control topic; available after start()."""
        return self._control_sub
