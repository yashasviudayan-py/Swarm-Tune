"""
Gradient serialization for network transport.

Gradients must be converted to bytes to travel over libp2p.
We use torch.save() with an in-memory buffer — this is safe because
we set weights_only=True on the receiving end, which prevents
arbitrary code execution from malicious pickle payloads.

Wire format (v1):
  [4 bytes: magic]  [4 bytes: version]  [N bytes: torch-serialized dict]

The magic + version header lets future protocol versions coexist on
the same gossip topic without breaking existing nodes.
"""

from __future__ import annotations

import io
import struct

import structlog
import torch

log: structlog.BoundLogger = structlog.get_logger(__name__)

_MAGIC = b"SWRM"
_VERSION = 1
_HEADER_FORMAT = "!4sI"  # big-endian: 4-char magic + uint32 version
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)


class GradientSerializer:
    """Serializes and deserializes gradient dicts to/from bytes."""

    def serialize(self, gradients: dict[str, torch.Tensor]) -> bytes:
        """
        Convert a gradient dict to a byte payload suitable for network transport.

        Args:
            gradients: {param_name -> cpu tensor}

        Returns:
            Raw bytes with magic header + torch-serialized dict.
        """
        buf = io.BytesIO()
        buf.write(struct.pack(_HEADER_FORMAT, _MAGIC, _VERSION))
        torch.save(gradients, buf)
        payload = buf.getvalue()
        log.debug("serialized gradients", bytes=len(payload), params=len(gradients))
        return payload

    def deserialize(self, data: bytes) -> dict[str, torch.Tensor]:
        """
        Reconstruct a gradient dict from bytes received over the network.

        Args:
            data: raw bytes as produced by serialize().

        Returns:
            {param_name -> cpu tensor}

        Raises:
            ValueError: if the header is invalid or the version is unsupported.
        """
        if len(data) < _HEADER_SIZE:
            raise ValueError("Payload too short to contain a valid header.")

        magic, version = struct.unpack_from(_HEADER_FORMAT, data)
        if magic != _MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic!r}. Expected {_MAGIC!r}.")
        if version != _VERSION:
            raise ValueError(
                f"Unsupported protocol version {version}. This node supports v{_VERSION}."
            )

        buf = io.BytesIO(data[_HEADER_SIZE:])
        # weights_only=True prevents arbitrary code execution from pickle
        gradients: dict[str, torch.Tensor] = torch.load(buf, map_location="cpu", weights_only=True)
        log.debug("deserialized gradients", params=len(gradients))
        return gradients
