# Security Policy

## Scope

Swarm-Tune is a decentralized P2P system where untrusted nodes exchange serialized gradient tensors over the internet. The attack surface is significant and security is a first-class concern.

Known risk areas:
- **Gradient poisoning**: a malicious peer sends adversarially crafted gradients to corrupt the global model
- **Sybil attacks**: a single actor spawns many fake peers to dominate the aggregation
- **Deserialization vulnerabilities**: unsafe unpickling of tensor data from untrusted peers
- **Eclipse attacks**: isolating a node from honest peers by flooding its peer table with malicious entries

## Supported Versions

| Version | Supported |
|---------|-----------|
| `main`  | Yes       |
| older   | No        |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report privately via GitHub's Security Advisory feature:
`https://github.com/yashasviudayan-py/Swarm-Tune/security/advisories/new`

Include:
1. A clear description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

We will respond within 72 hours and aim to patch critical issues within 14 days.

## Security Principles

1. **Never `pickle.loads()` untrusted data.** Gradient deserialization uses `torch.load()` with `weights_only=True` to prevent arbitrary code execution.
2. **Peer identities are cryptographic.** libp2p peer IDs are derived from public keys. Spoofing a peer ID requires breaking the underlying key algorithm.
3. **Gradient bounds checking.** Received gradients are validated for shape, dtype, and magnitude before application.
