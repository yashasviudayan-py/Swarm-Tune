# Contributing to Swarm-Tune

Thank you for helping break the NVIDIA cluster monopoly.

## Before You Start

Read `CLAUDE.md`. Every architectural decision is documented there. If your contribution violates the three core rules (No Central Server, No Standard DDP, Straggler Tolerance), it will not be merged.

## Development Setup

```bash
git clone https://github.com/yashasviudayan-py/Swarm-Tune.git
cd Swarm-Tune
make bootstrap
```

This creates a virtual environment, installs all dependencies including dev tools, and installs pre-commit hooks.

## Making Changes

### Branching

```
main          — stable, protected
dev           — integration branch
feature/<name> — your feature branch
fix/<name>    — bug fix branch
```

Always branch from `dev`, not `main`.

```bash
git checkout dev
git checkout -b feature/your-feature-name
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(p2p): add mDNS peer discovery
fix(aggregator): handle empty gradient list on timeout
docs(readme): clarify straggler tolerance math
test(chaos): add 3-node failure scenario
chore(deps): bump torch to 2.4.0
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`

### Code Standards

- Python 3.12+ with full type annotations (`mypy --strict` clean)
- Formatting and linting via `ruff` (enforced by pre-commit hooks)
- All new logic must have unit tests
- New P2P behaviour must have integration tests

Run all checks before pushing:

```bash
make check    # lint + types
make test     # unit + integration tests
```

### Pull Requests

1. Open a PR against `dev`, not `main`
2. Fill in the PR template
3. Ensure all CI checks pass
4. Request a review

## Reporting Issues

Use the GitHub issue templates:
- **Bug Report** — something broken
- **Feature Request** — something missing

## Questions

Open a Discussion on GitHub, not an Issue.
