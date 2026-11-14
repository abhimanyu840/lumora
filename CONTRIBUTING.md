# Contributing to Lumora

Thank you for contributing to Lumora! This document provides guidelines and instructions for setting up the local development environment, testing code changes, and adhering to repository conventions.

## Code of Conduct & Principles

1. **Non-Destructive Principle:** Original image assets must never be modified in-place. All editing operations operate parametrically over the DAG pipeline.
2. **Offline-First:** Core editing features and AI models must execute locally on-device without requiring network access.
3. **Evidence Discipline:** Every bug report, PR, and architectural finding must reference concrete test results and path:line citations.

## Development Setup

### Rust Image Engine (`engine/`)

Ensure you have a recent Rust toolchain installed:
```powershell
cargo --version
```

Run test suite:
```powershell
cargo test --manifest-path engine/Cargo.toml
```

Format code:
```powershell
cargo fmt --manifest-path engine/Cargo.toml --all
```

Check with clippy:
```powershell
cargo clippy --manifest-path engine/Cargo.toml --all-targets -- -D warnings
```

### Backend (`backend/`)

Ensure Python 3.11+ is installed:
```powershell
python --version
```

Run pytest suite:
```powershell
python -m pytest backend/tests
```

### Container Services (`compose.yaml`)

Run the full local stack via Podman or Docker Compose:
```powershell
podman compose -f compose.yaml up -d
```

## Commit Guidelines

Follow Conventional Commits:
- `feat:` new user-facing functionality or core engine operation
- `fix:` bug fixes in algorithms or endpoints
- `perf:` performance improvements in image rendering loops
- `test:` test suite additions or benchmarks
- `docs:` documentation improvements
