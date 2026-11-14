# Lumora

Lumora is a cross-platform mobile photo editing suite combining the precision tools of Adobe Lightroom (HSL, curves, color grading, non-destructive RAW pipeline) and Snapseed (selective control points, healing, structure, drama, perspective correction) with local on-device AI enhancements (DnCNN denoising, MiDaS depth/bokeh, MobileNet scene detection, Real-ESRGAN super-resolution, and Zero-DCE low-light enhancement).

Status: new project, 0 commits. Workspace initialized with standard AI directories and prompt archive.

## Stack

- **Mobile Client:** Exact (`abhimanyu840/exact`) runtime with Contract declarative UI and Rust logic modules targeting iOS and Android.
- **Image Processing Engine:** Rust (cargo 1.94.0) — zero-copy, SIMD-accelerated, non-destructive node graph pipeline for pixel operations, color curves, LUTs, and selective masks.
- **Local AI Inference:** On-device quantized execution (ONNX Runtime / Tract / Candle) running locally without cloud data transfer.
- **Backend Service:** Python 3.11.9 with FastAPI — user authentication, preset marketplace, and cloud sync.
- **Cloud Database:** PostgreSQL with SQLAlchemy ORM — user accounts, social presets, metadata storage.
- **Local Database:** SQLite with SQLAlchemy/SQLx — offline-first image metadata, non-destructive edit histories, and local cataloging.
- **Containerization & Dev:** Podman 6.1.4 (compose) and Docker 29.2.1 — containerized local development and production deployments.

## Commands

| Task | Command | Status |
|---|---|---|
| check git | `git --version` | `verified` — git version 2.53.0.windows.1 |
| check python | `python --version` | `verified` — Python 3.11.9 |
| check cargo | `cargo --version` | `verified` — cargo 1.94.0 (2026-01-15) |
| check podman | `podman --version` | `verified` — podman version 6.1.4 |
| check docker | `docker --version` | `verified` — Docker version 29.2.1 |
| check node | `node --version` | `verified` — v25.8.0 |
| check gh | `gh --version` | `verified` — gh version 2.88.0 |
| backend install | `pip install -r backend/requirements.txt` | `unverified` — backend not scaffolded yet |
| backend run | `uvicorn backend.app.main:app --reload --port 8000` | `unverified` — backend not scaffolded yet |
| engine build | `cargo build --manifest-path engine/Cargo.toml` | `unverified` — engine not scaffolded yet |
| engine test | `cargo test --manifest-path engine/Cargo.toml` | `unverified` — engine not scaffolded yet |
| podman compose up | `podman compose -f compose.yaml up -d` | `unverified` — compose.yaml not written yet |

## Layout (Planned)

```
ai-workspace/
  plans/              # Deep architecture and execution plans (created)
  prompts/            # Source user prompts and task specs (created)
  research/           # Deep research findings and competitive teardowns (created)
  scratch/            # Ephemeral build artifacts and tests (ignored)
backend/              # FastAPI cloud sync & marketplace service (planned)
  app/
    api/              # Routes (auth, presets, marketplace)
    core/             # Config, security, database session
    models/           # SQLAlchemy ORM models
    schemas/          # Pydantic data transfer schemas
engine/               # Rust core image processing engine (planned)
  crates/
    lumora-core/      # Pixel buffer, color spaces, non-destructive node graph
    lumora-ops/       # Lightroom & Snapseed image manipulation kernels
    lumora-ai/        # On-device model execution bindings (ONNX / Tract)
mobile/               # Exact runtime mobile application (planned)
  contracts/          # Contract UI view declarations
  logic/              # Rust / TS logic modules bridging engine and UI
models/               # Quantized model conversion & weights manifest (planned)
compose.yaml          # Podman / Docker multi-container composition (planned)
```

## Conventions

- **Environment & Shell:** Windows 11 host. Always use PowerShell commands (e.g. `Get-ChildItem`, `Get-Content`, `;` chaining, not `&&`).
- **Non-Destructive Image Pipeline:** Original image files are strictly read-only. Edits are recorded as non-destructive parameter graphs (curves, HSL adjustments, selective mask coordinates, active AI passes) and rendered in real-time or exported on demand.
- **Offline-First Principle:** Core editing and AI models execute strictly on-device without mandatory internet connection. Cloud connectivity is reserved for auth, preset sharing, and optional backup.
- **Evidence Discipline:** Every factual statement about code or configuration must cite `path:line`. Never describe code without opening it.
- **Secrets Management:** Secrets and API keys reside in `.env` files (excluded by `.gitignore`), never hardcoded in source.

## First Slice

1. Initialize Git repository baseline and initial commit.
2. Scaffold `engine/` Rust library with non-destructive image adjusters (brightness, contrast, saturation, HSL curves).
3. Scaffold `backend/` FastAPI service with health check, SQLite/PostgreSQL database models, and JWT authentication.
4. Set up `compose.yaml` for Podman/Docker to spin up PostgreSQL and FastAPI service.
5. Create benchmark test demonstrating Rust engine rendering a 12MP image under 30ms.

Done when:
- `cargo test --manifest-path engine/Cargo.toml` passes with pixel manipulation tests.
- `podman compose up` spins up backend and responds with HTTP 200 on `/health`.

## Not Yet Built

- **Mobile Application (`mobile/`):** Exact Contract layouts, gesture canvas, and mobile bridge not started.
- **Image Processing Engine (`engine/`):** Rust crates not yet scaffolded; color grading, curve math, and selective masks not implemented.
- **On-Device AI Pipeline (`models/`):** Model quantizations (DnCNN, MiDaS, MobileNet, Real-ESRGAN, Zero-DCE) and runtime bindings not yet prepared.
- **Cloud Backend (`backend/`):** FastAPI app, database migrations, and preset marketplace endpoints not yet created.
- **Test Suites:** No unit, integration, or visual regression tests created yet.

## Verification

While application code is not yet written, toolchain environment is verified:
- Python 3.11.9, Cargo 1.94.0, Podman 6.1.4, Docker 29.2.1, Node v25.8.0, Git 2.53.0.
- First runnable verification will be:
  `cargo check --manifest-path engine/Cargo.toml` and `python -m pytest backend/tests` once initial slice is scaffolded.

## Open Questions

1. **Exact Framework Android Support:** Exact currently officially documents macOS, iOS, web, and Linux. For Android deployment, will we use Exact's native C/Rust JNI bridge with Android SurfaceView, or a dedicated Android host runtime?
2. **On-Device AI Runtime Engine:** Selection between ONNX Runtime Mobile (C++ / Rust bindings), Tract (pure Rust ONNX inference), and NCNN (Tencent's mobile-optimized neural network engine) for executing quantized models on mobile.
3. **Snapseed Control Point Algorithm:** Implementation strategy for Snapseed's patented "Control Point" U-Point selective masking algorithm (luminance/chrominance Euclidean distance weighting in Lab color space).
4. **Preset Marketplace Storage:** Cloud storage provider for preset files and preview assets (local filesystem / MinIO S3 for Podman vs AWS S3).
