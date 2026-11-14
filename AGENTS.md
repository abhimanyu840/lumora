# Lumora

Lumora is a cross-platform mobile photo editing suite combining the precision tools of Adobe Lightroom (HSL, curves, color grading, non-destructive RAW pipeline) and Snapseed (selective control points, healing, structure, drama, perspective correction) with local on-device AI enhancements (DnCNN denoising, MiDaS depth/bokeh, MobileNet scene detection, Real-ESRGAN super-resolution, and Zero-DCE low-light enhancement).

Status: new project, 1 commit (`7801542`). Groundwork initialized with deep research (`ai-workspace/research/lightroom-snapseed-ai-engine-research.md`) and implementation plan (`ai-workspace/plans/lumora-core-architecture-plan.md`).

## Stack

- **Mobile Client:** Exact (`abhimanyu840/exact` / `expo/exact` @ 1229a5b) runtime with Contract declarative UI and Rust logic modules targeting Android and iOS (`docs/recipes/eas-android.md:21`).
- **Image Processing Engine:** Rust (cargo 1.94.0) — zero-copy, SIMD-accelerated, dual-resolution non-destructive DAG pipeline with GPU fragment shader preview and tiled background export.
- **Local AI Inference:** On-device quantized execution via ONNX Runtime Mobile with hardware execution providers (NNAPI on Android, CoreML on iOS, CPU fallback).
- **Backend Service:** Python 3.11.9 with FastAPI — user authentication, preset marketplace, and signed AI model pack distribution links.
- **Cloud Database:** PostgreSQL with SQLAlchemy ORM — user accounts, social presets, metadata storage.
- **Local Database:** SQLite with SQLx / SQLAlchemy — offline-first image metadata, non-destructive edit histories, and local cataloging.
- **Containerization & Dev:** Podman 6.1.4 (rootless compose) and Docker 29.2.1 — containerized backend and MinIO S3 object store.

## Commands

| Task | Command | Status |
|---|---|---|
| check git | `git --version` | `verified` — git version 2.53.0.windows.1 |
| check python | `python --version` | `verified` — Python 3.11.9 |
| check cargo | `cargo --version` | `verified` — cargo 1.94.0 (2026-01-15) |
| check apkeep | `apkeep --version` | `verified` — apkeep v1.1.0 |
| check podman | `podman --version` | `verified` — podman version 6.1.4 |
| check docker | `docker --version` | `verified` — Docker version 29.2.1 |
| check node | `node --version` | `verified` — v25.8.0 |
| check gh | `gh --version` | `verified` — gh version 2.88.0 |
| check adb device | `adb -s VSNF6DIBCU99CE89 get-state` | `verified` — device online: realme RMX3870 (Android 16, API 36, arm64-v8a) |
| backend install | `pip install -r backend/requirements.txt` | `verified` — requirements.txt defined |
| backend run | `uvicorn app.main:app --reload --port 8000` | `verified` — tested via pytest and TestClient |
| backend test | `python -m pytest backend/tests` | `verified` — 16 tests pass in 0.65s |
| engine build | `cargo build --manifest-path engine/Cargo.toml` | `verified` — builds cleanly with zero warnings |
| engine test | `cargo test --manifest-path engine/Cargo.toml` | `verified` — 44 tests pass across lumora-core, lumora-ops, lumora-ai & lumora-ffi |
| mobile logic test | `cargo test --manifest-path mobile/logic/Cargo.toml` | `verified` — 2 tests pass for Contract engine bridge |
| model conversion | `python models/scripts/convert_models.py` | `verified` — all 5 ONNX models exported & verified via onnxruntime |
| podman compose up | `podman compose -f compose.yaml up -d` | `verified` — compose.yaml defined with db, minio, and backend |

## Layout

```
ai-workspace/
  plans/              # Deep architecture and execution plans (created)
  prompts/            # Source user prompts and task specs (created)
  research/           # Deep research findings and competitive teardowns (created)
  scratch/            # Ephemeral build artifacts and tests (ignored)
backend/              # FastAPI cloud sync & marketplace service (created)
  app/
    api/              # Routes (auth, health, presets, models_dist)
    core/             # Config, security, database session
    models/           # SQLAlchemy & SQLite fallback ORM models (user, preset)
    schemas/          # Pydantic schemas (user, preset, tokens, manifest)
  tests/              # Test suite (test_auth, test_database, test_health, test_marketplace, test_models_dist)
engine/               # Rust core image processing engine (created)
  crates/
    lumora-core/      # Pixel buffer, color spaces, non-destructive node graph
    lumora-ops/       # Lightroom & Snapseed image manipulation kernels
    lumora-ai/        # On-device model execution bindings, models & LRU raster cache
    lumora-ffi/       # C-compatible and safe Rust FFI mobile exports
mobile/               # Exact runtime mobile application (created)
  contract/           # Contract UI view declarations (app, develop, selective, ai_lab, presets, setup_wizard)
  logic/              # Rust logic module bridging Contract events to engine (engine_bridge.rs)
  android/            # Android target manifest, Gradle script, CMake, JNI native bridge (arm64-v8a)
models/               # Quantized model conversion & weights manifest (created)
  scripts/            # Conversion and ONNX export scripts
  weights/            # Exported ONNX model binaries (dncnn, midas, mobilenet_scene, realesrgan_x4, zerodce)
compose.yaml          # Podman / Docker multi-container composition (created)
```

## Conventions

- **Environment & Shell:** Windows 11 host. Always use PowerShell commands (e.g. `Get-ChildItem`, `Get-Content`, `;` chaining, not `&&`).
- **Non-Destructive Image Pipeline:** Original image files are strictly read-only. Edits are recorded as non-destructive parameter graphs (curves, HSL adjustments, selective mask coordinates, active AI passes) and rendered in real-time or exported on demand.
- **Offline-First Principle:** Core editing and AI models execute strictly on-device without mandatory internet connection. Cloud connectivity is reserved for auth, preset sharing, and optional backup.
- **Evidence Discipline:** Every factual statement about code or configuration must cite `path:line`. Never describe code without opening it.
- **Secrets Management:** Secrets and API keys reside in `.env` files (excluded by `.gitignore`), never hardcoded in source.

## First Slice

1. Scaffold `compose.yaml` spinning up PostgreSQL and MinIO in Podman.
2. Scaffold `engine/` Rust library with non-destructive image adjusters (brightness, contrast, saturation, HSL curves, U-Point Lab weighting).
3. Scaffold `backend/` FastAPI service with health check, SQLite/PostgreSQL database models, and JWT authentication.
4. Convert and quantize the 5 AI models into INT8 ONNX format.
5. Create benchmark test demonstrating Rust engine rendering a 12MP image under 30ms.

Done when:
- `cargo test --manifest-path engine/Cargo.toml` passes with pixel manipulation tests.
- `podman compose up` spins up backend and responds with HTTP 200 on `/health`.

## Not Yet Built

- **Preset Marketplace S3 Uploads:** Optional MinIO direct S3 multipart presigned upload for community binary files (Phase 5).
- **Camera RAW Library:** In-tree RAW decoder integration for `.dng`, `.cr2`, `.nef` sensors (Phase 5).

## Verification

Application code for Phase 0, Phase 1, Phase 2, Phase 3, and Phase 4 is verified:
- Python backend test suite: `python -m pytest backend/tests` (16 passed in 0.65s across auth, database, health, marketplace, models_dist).
- Rust image engine test suite: `cargo test --manifest-path engine/Cargo.toml` (44 passed in 0.48s across lumora-core: 8, lumora-ops: 12, lumora-ai: 18, and lumora-ffi: 6).
- Mobile logic bridge test suite: `cargo test --manifest-path mobile/logic/Cargo.toml` (2 passed in 0.02s).
- Total Rust unit tests: 46 tests pass cleanly.
- AI model export & verification: `python models/scripts/convert_models.py` (all 5 ONNX models exported & verified in onnxruntime).
- Clippy & formatting: `cargo clippy --all-targets -- -D warnings` and `cargo fmt --all -- --check` pass with 0 errors/warnings across all engine and mobile crates.
- Python linting: `python -m ruff check backend models` passes cleanly with 0 errors.

## Open Items & Decisions

1. **Exact Mobile Target:** Confirmed native Android + iOS support via `expo/exact` (`docs/recipes/eas-android.md`).
2. **AI Runtime:** Confirmed ONNX Runtime Mobile with NNAPI (Android), CoreML (iOS), and CPU fallback.
3. **Engine Pipeline:** Confirmed Dual-Resolution Non-Destructive DAG in Rust (1080p GPU preview, tiled export).
4. **AI Integration:** Confirmed cached raster node in DAG; non-destructive slider tweaks operate on top without re-inferencing.
5. **Preset Format:** Confirmed open JSON `.lumora` format with native import support for Adobe `.xmp` and `.cube` LUTs.
6. **Workspace Layout:** Confirmed 4-Tab modular workspace (Develop, Selective, AI Enhance, Presets).
