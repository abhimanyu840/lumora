# Lumora

Lumora is a cross-platform mobile photo editing suite combining the precision tools of Adobe Lightroom (HSL, curves, color grading, non-destructive RAW develop pipeline) and Snapseed (selective U-Point control points, healing, structure, drama, perspective correction) with local on-device AI enhancements (DnCNN denoising, MiDaS depth/bokeh, MobileNet scene detection, Real-ESRGAN super-resolution, and Zero-DCE low-light enhancement).

## Key Features

- **Non-Destructive DAG Image Engine:** Written in Rust, supporting real-time 60 FPS previews and tiled high-resolution exports.
- **Parametric Tone Curves:** Cubic spline tone curves matching Adobe DNG spline specifications (Master RGB, Red, Green, Blue channels).
- **8-Channel HSL Mixer:** Fine-grained hue, saturation, and luminance shifts across 8 discrete color channels.
- **3-Way Color Grading Wheels:** Cinematic Shadows, Midtones, and Highlights split-toning with balance and blending controls.
- **Snapseed U-Point Technology:** Selective editing using Euclidean distance falloff in CIE-Lab color space.
- **Local On-Device AI Models:** Quantized INT8 execution via ONNX Runtime Mobile (no cloud reliance for inference).
- **Cloud Preset Marketplace:** FastAPI backend with PostgreSQL and MinIO S3 object storage for preset sharing and model pack distribution.

## Architecture & Layout

```
lumora/
├── backend/            # FastAPI service (Auth, Preset Marketplace, Model Distribution)
│   ├── app/            # API endpoints, database models, and schemas
│   ├── tests/          # Pytest test suite
│   ├── Dockerfile      # Container specification
│   └── requirements.txt
├── engine/             # Rust core image processing engine
│   ├── crates/
│   │   ├── lumora-core/ # Pixel buffers, color conversions (sRGB, Lab, XYZ), DAG pipeline
│   │   └── lumora-ops/  # Tone curves, HSL mixer, 3-way color grading, U-Point masks
│   └── Cargo.toml
├── compose.yaml        # Podman / Docker multi-container composition
├── README.md           # Project documentation
└── CONTRIBUTING.md     # Development & contribution guidelines
```

## Getting Started

### Prerequisites

- **Rust:** `cargo 1.80+` (recommended 1.94+)
- **Python:** `3.11+`
- **Podman / Docker:** for containerized services

### Running the Rust Image Engine

```powershell
# Run unit and benchmark tests
cargo test --manifest-path engine/Cargo.toml
```

### Running Backend Locally

```powershell
# Run backend tests
python -m pytest backend/tests -v

# Run local development server
uvicorn backend.app.main:app --reload --port 8000
```

### Running Container Services (Podman / Docker)

```powershell
podman compose -f compose.yaml up -d
# Check health
curl http://localhost:8000/health
```

## License

Apache-2.0
