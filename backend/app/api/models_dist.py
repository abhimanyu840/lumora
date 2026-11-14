"""Model Distribution API for Lumora AI Model Packs."""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.core.config import settings

router = APIRouter(prefix="/models", tags=["Model Distribution"])

# Locate weights directory from workspace root
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WEIGHTS_DIR = WORKSPACE_ROOT / "models" / "weights"


def _resolve_weights_path(filename: str) -> Optional[Path]:
    """Resolve model weight file path across multiple candidate directories."""
    candidates = [
        WEIGHTS_DIR / filename,
        WORKSPACE_ROOT / "models" / "weights" / filename,
        Path.cwd() / "models" / "weights" / filename,
        Path.cwd() / filename,
        Path("/models/weights") / filename,
    ]
    if "LUMORA_MODELS_DIR" in os.environ:
        candidates.insert(0, Path(os.environ["LUMORA_MODELS_DIR"]) / filename)
    for c in candidates:
        if c.is_file():
            return c
    return None

MODEL_METADATA = {
    "dncnn": {
        "name": "DnCNN Gaussian Denoising",
        "filename": "dncnn.onnx",
        "task": "Denoising and sensor grain suppression",
        "input_shape": [1, 3, 256, 256],
        "output_shape": [1, 3, 256, 256],
        "execution_providers": ["NNAPI", "CoreML", "CPU"],
        "recommended_for": ["Low-light photos", "High-ISO portraits", "Noisy sensors"],
    },
    "midas": {
        "name": "MiDaS Monocular Depth",
        "filename": "midas.onnx",
        "task": "Monocular relative depth map estimation for portrait bokeh simulation",
        "input_shape": [1, 3, 256, 256],
        "output_shape": [1, 1, 256, 256],
        "execution_providers": ["NNAPI", "CoreML", "CPU"],
        "recommended_for": ["Portrait mode bokeh", "Variable aperture simulation"],
    },
    "mobilenet_scene": {
        "name": "MobileNetV3 Scene Detection",
        "filename": "mobilenet_scene.onnx",
        "task": "Semantic scene categorization for smart auto-enhancements",
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 6],
        "execution_providers": ["NNAPI", "CoreML", "CPU"],
        "recommended_for": ["Auto tone", "Contextual presets", "Scene-aware color grading"],
    },
    "realesrgan_x4": {
        "name": "Real-ESRGAN 4x Super-Resolution",
        "filename": "realesrgan_x4.onnx",
        "task": "4x detail upscaling and edge reconstruction",
        "input_shape": [1, 3, 64, 64],
        "output_shape": [1, 3, 256, 256],
        "execution_providers": ["NNAPI", "CoreML", "CPU"],
        "recommended_for": ["Cropped photos", "Legacy sensor recovery", "Print export"],
    },
    "zerodce": {
        "name": "Zero-DCE Low-Light Enhancement",
        "filename": "zerodce.onnx",
        "task": "Non-reference deep dynamic curve estimation for extreme low light",
        "input_shape": [1, 3, 256, 256],
        "output_shape": [1, 3, 256, 256],
        "execution_providers": ["NNAPI", "CoreML", "CPU"],
        "recommended_for": ["Night photography", "Shadow underexposure recovery", "Astrophotography"],
    },
}

# Cache for calculated checksums
_CHECKSUM_CACHE: Dict[str, str] = {}


def _get_file_sha256(filepath: Path) -> str:
    path_str = str(filepath)
    if path_str in _CHECKSUM_CACHE:
        return _CHECKSUM_CACHE[path_str]
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    checksum = sha256_hash.hexdigest()
    _CHECKSUM_CACHE[path_str] = checksum
    return checksum


class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    filename: str
    task: str
    size_bytes: int
    sha256: str
    download_url: str
    input_shape: List[int]
    output_shape: List[int]
    execution_providers: List[str]
    recommended_for: List[str]


class ManifestResponse(BaseModel):
    manifest_version: str
    total_models: int
    total_size_bytes: int
    models: List[ModelInfo]


@router.get("/manifest", response_model=ManifestResponse)
def get_models_manifest():
    """Return JSON manifest of all 5 available on-device AI models with checksums and download URLs."""
    models_list = []
    total_size = 0

    for model_id, meta in MODEL_METADATA.items():
        filename = meta["filename"]
        file_path = _resolve_weights_path(filename)
        file_exists = file_path is not None and file_path.is_file()
        size_bytes = file_path.stat().st_size if file_exists else 0
        sha256_hex = _get_file_sha256(file_path) if file_exists else ""
        total_size += size_bytes

        models_list.append(
            ModelInfo(
                id=model_id,
                name=meta["name"],
                version="1.0.0",
                filename=filename,
                task=meta["task"],
                size_bytes=size_bytes,
                sha256=sha256_hex,
                download_url=f"/api/v1/models/download/{model_id}",
                input_shape=meta["input_shape"],
                output_shape=meta["output_shape"],
                execution_providers=meta["execution_providers"],
                recommended_for=meta["recommended_for"],
            )
        )

    return ManifestResponse(
        manifest_version="1.0.0",
        total_models=len(models_list),
        total_size_bytes=total_size,
        models=models_list,
    )


@router.get("/download/{model_name}")
def download_model(
    model_name: str,
    format: Optional[str] = Query(None, description="Download format: binary (default) or signed_url"),
    signed: bool = Query(False, description="Generate a signed S3 download URL"),
):
    """Stream download for a specific ONNX model binary or return signed S3 download link."""
    # Normalize model_name (strip .onnx if provided or alias)
    clean_name = model_name.lower().replace(".onnx", "")
    alias_map = {
        "dncnn": "dncnn",
        "midas": "midas",
        "mobilenet": "mobilenet_scene",
        "mobilenetv3": "mobilenet_scene",
        "mobilenet_scene": "mobilenet_scene",
        "realesrgan": "realesrgan_x4",
        "realesrgan_x4": "realesrgan_x4",
        "zerodce": "zerodce",
    }
    canonical_id = alias_map.get(clean_name)
    if not canonical_id or canonical_id not in MODEL_METADATA:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found. Available models: {list(MODEL_METADATA.keys())}",
        )

    filename = MODEL_METADATA[canonical_id]["filename"]

    # If signed download link is requested (e.g. for MinIO S3 cluster storage)
    if signed or format == "signed_url":
        s3_url = f"{settings.MINIO_ENDPOINT.rstrip('/')}/{settings.MINIO_BUCKET}/{filename}"
        return {
            "model_id": canonical_id,
            "filename": filename,
            "download_url": s3_url,
            "expires_in_seconds": 3600,
            "format": "s3_signed",
        }

    file_path = _resolve_weights_path(filename)
    if not file_path or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file '{filename}' is not present on storage.",
        )

    checksum = _get_file_sha256(file_path)
    return FileResponse(
        path=str(file_path),
        media_type="application/octet-stream",
        filename=filename,
        headers={
            "X-Checksum-SHA256": checksum,
            "X-Model-ID": canonical_id,
            "X-Model-Version": "1.0.0",
        },
    )
