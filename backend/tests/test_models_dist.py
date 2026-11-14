"""Tests for Model Distribution service (manifest & downloads)."""

import hashlib


def test_models_manifest_endpoint(client):
    res = client.get("/api/v1/models/manifest")
    assert res.status_code == 200
    manifest = res.json()
    assert manifest["manifest_version"] == "1.0.0"
    assert manifest["total_models"] == 5
    assert manifest["total_size_bytes"] > 0
    assert len(manifest["models"]) == 5

    model_ids = {m["id"] for m in manifest["models"]}
    expected_ids = {"dncnn", "midas", "mobilenet_scene", "realesrgan_x4", "zerodce"}
    assert model_ids == expected_ids

    for m in manifest["models"]:
        assert m["size_bytes"] > 0
        assert len(m["sha256"]) == 64  # valid SHA-256 hex string
        assert m["download_url"].startswith("/api/v1/models/download/")
        assert "NNAPI" in m["execution_providers"]
        assert "CoreML" in m["execution_providers"]
        assert len(m["input_shape"]) == 4


def test_model_download_stream_and_checksum_verification(client):
    # Test downloading dncnn
    res = client.get("/api/v1/models/download/dncnn")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/octet-stream"
    assert "attachment; filename=\"dncnn.onnx\"" in res.headers["content-disposition"]
    assert "x-checksum-sha256" in res.headers

    content = res.content
    assert len(content) > 0
    computed_sha256 = hashlib.sha256(content).hexdigest()
    assert computed_sha256 == res.headers["x-checksum-sha256"]

    # Test downloading with alias and extension: realesrgan
    alias_res = client.get("/api/v1/models/download/realesrgan")
    assert alias_res.status_code == 200
    assert alias_res.headers["x-model-id"] == "realesrgan_x4"

    # Test downloading zerodce
    zero_res = client.get("/api/v1/models/download/zerodce.onnx")
    assert zero_res.status_code == 200
    assert zero_res.headers["x-model-id"] == "zerodce"

    # Test requesting signed S3 distribution URL
    signed_res = client.get("/api/v1/models/download/dncnn?signed=true")
    assert signed_res.status_code == 200
    signed_data = signed_res.json()
    assert signed_data["format"] == "s3_signed"
    assert "download_url" in signed_data
    assert "dncnn.onnx" in signed_data["download_url"]


def test_unknown_model_download_404(client):
    res = client.get("/api/v1/models/download/super_fake_model")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()
