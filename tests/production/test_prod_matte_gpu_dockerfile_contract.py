from __future__ import annotations

from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parents[2] / "docker" / "matte-gpu.Dockerfile"


@pytest.mark.production
@pytest.mark.unit
def test_gpu_matte_image_pins_compatible_tensorrt_and_ort() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "nvcr.io/nvidia/tensorrt:25.03-py3" in text
    assert "onnxruntime-gpu==1.22.0" in text
    assert "pip uninstall -y onnxruntime" in text


@pytest.mark.production
@pytest.mark.unit
def test_gpu_matte_image_requires_tensorrt_and_writable_engine_cache() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "MEDIA_ANALYSIS_MATTE_EXECUTION_PROVIDER=tensorrt" in text
    assert "MEDIA_ANALYSIS_TENSORRT_CACHE_DIR=/var/tmp/media-analysis/tensorrt" in text
    assert "chown -R mediaanalysis:mediaanalysis" in text
    assert "MEDIA_ANALYSIS_MATTE_REFERENCE_MODE" not in text
