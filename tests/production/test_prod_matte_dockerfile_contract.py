from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "matte-cpu.Dockerfile"
ENTRYPOINT = REPO_ROOT / "docker" / "matte-entrypoint.sh"


@pytest.mark.production
@pytest.mark.unit
def test_prod_matte_dockerfile_targets_python312_slim() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "python:3.12-slim-bookworm@sha256:" in text
    assert "MEDIA_ANALYSIS_IMAGE=matte-cpu" in text


@pytest.mark.production
@pytest.mark.unit
def test_prod_matte_dockerfile_vendors_matte_cpu_profile_only() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "--profile matte-cpu" in text
    assert "vendor_models" in text
    assert "analysis-cpu" not in text


@pytest.mark.production
@pytest.mark.unit
def test_prod_matte_dockerfile_has_no_v1_weights() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "yolox" not in text
    assert "yunet" not in text
    assert "PP-OCR" not in text


@pytest.mark.production
@pytest.mark.unit
def test_prod_matte_dockerfile_runs_non_root_one_worker() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "USER mediaanalysis" in text
    assert "MEDIA_ANALYSIS_MATTE_REFERENCE_MODE=1" in text
    entry = ENTRYPOINT.read_text(encoding="utf-8")
    assert "--workers 1" in entry
    assert "uvicorn" in entry


@pytest.mark.production
@pytest.mark.unit
def test_prod_matte_entrypoint_does_not_vendor_at_runtime() -> None:
    entry = ENTRYPOINT.read_text(encoding="utf-8")
    assert "vendor_models" not in entry
