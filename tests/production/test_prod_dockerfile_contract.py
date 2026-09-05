from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "analysis-cpu.Dockerfile"
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


@pytest.mark.production
@pytest.mark.unit
def test_prod_dockerfile_targets_amd64_python312_slim() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "linux/amd64" in text
    assert "python:3.12-slim" in text
    assert "python:3.12-slim-bookworm@sha256:" in text


@pytest.mark.production
@pytest.mark.unit
def test_prod_dockerfile_installs_ffmpeg_and_vendors_at_build() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "ffmpeg" in text
    assert "vendor_models" in text
    assert "MEDIA_ANALYSIS_MODEL_DIR=/models" in text
    assert "MEDIA_ANALYSIS_IMAGE=analysis-cpu" in text
    assert "MEDIA_ANALYSIS_WORKER_ROLE=general" in text
    assert "vendor_models" not in ENTRYPOINT.read_text(encoding="utf-8")


@pytest.mark.production
@pytest.mark.unit
def test_prod_dockerfile_uses_non_editable_install() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "pip install" in text
    assert "-e ." not in text
    assert "pip install --constraint docker/constraints.txt ." in text


@pytest.mark.production
@pytest.mark.unit
def test_prod_dockerfile_runs_as_non_root_with_bounded_temp() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "USER mediaanalysis" in text
    assert "MEDIA_ANALYSIS_TMPDIR=/var/tmp/media-analysis" in text
    assert "TMPDIR=/var/tmp/media-analysis" in text
    assert "/var/tmp/media-analysis" in text


@pytest.mark.production
@pytest.mark.unit
def test_prod_dockerfile_models_and_app_readable_by_runtime_user() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "chown -R mediaanalysis:mediaanalysis /app /models" in text
    assert "chmod -R a+rX /app /models" in text


@pytest.mark.production
@pytest.mark.unit
def test_prod_dockerfile_documents_interim_ppocr_export() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "MEDIA_ANALYSIS_PP_OCR_EXPORT=community-interim" in text


@pytest.mark.production
@pytest.mark.unit
def test_prod_entrypoint_runs_single_uvicorn_worker() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "uvicorn" in text
    assert "--workers 1" in text


@pytest.mark.production
@pytest.mark.unit
def test_prod_build_context_excludes_secrets_and_local_artifacts() -> None:
    patterns = set(DOCKERIGNORE.read_text(encoding="utf-8").splitlines())
    assert {".env", ".git", ".venv", "tests", "models/*.onnx"} <= patterns
