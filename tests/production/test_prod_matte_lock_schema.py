from __future__ import annotations

import json
from pathlib import Path

import pytest

from media_analysis.models_manifest import ModelEntry, verify_models
from media_analysis.person_matte.readiness import (
    matte_reference_mode_enabled,
    matte_worker_ready_for_production,
    matte_worker_ready_for_serving,
)
from media_analysis.tools.model_lock import load_lock

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "models" / "manifest.lock.json"
DOCKERFILE = REPO_ROOT / "docker" / "matte-cpu.Dockerfile"


@pytest.mark.production
@pytest.mark.unit
def test_prod_matte_cpu_lock_has_modnet_stub_only() -> None:
    lock = load_lock(LOCK_PATH, profile="matte-cpu")
    names = {item.name for item in lock.models}
    assert names == {"modnet"}
    assert lock.gates is not None
    assert lock.gates.matte_production_enabled is False
    assert lock.gates.benchmark_gate_passed is False


@pytest.mark.production
@pytest.mark.unit
def test_prod_matte_stub_never_production_ready() -> None:
    lock = load_lock(LOCK_PATH, profile="matte-cpu")
    entry = lock.models[0]
    assert entry.stub is True
    ready, errors = matte_worker_ready_for_production(
        modnet_entry=ModelEntry(
            name=entry.name,
            file=entry.file,
            sha256=entry.sha256,
            license=entry.license,
            stub=True,
            version=entry.revision,
        ),
        gates=lock.gates,
    )
    assert ready is False
    assert errors


@pytest.mark.production
@pytest.mark.unit
def test_prod_matte_docker_sets_reference_mode_explicitly() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "MEDIA_ANALYSIS_MATTE_REFERENCE_MODE=1" in text


@pytest.mark.production
@pytest.mark.unit
def test_prod_matte_manifest_verify_does_not_imply_production_ready(tmp_path: Path) -> None:
    lock = load_lock(LOCK_PATH, profile="matte-cpu")
    entry = lock.models[0]
    (tmp_path / entry.file).write_bytes(b"stub-modnet")
    manifest = {
        "models": [
            {
                "name": entry.name,
                "file": entry.file,
                "sha256": entry.sha256,
                "license": entry.license,
                "version": entry.revision,
                "stub": True,
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    state = verify_models(tmp_path, image="matte-cpu")
    assert state.ready
    serving, _ = matte_worker_ready_for_serving(
        modnet_entry=state.by_name("modnet"),
        gates=lock.gates,
        reference_mode=matte_reference_mode_enabled("0"),
    )
    assert serving is False
