from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tests.conftest import write_stub_models

from media_analysis.analyze import _resolve_vad_session
from media_analysis.config import Settings
from media_analysis.models_manifest import ManifestState, ModelEntry
from media_analysis.runtime import RuntimeState, load_runtime
from media_analysis.tools.model_lock import ProfileGates


def _mark_models_real(model_dir: Path) -> None:
    path = model_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for model in manifest["models"]:
        model["stub"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.unit
def test_stub_artifacts_are_forbidden_by_default(tmp_path: Path) -> None:
    write_stub_models(tmp_path)
    state = load_runtime(
        Settings(
            media_analysis_model_dir=tmp_path,
            media_analysis_allow_stub_models=False,
        )
    )
    assert state.ready is False
    assert state.loaded_models == ()
    assert "forbidden" in state.errors[0]


@pytest.mark.unit
def test_stub_artifacts_are_explicitly_allowed_for_tests(tmp_path: Path) -> None:
    write_stub_models(tmp_path)
    state = load_runtime(
        Settings(
            media_analysis_model_dir=tmp_path,
            media_analysis_allow_stub_models=True,
        )
    )
    assert state.ready is True
    assert state.warmup_complete is True
    assert state.loaded_models == (
        "yolox-tiny",
        "yunet",
        "PP-OCRv5_mobile_det",
        "silero-vad",
    )
    assert state.subject_detector is None


@pytest.mark.unit
def test_worker_roles_only_publish_their_own_loaded_models(tmp_path: Path) -> None:
    write_stub_models(tmp_path)
    general = load_runtime(
        Settings(
            media_analysis_model_dir=tmp_path,
            media_analysis_allow_stub_models=True,
            media_analysis_worker_role="general",
        )
    )
    ocr = load_runtime(
        Settings(
            media_analysis_model_dir=tmp_path,
            media_analysis_allow_stub_models=True,
            media_analysis_worker_role="ocr",
        )
    )
    assert general.loaded_models == ("yolox-tiny", "yunet", "silero-vad")
    assert ocr.loaded_models == ("PP-OCRv5_mobile_det",)


@pytest.mark.unit
def test_general_worker_does_not_require_ocr_artifact(tmp_path: Path) -> None:
    write_stub_models(tmp_path)
    (tmp_path / "PP-OCRv5_mobile_det.onnx").unlink()
    state = load_runtime(
        Settings(
            media_analysis_model_dir=tmp_path,
            media_analysis_allow_stub_models=True,
            media_analysis_worker_role="general",
        )
    )
    assert state.ready is True
    assert "PP-OCRv5_mobile_det" not in state.loaded_models


@pytest.mark.unit
def test_checksum_valid_but_unloadable_model_is_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_stub_models(tmp_path)
    _mark_models_real(tmp_path)

    def fail_subject(_path: Path):
        raise RuntimeError("bad graph")

    monkeypatch.setattr("media_analysis.runtime.build_yolox_detector", fail_subject)
    state = load_runtime(Settings(media_analysis_model_dir=tmp_path))

    assert state.ready is False
    assert state.warmup_complete is False
    assert any("yolox-tiny load/warmup failed" in item for item in state.errors)


@pytest.mark.unit
def test_stub_mode_allows_fake_silero_without_runtime_session(tmp_path: Path) -> None:
    write_stub_models(tmp_path)
    state = load_runtime(
        Settings(
            media_analysis_model_dir=tmp_path,
            media_analysis_allow_stub_models=True,
        )
    )
    assert state.ready is True
    assert "silero-vad" in state.loaded_models
    assert state.vad_session is None


@pytest.mark.unit
def test_allow_stub_setting_does_not_replace_a_real_vad_session() -> None:
    entry = ModelEntry(
        name="silero-vad",
        file="silero_vad.onnx",
        sha256="0" * 64,
        license="MIT",
        stub=False,
        version="6.2",
    )
    manifest = ManifestState(
        entries=(entry,),
        ready=True,
        loaded=("silero-vad",),
        errors=(),
    )
    session = object()
    runtime = RuntimeState(
        manifest=manifest,
        ready=True,
        loaded_models=("silero-vad",),
        warmup_complete=True,
        vad_session=session,
    )
    resolved = _resolve_vad_session(
        runtime,
        manifest,
        Settings(media_analysis_allow_stub_models=True),
    )
    assert resolved is session


@pytest.mark.unit
def test_closed_silero_production_gate_prevents_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_stub_models(tmp_path)
    _mark_models_real(tmp_path)

    class FakeDetector:
        def detect(self, _image: np.ndarray):
            return []

    monkeypatch.setattr("media_analysis.runtime.build_yolox_detector", lambda _path: FakeDetector())
    monkeypatch.setattr("media_analysis.runtime.load_yunet_detector", lambda _path: FakeDetector())
    monkeypatch.setattr("media_analysis.runtime.create_det_session", lambda _path: object())
    monkeypatch.setattr("media_analysis.runtime.run_det_inference", lambda _session, _image: [])
    monkeypatch.setattr("media_analysis.runtime.load_silero_vad_session", lambda _path: object())
    monkeypatch.setattr("media_analysis.runtime.warmup_silero_vad", lambda _session: None)
    monkeypatch.setattr(
        "media_analysis.runtime._profile_gates",
        lambda _settings: ProfileGates(
            owned_ppocr_export_recorded=False,
            silero_vad_production_enabled=False,
        ),
    )

    state = load_runtime(Settings(media_analysis_model_dir=tmp_path))

    assert state.ready is False
    assert state.production_inference_ready is False
    assert any("silero-vad production gate is closed" in item for item in state.errors)


@pytest.mark.unit
def test_all_real_runtimes_must_load_and_warm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_stub_models(tmp_path)
    _mark_models_real(tmp_path)

    class FakeDetector:
        def detect(self, _image: np.ndarray):
            return []

    class FakeOcrSession:
        pass

    monkeypatch.setattr(
        "media_analysis.runtime.build_yolox_detector",
        lambda _path: FakeDetector(),
    )
    monkeypatch.setattr(
        "media_analysis.runtime.load_yunet_detector",
        lambda _path: FakeDetector(),
    )
    monkeypatch.setattr(
        "media_analysis.runtime.create_det_session",
        lambda _path: FakeOcrSession(),
    )
    monkeypatch.setattr(
        "media_analysis.runtime.run_det_inference",
        lambda _session, _image: [],
    )
    vad_session = object()
    monkeypatch.setattr(
        "media_analysis.runtime.load_silero_vad_session",
        lambda _path: vad_session,
    )
    monkeypatch.setattr(
        "media_analysis.runtime.warmup_silero_vad",
        lambda _session: None,
    )
    monkeypatch.setattr(
        "media_analysis.runtime._profile_gates",
        lambda _settings: ProfileGates(
            owned_ppocr_export_recorded=True,
            silero_vad_production_enabled=True,
        ),
    )

    state = load_runtime(Settings(media_analysis_model_dir=tmp_path))
    assert state.ready is True
    assert state.warmup_complete is True
    assert state.loaded_models == (
        "yolox-tiny",
        "yunet",
        "PP-OCRv5_mobile_det",
        "silero-vad",
    )
    assert state.subject_detector is not None
    assert state.face_detector is not None
    assert state.ocr_session is not None
    assert state.vad_session is vad_session
    assert state.production_inference_ready is True
