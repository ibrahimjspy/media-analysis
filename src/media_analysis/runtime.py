"""Load and warm vendored model runtimes once per worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from media_analysis.config import Settings
from media_analysis.features.ocr import preprocessing_version
from media_analysis.features.ppocr import create_det_session, run_det_inference
from media_analysis.features.silero_vad import (
    load_silero_vad_session,
    warmup_silero_vad,
)
from media_analysis.features.subjects import build_yolox_detector
from media_analysis.features.yunet import load_yunet_detector
from media_analysis.models_manifest import (
    REQUIRED_CPU_MODELS,
    SILERO_VAD_MODEL_NAME,
    ManifestState,
    verify_models,
)
from media_analysis.person_matte.modnet import (
    ReferenceModnetSession,
    infer_modnet_alpha,
    load_modnet_session,
)
from media_analysis.person_matte.readiness import matte_worker_ready_for_serving
from media_analysis.tools.model_lock import ProfileGates, load_lock


@dataclass(frozen=True, slots=True)
class RuntimeState:
    manifest: ManifestState
    ready: bool
    loaded_models: tuple[str, ...]
    warmup_complete: bool
    subject_detector: Any | None = None
    face_detector: Any | None = None
    ocr_session: Any | None = None
    vad_session: Any | None = None
    modnet_session: Any | None = None
    errors: tuple[str, ...] = ()
    reference_mode: bool = False
    production_inference_ready: bool = False
    production_blockers: tuple[str, ...] = ()
    execution_providers: tuple[str, ...] = ()


def _model_path(settings: Settings, manifest: ManifestState, name: str) -> Path:
    entry = manifest.by_name(name)
    if entry is None:
        raise RuntimeError(f"missing manifest entry: {name}")
    return settings.media_analysis_model_dir / entry.file


def _profile_gates(settings: Settings) -> ProfileGates | None:
    candidates = (
        settings.media_analysis_model_dir / "manifest.lock.json",
        Path("models/manifest.lock.json"),
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            lock = load_lock(path, profile=settings.media_analysis_image)
        except ValueError:
            continue
        return lock.gates
    return None


def _required_stub_entries(
    manifest: ManifestState,
    required_models: set[str],
) -> tuple[Any, ...]:
    return tuple(
        entry
        for entry in manifest.entries
        if entry.name in required_models or entry.name == "modnet"
    )


def _silero_production_ready(
    entry: Any | None,
    *,
    gates: ProfileGates | None,
    settings: Settings,
) -> bool:
    if entry is None or entry.stub:
        return False
    if gates is not None and not gates.silero_vad_production_enabled:
        return False
    return not settings.media_analysis_allow_stub_models


def load_runtime(settings: Settings) -> RuntimeState:
    """Verify artifacts, construct sessions, and run deterministic warmups."""
    required_for_role: tuple[str, ...] | None = None
    if settings.worker_role == "general":
        required_for_role = tuple(
            name for name in REQUIRED_CPU_MODELS if name != "PP-OCRv5_mobile_det"
        )
    elif settings.worker_role == "ocr":
        required_for_role = ("PP-OCRv5_mobile_det",)
    manifest = verify_models(
        settings.media_analysis_model_dir,
        image=settings.media_analysis_image,
        required_models=required_for_role,
    )
    gates = _profile_gates(settings)
    if not manifest.ready:
        return RuntimeState(
            manifest=manifest,
            ready=False,
            loaded_models=(),
            warmup_complete=False,
            errors=manifest.errors,
        )

    if settings.media_analysis_image.startswith("matte"):
        modnet_entry = manifest.by_name("modnet")
        serving_ready, serving_errors = matte_worker_ready_for_serving(
            modnet_entry=modnet_entry,
            gates=gates,
        )
        if not serving_ready:
            return RuntimeState(
                manifest=manifest,
                ready=False,
                loaded_models=(),
                warmup_complete=False,
                errors=serving_errors,
            )
        modnet_session = None
        loaded: list[str] = []
        errors: list[str] = []
        if modnet_entry is not None and modnet_entry.stub:
            modnet_session = ReferenceModnetSession()
            loaded.append("modnet")
        else:
            try:
                modnet_session = load_modnet_session(
                    _model_path(settings, manifest, "modnet"),
                    execution_provider=settings.media_analysis_matte_execution_provider,
                    engine_cache_dir=settings.media_analysis_tensorrt_cache_dir,
                )
                infer_modnet_alpha(
                    modnet_session,
                    np.zeros((256, 256, 3), dtype=np.uint8),
                )
                loaded.append("modnet")
            except Exception as exc:
                errors.append(f"modnet load/warmup failed: {type(exc).__name__}")
        production_ready, production_errors = matte_worker_ready_for_serving(
            modnet_entry=modnet_entry,
            gates=gates,
            reference_mode=False,
        )
        execution_providers = (
            tuple(modnet_session.get_providers())
            if modnet_session is not None and hasattr(modnet_session, "get_providers")
            else (("ReferenceExecutionProvider",) if modnet_session is not None else ())
        )
        gpu_ready = any(
            provider in {"TensorrtExecutionProvider", "CUDAExecutionProvider"}
            for provider in execution_providers
        )
        if production_ready and not gpu_ready:
            errors.append("production matte inference requires TensorRT or CUDA")
            production_errors = (
                *production_errors,
                "production matte inference requires TensorRT or CUDA",
            )
        ready = not errors and bool(loaded)
        return RuntimeState(
            manifest=manifest,
            ready=ready,
            loaded_models=tuple(loaded),
            warmup_complete=ready,
            modnet_session=modnet_session,
            errors=tuple(errors),
            reference_mode=not production_ready,
            production_inference_ready=production_ready and gpu_ready and ready,
            production_blockers=production_errors,
            execution_providers=execution_providers,
        )

    required_models = set(REQUIRED_CPU_MODELS)
    if settings.worker_role == "general":
        required_models.discard("PP-OCRv5_mobile_det")
    elif settings.worker_role == "ocr":
        required_models = {"PP-OCRv5_mobile_det"}

    required_stub = _required_stub_entries(manifest, required_models)
    if any(entry.stub for entry in required_stub) and not settings.media_analysis_allow_stub_models:
        return RuntimeState(
            manifest=manifest,
            ready=False,
            loaded_models=(),
            warmup_complete=False,
            errors=("stub model artifacts are forbidden outside tests",),
        )

    if any(entry.stub for entry in required_stub) and settings.media_analysis_allow_stub_models:
        loaded_stub = [name for name in manifest.loaded if name in required_models]
        return RuntimeState(
            manifest=manifest,
            ready=True,
            loaded_models=tuple(loaded_stub),
            warmup_complete=True,
            reference_mode=True,
            production_blockers=("stub model artifacts are not production-ready",),
        )

    if settings.worker_role == "ocr":
        entry = manifest.by_name("PP-OCRv5_mobile_det")
        errors: list[str] = []
        session = None
        try:
            if entry is None:
                raise RuntimeError("OCR model is absent")
            session = create_det_session(_model_path(settings, manifest, entry.name))
            run_det_inference(session, np.zeros((320, 320, 3), dtype=np.uint8))
        except Exception as exc:
            errors.append(
                "PP-OCRv5_mobile_det load/warmup failed: "
                f"{type(exc).__name__} ({preprocessing_version()})"
            )
        owned_ocr_ready = gates is not None and gates.owned_ppocr_export_recorded
        blockers = () if owned_ocr_ready else ("owned PP-OCR export parity gate is closed",)
        ready = not errors and session is not None
        return RuntimeState(
            manifest=manifest,
            ready=ready,
            loaded_models=("PP-OCRv5_mobile_det",) if ready else (),
            warmup_complete=ready,
            ocr_session=session,
            errors=tuple(errors),
            production_inference_ready=ready and owned_ocr_ready,
            production_blockers=blockers,
        )

    loaded = []
    errors: list[str] = []
    subject_detector = None
    face_detector = None
    ocr_session = None
    vad_session = None

    try:
        subject_detector = build_yolox_detector(
            _model_path(settings, manifest, "yolox-tiny")
        )
        subject_detector.detect(np.zeros((416, 416, 3), dtype=np.uint8))
        loaded.append("yolox-tiny")
    except Exception as exc:
        errors.append(f"yolox-tiny load/warmup failed: {type(exc).__name__}")

    try:
        face_detector = load_yunet_detector(_model_path(settings, manifest, "yunet"))
        face_detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
        loaded.append("yunet")
    except Exception as exc:
        errors.append(f"yunet load/warmup failed: {type(exc).__name__}")

    if settings.worker_role != "general":
        try:
            ocr_session = create_det_session(
                _model_path(settings, manifest, "PP-OCRv5_mobile_det")
            )
            run_det_inference(
                ocr_session,
                np.zeros((320, 320, 3), dtype=np.uint8),
            )
            loaded.append("PP-OCRv5_mobile_det")
        except Exception as exc:
            errors.append(
                "PP-OCRv5_mobile_det load/warmup failed: "
                f"{type(exc).__name__} ({preprocessing_version()})"
            )

    vad_entry = manifest.by_name(SILERO_VAD_MODEL_NAME)
    if vad_entry is not None and not vad_entry.stub:
        try:
            vad_session = load_silero_vad_session(
                _model_path(settings, manifest, SILERO_VAD_MODEL_NAME)
            )
            warmup_silero_vad(vad_session)
            loaded.append(SILERO_VAD_MODEL_NAME)
        except Exception as exc:
            errors.append(f"{SILERO_VAD_MODEL_NAME} load/warmup failed: {type(exc).__name__}")

    silero_ready = _silero_production_ready(vad_entry, gates=gates, settings=settings)
    if not silero_ready:
        errors.append(
            "silero-vad production gate is closed or its artifact is not production-ready"
        )
    ready = not errors and required_models.issubset(set(loaded))
    owned_ocr_ready = (
        settings.worker_role == "general"
        or (gates is not None and gates.owned_ppocr_export_recorded)
    )
    production_blockers: list[str] = []
    if not silero_ready:
        production_blockers.append(
            "silero-vad production gate is closed or its artifact is not production-ready"
        )
    if not owned_ocr_ready:
        production_blockers.append("owned PP-OCR export parity gate is closed")
    return RuntimeState(
        manifest=manifest,
        ready=ready,
        loaded_models=tuple(loaded),
        warmup_complete=ready,
        subject_detector=subject_detector,
        face_detector=face_detector,
        ocr_session=ocr_session,
        vad_session=vad_session,
        errors=tuple(errors),
        production_inference_ready=ready and silero_ready and owned_ocr_ready,
        production_blockers=tuple(production_blockers),
    )
