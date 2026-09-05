from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any

from media_analysis import __version__
from media_analysis.build_provenance import analyze_build_provenance
from media_analysis.cache import LocalMediaCache, get_local_media_cache
from media_analysis.config import IMPLEMENTED_CPU, MATTE_FEATURES, Settings
from media_analysis.decode import (
    DECODE_PIPELINE_VERSION,
    canonicalize,
    decoded_pixel_count,
    enforce_limits,
    needs_canonical_audio,
    probe,
)
from media_analysis.errors import (
    CANCELLED,
    FEATURE_UNAVAILABLE,
    INVALID_REQUEST,
    LIMIT_EXCEEDED,
    MODEL_NOT_READY,
    TIMEOUT,
    AnalyzeError,
)
from media_analysis.features.audio import (
    analyze_audio,
    audio_capability,
    audio_provenance,
)
from media_analysis.features.audio_pcm import PCM_EXTRACTION_VERSION, extract_analysis_pcm
from media_analysis.features.exposure import (
    EXPOSURE_ANALYZER_VERSION,
    analyze_exposure,
    exposure_policy_metadata,
)
from media_analysis.features.faces import (
    FACE_ASSOCIATION_VERSION,
    FACE_TRACKER_VERSION,
    YUNET_CAPABILITY_VERSION,
    analyze_faces,
    empty_face_analysis,
)
from media_analysis.features.motion import (
    MOTION_ANALYZER_VERSION,
    SubjectBoxesByFrame,
    analyze_motion,
)
from media_analysis.features.ocr import (
    OCR_DETECTOR_VERSION,
    OCR_MERGE_VERSION,
    OCR_SAMPLER_VERSION,
    analyze_ocr,
    preprocessing_version,
)
from media_analysis.features.quality import (
    QUALITY_POLICY_VERSION,
    analyze_quality,
    make_internal_motion_estimator,
)
from media_analysis.features.shots import (
    SHOT_ANALYZER_VERSION,
    SHOT_CLASSIFIER_VERSION,
    analyze_shots,
    classify_shots_with_subjects,
)
from media_analysis.features.silero_vad import FakeSileroVadSession
from media_analysis.features.subjects import (
    SAMPLING_POLICY_VERSION,
    TRACKER_VERSION,
    analyze_subjects,
    subject_model_provenance,
)
from media_analysis.features.thumbnails import (
    THUMBNAIL_ANALYZER_VERSION,
    THUMBNAIL_MIME_TYPE,
    ThumbnailGrant,
    analyze_thumbnails,
    thumbnail_policy_metadata,
)
from media_analysis.features.visual import model_provenance
from media_analysis.features.waveform import (
    WAVEFORM_ANALYZER_VERSION,
    analyze_waveform,
    waveform_capability,
)
from media_analysis.features.yunet import YUNET_PREPROCESSING_VERSION
from media_analysis.frame_access import BoundedFrameAccess, FrameAccessConfig
from media_analysis.frames import Rational, analysis_transform, duration_sec
from media_analysis.jobs import Job
from media_analysis.models_manifest import SILERO_VAD_MODEL_NAME
from media_analysis.person_matte.constants import MATTE_TEMPORAL_POLICY_VERSION
from media_analysis.person_matte.pipeline import run_person_matte_stage1
from media_analysis.person_matte.provenance import matte_provenance_block
from media_analysis.person_matte.types import CanonicalMediaSpec
from media_analysis.request_hash import request_hash
from media_analysis.runtime import RuntimeState
from media_analysis.schemas import AnalyzeRequest
from media_analysis.source import assert_fetch_url, assert_source_fresh, fetch_source_to_path
from media_analysis.telemetry import JobTelemetry
from media_analysis.tools.model_lock import load_lock
from media_analysis.upload import upload_artifact

SHOT_TIMELINE_FEATURES = frozenset(
    {"shots", "subjects", "faces", "ocr", "quality", "exposure", "thumbnails"}
)
VISUAL_SHARED = frozenset({"motion", "quality", "exposure", "thumbnails"})
AUDIO_FEATURES = frozenset({"audio", "waveform"})
FILL_ONLY_VISUAL = frozenset({"exposure", "thumbnails"})


def _check_job(job: Job, deadline: float) -> None:
    if job.cancelled():
        raise AnalyzeError(CANCELLED, "Analysis cancelled")
    if time.monotonic() >= deadline:
        raise AnalyzeError(TIMEOUT, "Analysis exceeded MEDIA_ANALYSIS_JOB_TIMEOUT_SEC")


def _check_stage(job: Job, job_deadline: float, stage_deadline: float, stage: str) -> None:
    _check_job(job, job_deadline)
    if time.monotonic() >= stage_deadline:
        raise AnalyzeError(
            TIMEOUT,
            f"Stage {stage} exceeded MEDIA_ANALYSIS_STAGE_TIMEOUT_SEC",
        )


class StageDeadline:
    """Per-stage cancel guard. Callable so existing cancel_check=stage still works."""

    def __init__(
        self,
        name: str,
        job: Job,
        job_deadline: float,
        stage_deadline: float,
    ) -> None:
        self.name = name
        self.job = job
        self.job_deadline = job_deadline
        self.stage_deadline = stage_deadline

    def check(self) -> None:
        _check_stage(self.job, self.job_deadline, self.stage_deadline, self.name)

    def __call__(self) -> None:
        self.check()

    def remaining(self) -> float:
        self.check()
        remaining = min(self.job_deadline, self.stage_deadline) - time.monotonic()
        if remaining <= 0:
            raise AnalyzeError(
                TIMEOUT,
                f"Stage {self.name} exceeded MEDIA_ANALYSIS_STAGE_TIMEOUT_SEC",
            )
        return remaining


@contextmanager
def _feature_stage(
    name: str,
    telemetry: JobTelemetry,
    job: Job,
    job_deadline: float,
    budget_sec: float,
) -> Iterator[StageDeadline]:
    if budget_sec <= 0:
        raise AnalyzeError(
            TIMEOUT,
            f"Stage {name} exceeded MEDIA_ANALYSIS_STAGE_TIMEOUT_SEC",
        )
    stage = StageDeadline(
        name,
        job,
        job_deadline,
        min(job_deadline, time.monotonic() + budget_sec),
    )
    stage.check()
    with telemetry.stage(name):
        yield stage
        stage.check()


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AnalyzeError(TIMEOUT, "Analysis exceeded MEDIA_ANALYSIS_JOB_TIMEOUT_SEC")
    return remaining


def validate_features(features: list[str], settings: Settings) -> None:
    if settings.media_analysis_image == "analysis-cpu" and set(features) & MATTE_FEATURES:
        raise AnalyzeError(INVALID_REQUEST, "person_matte is not served by analysis-cpu")
    if settings.media_analysis_image.startswith("matte"):
        unsupported = set(features) - MATTE_FEATURES
        if unsupported:
            raise AnalyzeError(
                FEATURE_UNAVAILABLE,
                "matte image does not substitute for analysis-cpu features",
            )
    if settings.worker_role == "general" and "ocr" in features:
        raise AnalyzeError(
            FEATURE_UNAVAILABLE,
            "ocr is served by a dedicated OCR worker",
        )
    if settings.worker_role == "ocr":
        unsupported = set(features) - {"ocr", "shots"}
        if unsupported:
            raise AnalyzeError(
                FEATURE_UNAVAILABLE,
                "OCR worker only serves ocr and shots",
            )
    unimplemented = [
        name for name in features if name not in IMPLEMENTED_CPU and name not in MATTE_FEATURES
    ]
    if unimplemented:
        joined = ", ".join(unimplemented)
        raise AnalyzeError(FEATURE_UNAVAILABLE, f"feature not implemented yet: {joined}")


def validate_analyze_request(request: AnalyzeRequest, settings: Settings) -> None:
    validate_features(request.features, settings)
    features = set(request.features)

    if "thumbnails" in features:
        if request.outputGrants is None or not request.outputGrants.thumbnails:
            raise AnalyzeError(
                INVALID_REQUEST,
                "outputGrants.thumbnails is required for thumbnails",
            )

    if "person_matte" in features:
        if request.matteTarget is None:
            raise AnalyzeError(INVALID_REQUEST, "matteTarget is required for person_matte")
        if request.matteTarget.mode != "all_people":
            raise AnalyzeError(
                INVALID_REQUEST,
                "matteTarget.mode subject_tracks is Stage 2 and is not implemented",
            )
        if request.matteFrameRanges is not None:
            raise AnalyzeError(
                INVALID_REQUEST,
                "segmented matte delivery is Stage 2 and is not implemented",
            )
        if (
            request.outputGrants is None
            or not request.outputGrants.matteAssets
            or len(request.outputGrants.matteAssets) != 1
        ):
            raise AnalyzeError(
                INVALID_REQUEST,
                "Stage 1 person_matte requires exactly one matteAssets grant",
            )
        if request.priorFacts is None or not request.priorFacts.shots:
            raise AnalyzeError(INVALID_REQUEST, "priorFacts.shots is required for person_matte")

    if request.priorFacts is not None and "shots" not in features:
        if features & FILL_ONLY_VISUAL and not request.priorFacts.shots:
            raise AnalyzeError(
                INVALID_REQUEST,
                "priorFacts.shots is required for fill-only exposure/thumbnail analysis",
            )


def _prior_shot_dicts(request: AnalyzeRequest) -> list[dict[str, Any]] | None:
    if request.priorFacts is None or not request.priorFacts.shots:
        return None
    return [
        shot.model_dump(mode="python", by_alias=True) for shot in request.priorFacts.shots
    ]


def _subject_boxes_by_frame(
    subjects: list[dict[str, Any]] | None,
) -> SubjectBoxesByFrame | None:
    if not subjects:
        return None
    boxes: SubjectBoxesByFrame = {}
    for track in subjects:
        for sample in track.get("samples", []):
            frame = int(sample["sourceFrame"])
            raw_box = sample.get("box")
            if raw_box is None:
                continue
            if hasattr(raw_box, "model_dump"):
                raw_box = raw_box.model_dump()
            boxes.setdefault(frame, []).append(dict(raw_box))
    return boxes or None


def _resolve_vad_session(
    runtime: RuntimeState,
    manifest: Any,
    settings: Settings,
) -> FakeSileroVadSession | Any | None:
    entry = manifest.by_name(SILERO_VAD_MODEL_NAME)
    if entry is None:
        return None
    if entry.stub:
        if not settings.media_analysis_allow_stub_models:
            return None
        return FakeSileroVadSession(constant=0.05)
    return runtime.vad_session


def _audio_vad_available(
    runtime: RuntimeState,
    manifest: Any,
    settings: Settings,
) -> bool:
    return _resolve_vad_session(runtime, manifest, settings) is not None


def _thumbnail_grants(request: AnalyzeRequest) -> list[dict[str, Any]] | None:
    if request.outputGrants is None or not request.outputGrants.thumbnails:
        return None
    return [
        grant.model_dump(mode="python", by_alias=True)
        for grant in request.outputGrants.thumbnails
    ]


def _output_grants_dict(request: AnalyzeRequest) -> dict[str, Any] | None:
    if request.outputGrants is None:
        return None
    return request.outputGrants.model_dump(mode="python", by_alias=True)


def _prior_facts_dict(request: AnalyzeRequest) -> dict[str, Any] | None:
    if request.priorFacts is None:
        return None
    return request.priorFacts.model_dump(mode="python", by_alias=True)


def _matte_target_dict(request: AnalyzeRequest) -> dict[str, Any] | None:
    if request.matteTarget is None:
        return None
    return request.matteTarget.model_dump(mode="python", by_alias=True)


def run_analyze(
    request: AnalyzeRequest,
    *,
    settings: Settings,
    runtime: RuntimeState,
    job: Job,
) -> dict[str, Any]:
    if not runtime.ready:
        raise AnalyzeError(MODEL_NOT_READY, "Required models are not loaded")
    validate_analyze_request(request, settings)
    if "audio" in request.features and settings.media_analysis_image == "analysis-cpu":
        if not _audio_vad_available(runtime, runtime.manifest, settings):
            raise AnalyzeError(
                FEATURE_UNAVAILABLE,
                "Silero VAD is not available for audio analysis",
            )
    deadline = time.monotonic() + settings.media_analysis_job_timeout_sec
    telemetry = JobTelemetry()
    _check_job(job, deadline)
    stage_budget = settings.media_analysis_stage_timeout_sec
    media_cache = get_local_media_cache(
        settings.media_analysis_cache_dir,
        max_bytes=settings.media_analysis_cache_max_bytes,
        ttl_sec=settings.media_analysis_cache_ttl_sec,
        enabled=settings.media_analysis_cache_enabled,
    )
    declared_fingerprint = (
        request.source.sha256.lower() if request.source.sha256 is not None else None
    )

    with tempfile.TemporaryDirectory(prefix="media-analysis-") as tmp:
        src = Path(tmp) / "source.bin"
        with _feature_stage("download", telemetry, job, deadline, stage_budget) as stage:
            cached_source = (
                media_cache.get("source", declared_fingerprint, suffix=".media", dest=src)
                if declared_fingerprint is not None
                else None
            )
            if cached_source is not None:
                # Cache hits still honor the signed request's freshness and URL policy.
                assert_source_fresh(request.source.expiresAt)
                assert_fetch_url(request.source.signedGetUrl, settings)
                if cached_source.byte_count > settings.media_analysis_max_bytes:
                    raise AnalyzeError(LIMIT_EXCEEDED, "Source exceeds MEDIA_ANALYSIS_MAX_BYTES")
                src = cached_source.path
                fingerprint = declared_fingerprint
                telemetry.source_cache_hit = True
            else:
                downloaded = fetch_source_to_path(
                    request.source.signedGetUrl,
                    src,
                    settings=settings,
                    expires_at=request.source.expiresAt,
                    expected_sha256=declared_fingerprint,
                    timeout_sec=min(
                        settings.media_analysis_download_timeout_sec,
                        stage.remaining(),
                    ),
                    cancel_check=stage,
                )
                telemetry.bytes_downloaded = downloaded.byte_count
                fingerprint = downloaded.sha256
                if declared_fingerprint is not None:
                    media_cache.put_file(
                        "source",
                        declared_fingerprint,
                        src,
                        suffix=".media",
                    )
        work = src
        preserve_audio = needs_canonical_audio(
            canonicalize=request.canonicalize,
            features=request.features,
        )
        with _feature_stage("probe", telemetry, job, deadline, stage_budget) as stage:
            media = probe(src, timeout_sec=stage.remaining())
            stage.check()
        telemetry.frames_decoded = media.frame_count
        telemetry.decoded_pixels = decoded_pixel_count(media)
        enforce_limits(media, settings)
        if request.canonicalize:
            dest = Path(tmp) / "canonical.mp4"
            with _feature_stage("canonicalize", telemetry, job, deadline, stage_budget) as stage:
                canonical_key = LocalMediaCache.stable_key(
                    fingerprint,
                    DECODE_PIPELINE_VERSION,
                    "audio" if preserve_audio else "video-only",
                )
                cached_canonical = (
                    media_cache.get("canonical", canonical_key, suffix=".mp4", dest=dest)
                    if declared_fingerprint is not None
                    else None
                )
                if cached_canonical is not None:
                    work = cached_canonical.path
                    media = probe(work, timeout_sec=stage.remaining())
                    telemetry.canonical_cache_hit = True
                else:
                    media = canonicalize(
                        src,
                        dest,
                        timeout_sec=stage.remaining(),
                        preserve_audio=preserve_audio,
                    )
                    work = dest
                    if declared_fingerprint is not None:
                        media_cache.put_file(
                            "canonical",
                            canonical_key,
                            dest,
                            suffix=".mp4",
                        )
                stage.check()
            telemetry.frames_decoded = media.frame_count
            telemetry.decoded_pixels = decoded_pixel_count(media)
            enforce_limits(media, settings)
        if request.analysisResolution:
            analysis_width = request.analysisResolution.width
            analysis_height = request.analysisResolution.height
            if analysis_width > media.width or analysis_height > media.height:
                raise AnalyzeError(
                    INVALID_REQUEST,
                    "analysisResolution must not upscale canonical media",
                )
            scale_x = analysis_width / media.width
            scale_y = analysis_height / media.height
            if abs(scale_x - scale_y) > 1e-6:
                raise AnalyzeError(
                    INVALID_REQUEST,
                    "analysisResolution must preserve canonical aspect ratio",
                )
        result_key = _result_cache_key(request, runtime, settings, fingerprint)
        cacheable_result = media_cache.enabled and declared_fingerprint is not None and not (
            set(request.features) & {"thumbnails", "person_matte"}
        )
        cached_result = (
            media_cache.get_json("result", result_key) if cacheable_result else None
        )
        if cached_result is not None and cached_result.get("overallStatus") != "completed":
            cached_result = None
        if cached_result is not None:
            result = cached_result
            result["requestedFeatures"] = list(request.features)
            telemetry.result_cache_hit = True
        else:
            result = _compute(
                request,
                media,
                work,
                runtime,
                settings,
                job,
                deadline,
                work_dir=Path(tmp),
                telemetry=telemetry,
            )
            if cacheable_result and result.get("overallStatus") == "completed":
                media_cache.put_json("result", result_key, result)
        if (
            request.canonicalize
            and request.outputGrants is not None
            and request.outputGrants.canonicalMp4 is not None
        ):
            with _feature_stage(
                "upload_canonical", telemetry, job, deadline, stage_budget
            ) as stage:
                grant = request.outputGrants.canonicalMp4
                artifact_bytes = work.read_bytes()
                artifact = upload_artifact(
                    artifact_bytes,
                    grant.signedPutUrl,
                    settings=settings,
                    expires_at=grant.expiresAt,
                    mime_type="video/mp4",
                    timeout_sec=min(
                        settings.media_analysis_download_timeout_sec,
                        stage.remaining(),
                    ),
                    cancel_check=stage,
                )
                telemetry.add_uploaded_bytes(int(artifact["byteCount"]))
            result["canonicalMedia"].update(artifact)
        result["telemetry"] = telemetry.as_dict()
        return result


def _result_cache_key(
    request: AnalyzeRequest,
    runtime: RuntimeState,
    settings: Settings,
    media_fingerprint: str,
) -> str:
    model_identity = ",".join(
        f"{entry.name}:{entry.sha256}"
        for entry in sorted(runtime.manifest.entries, key=lambda item: item.name)
    )
    return LocalMediaCache.stable_key(
        media_fingerprint,
        request_hash(request.model_dump(mode="python", by_alias=True)),
        __version__,
        "analysis-cache-v2",
        DECODE_PIPELINE_VERSION,
        OCR_SAMPLER_VERSION,
        MOTION_ANALYZER_VERSION,
        QUALITY_POLICY_VERSION,
        EXPOSURE_ANALYZER_VERSION,
        model_identity,
        f"motion-max-dimension:{settings.media_analysis_motion_max_dimension}",
        f"matte-keyframe-interval:{settings.media_analysis_matte_keyframe_interval}",
    )


FeatureOutcome = tuple[dict[str, Any], dict[str, Any], list[str]]


def _run_ocr_feature(
    *,
    path: Path,
    media: Any,
    shots: list[dict[str, Any]] | None,
    runtime: RuntimeState,
    settings: Settings,
    job: Job,
    deadline: float,
    telemetry: JobTelemetry,
    frame_access: BoundedFrameAccess,
) -> FeatureOutcome:
    body: dict[str, Any] = {}
    capabilities: dict[str, Any] = {}
    warnings: list[str] = []
    with _feature_stage(
        "ocr",
        telemetry,
        job,
        deadline,
        settings.media_analysis_stage_timeout_sec,
    ) as stage_check:
        entry = runtime.manifest.by_name("PP-OCRv5_mobile_det")
        try:
            if entry is None:
                raise RuntimeError("OCR model is absent")
            if entry.stub:
                regions: list[dict[str, Any]] = []
                ocr_status = "completed"
            else:
                if runtime.ocr_session is None:
                    raise RuntimeError("OCR runtime is unavailable")
                result = analyze_ocr(
                    path,
                    media,
                    shots=shots,
                    model_path=settings.media_analysis_model_dir / entry.file,
                    session=runtime.ocr_session,
                    frame_provider=lambda _path, index: frame_access.read_bgr(index),
                    cancel_check=stage_check,
                )
                regions = result.regions
                ocr_status = result.status
            if ocr_status == "failed":
                capabilities["ocr"] = {
                    "status": "failed",
                    "version": OCR_DETECTOR_VERSION,
                    "warningCodes": ["OCR_UNAVAILABLE"],
                }
                warnings.append("OCR_UNAVAILABLE")
            else:
                body["reservedRegions"] = regions
                capabilities["ocr"] = {
                    "status": "completed",
                    "version": OCR_DETECTOR_VERSION,
                }
                if not regions:
                    warnings.append("OCR_EMPTY")
        except AnalyzeError:
            raise
        except Exception:
            capabilities["ocr"] = {
                "status": "failed",
                "version": OCR_DETECTOR_VERSION,
                "warningCodes": ["OCR_UNAVAILABLE"],
            }
            warnings.append("OCR_UNAVAILABLE")
    return body, capabilities, warnings


def _run_audio_features(
    *,
    requested_set: set[str],
    path: Path,
    media: Any,
    runtime: RuntimeState,
    settings: Settings,
    job: Job,
    deadline: float,
    telemetry: JobTelemetry,
) -> FeatureOutcome:
    body: dict[str, Any] = {}
    capabilities: dict[str, Any] = {}
    warnings: list[str] = []
    stage_budget = settings.media_analysis_stage_timeout_sec
    with _feature_stage("pcm", telemetry, job, deadline, stage_budget) as stage_check:
        shared_pcm = extract_analysis_pcm(
            path,
            has_audio=media.has_audio,
            source_duration_sec=media.duration,
            source_sample_rate=media.audio_sample_rate,
            source_channel_count=media.audio_channel_count,
            timeout_sec=stage_check.remaining(),
            cancel_check=stage_check,
        )
    vad_session = _resolve_vad_session(runtime, runtime.manifest, settings)

    if "audio" in requested_set:
        with _feature_stage("audio", telemetry, job, deadline, stage_budget) as stage_check:
            try:
                audio = analyze_audio(
                    shared_pcm,
                    source_path=path if media.has_audio else None,
                    fps=media.fps,
                    vad_session=vad_session,
                    require_vad=True,
                    timeout_sec=stage_check.remaining(),
                    cancel_check=stage_check,
                )
                body["audio"] = audio.to_payload()
                audio_warnings = list(audio.warning_codes)
                warnings.extend(audio_warnings)
                capabilities["audio"] = audio_capability(
                    status=audio.status,
                    warning_codes=audio_warnings or None,
                )
            except AnalyzeError:
                raise
            except Exception:
                capabilities["audio"] = audio_capability(status="failed")

    if "waveform" in requested_set:
        with _feature_stage("waveform", telemetry, job, deadline, stage_budget):
            try:
                waveform_payload, waveform_warnings = analyze_waveform(shared_pcm)
                body["waveform"] = waveform_payload
                warnings.extend(waveform_warnings)
                capabilities["waveform"] = waveform_capability(
                    status="completed",
                    warning_codes=waveform_warnings or None,
                )
            except AnalyzeError:
                raise
            except Exception:
                capabilities["waveform"] = waveform_capability(status="failed")
    return body, capabilities, warnings


def _compute(
    request: AnalyzeRequest,
    media: Any,
    path: Path,
    runtime: RuntimeState,
    settings: Settings,
    job: Job,
    deadline: float,
    *,
    work_dir: Path,
    telemetry: JobTelemetry,
) -> dict[str, Any]:
    if settings.media_analysis_image.startswith("matte"):
        return _compute_matte(
            request,
            media,
            path,
            runtime,
            settings,
            job,
            deadline,
            work_dir,
            telemetry,
        )

    with BoundedFrameAccess(
        path,
        config=FrameAccessConfig(
            max_cached_bytes=settings.media_analysis_frame_cache_bytes,
            max_spill_bytes=settings.media_analysis_frame_spill_bytes,
        ),
        cancel_check=lambda: _check_job(job, deadline),
    ) as frame_access:
        result = _compute_cpu(
            request,
            media,
            path,
            runtime,
            settings,
            job,
            deadline,
            telemetry=telemetry,
            frame_access=frame_access,
        )
        telemetry.unique_frames_decoded = frame_access.unique_decodes
        telemetry.frame_cache_hits = frame_access.cache_hits
        telemetry.frame_disk_hits = frame_access.disk_hits
        telemetry.frame_spill_bytes = frame_access.spill_bytes
        telemetry.frame_spill_limited = frame_access.spill_limited
        telemetry.record_stage("decode", frame_access.decode_duration_ms)
        return result


def _compute_cpu(
    request: AnalyzeRequest,
    media: Any,
    path: Path,
    runtime: RuntimeState,
    settings: Settings,
    job: Job,
    deadline: float,
    *,
    telemetry: JobTelemetry,
    frame_access: BoundedFrameAccess,
) -> dict[str, Any]:
    # Signal cooperative cancellation before ExitStack joins the background work.
    # The caller can only release frame access, media files, and model ownership
    # once every worker has stopped, including when a foreground feature fails.
    with ExitStack() as resources:
        try:
            return _compute_cpu_inner(
                request, media, path, runtime, settings, job, deadline,
                telemetry=telemetry, frame_access=frame_access, resources=resources,
            )
        except BaseException:
            job.cancel.set()
            raise


def _compute_cpu_inner(
    request: AnalyzeRequest,
    media: Any,
    path: Path,
    runtime: RuntimeState,
    settings: Settings,
    job: Job,
    deadline: float,
    *,
    telemetry: JobTelemetry,
    frame_access: BoundedFrameAccess,
    resources: ExitStack,
) -> dict[str, Any]:

    manifest = runtime.manifest
    fps = media.fps
    requested = list(request.features)
    requested_set = set(requested)
    capabilities: dict[str, Any] = {}
    body: dict[str, Any] = {}
    warnings: list[str] = []
    stage_budget = settings.media_analysis_stage_timeout_sec

    def cancel_check() -> None:
        _check_job(job, deadline)

    prior_shots = _prior_shot_dicts(request)
    fill_only = prior_shots is not None and "shots" not in requested_set
    internal_shots: list[dict[str, Any]] | None = None

    needs_shot_timeline = bool(SHOT_TIMELINE_FEATURES & requested_set) or prior_shots is not None

    def retain_decode_sample(frame_index: int, frame: object) -> None:
        if not ({"motion", "quality"} & requested_set):
            return
        if frame_index % 2 != 0 and frame_index != media.frame_count - 1:
            return
        frame_access.prime_gray(
            frame_index,
            frame,
            max_dimension=settings.media_analysis_motion_max_dimension,
        )

    def analyze_shot_timeline(stage_check: StageDeadline) -> list[dict[str, Any]]:
        if {"motion", "quality"} & requested_set:
            return analyze_shots(
                path,
                media,
                frame_tap=retain_decode_sample,
                cancel_check=stage_check,
            )
        return analyze_shots(path, media, cancel_check=stage_check)

    if "shots" in requested_set:
        with _feature_stage("shots", telemetry, job, deadline, stage_budget) as stage_check:
            try:
                internal_shots = analyze_shot_timeline(stage_check)
                stage_check()
                body["shots"] = internal_shots
                capabilities["shots"] = {
                    "status": "completed",
                    "version": SHOT_ANALYZER_VERSION,
                }
            except AnalyzeError:
                raise
            except Exception:
                capabilities["shots"] = {
                    "status": "failed",
                    "version": SHOT_ANALYZER_VERSION,
                }
    elif prior_shots is not None and needs_shot_timeline:
        internal_shots = prior_shots
    elif needs_shot_timeline:
        with _feature_stage(
            "shots_internal", telemetry, job, deadline, stage_budget
        ) as stage_check:
            internal_shots = analyze_shot_timeline(stage_check)
            stage_check()

    feature_calls: list[tuple[str, Any]] = []
    if "ocr" in requested_set:
        feature_calls.append(
            (
                "ocr",
                lambda: _run_ocr_feature(
                    path=path,
                    media=media,
                    shots=internal_shots,
                    runtime=runtime,
                    settings=settings,
                    job=job,
                    deadline=deadline,
                    telemetry=telemetry,
                    frame_access=frame_access,
                ),
            )
        )
    if AUDIO_FEATURES & requested_set:
        feature_calls.append(
            (
                "audio",
                lambda: _run_audio_features(
                    requested_set=requested_set,
                    path=path,
                    media=media,
                    runtime=runtime,
                    settings=settings,
                    job=job,
                    deadline=deadline,
                    telemetry=telemetry,
                ),
            )
        )
    background_slots = max(0, settings.media_analysis_feature_workers - 1)
    feature_pool = (
        resources.enter_context(ThreadPoolExecutor(
            max_workers=min(background_slots, len(feature_calls)),
            thread_name_prefix="media-feature",
        ))
        if background_slots and feature_calls
        else None
    )
    background_features: list[tuple[str, Future[FeatureOutcome]]] = []
    if feature_pool is not None:
        background_features = [
            (name, feature_pool.submit(call)) for name, call in feature_calls
        ]

    if "subjects" in requested_set:
        with _feature_stage("subjects", telemetry, job, deadline, stage_budget) as stage_check:
            entry = manifest.by_name("yolox-tiny")
            try:
                if entry is None:
                    raise RuntimeError("subject model is absent")
                if entry.stub:
                    subjects: list[dict[str, Any]] = []
                else:
                    if internal_shots is None or runtime.subject_detector is None:
                        raise RuntimeError("subject dependencies are unavailable")
                    subjects = analyze_subjects(
                        path,
                        media,
                        shots=internal_shots,
                        detector=runtime.subject_detector,
                        frame_provider=frame_access.read_bgr,
                        cancel_check=stage_check,
                    )
                body["subjects"] = subjects
                if internal_shots is not None:
                    classify_shots_with_subjects(internal_shots, subjects, media)
                capabilities["subjects"] = {
                    "status": "completed",
                    "version": TRACKER_VERSION,
                }
            except AnalyzeError:
                raise
            except Exception:
                capabilities["subjects"] = {
                    "status": "failed",
                    "version": TRACKER_VERSION,
                }

    if "faces" in requested_set:
        with _feature_stage("faces", telemetry, job, deadline, stage_budget) as stage_check:
            entry = manifest.by_name("yunet")
            try:
                if entry is None:
                    raise RuntimeError("face model is absent")
                if entry.stub:
                    faces = empty_face_analysis()
                else:
                    if internal_shots is None or runtime.face_detector is None:
                        raise RuntimeError("face dependencies are unavailable")
                    face_subjects = body.get("subjects")
                    if (
                        face_subjects is None
                        and request.priorFacts
                        and request.priorFacts.subjects is not None
                    ):
                        face_subjects = [
                            item.model_dump(by_alias=True)
                            for item in request.priorFacts.subjects
                        ]
                    faces = analyze_faces(
                        path,
                        media,
                        shots=internal_shots,
                        subjects=face_subjects,
                        analysis_width=(
                            request.analysisResolution.width
                            if request.analysisResolution
                            else None
                        ),
                        analysis_height=(
                            request.analysisResolution.height
                            if request.analysisResolution
                            else None
                        ),
                        detector=runtime.face_detector,
                        frame_reader=frame_access.resized_view(
                            width=(
                                request.analysisResolution.width
                                if request.analysisResolution
                                else media.width
                            ),
                            height=(
                                request.analysisResolution.height
                                if request.analysisResolution
                                else media.height
                            ),
                        ),
                        cancel_check=stage_check,
                    )
                body["faces"] = faces
                capabilities["faces"] = {
                    "status": "completed",
                    "version": YUNET_CAPABILITY_VERSION,
                }
            except AnalyzeError:
                raise
            except Exception:
                capabilities["faces"] = {
                    "status": "failed",
                    "version": YUNET_CAPABILITY_VERSION,
                }

    subject_tracks = body.get("subjects")
    if subject_tracks is None and request.priorFacts and request.priorFacts.subjects:
        subject_tracks = [
            track.model_dump(mode="python", by_alias=True)
            for track in request.priorFacts.subjects
        ]
    subject_boxes = _subject_boxes_by_frame(subject_tracks)

    visual_requested = VISUAL_SHARED & requested_set
    if visual_requested:
        with nullcontext(frame_access):
            read_frame = frame_access.read_bgr

            def read_gray_frame(index: int):
                return frame_access.read_gray(
                    index,
                    max_dimension=settings.media_analysis_motion_max_dimension,
                )

            motion_result = None
            motion_estimator = None

            if "motion" in requested_set:
                with _feature_stage(
                    "motion", telemetry, job, deadline, stage_budget
                ) as stage_check:
                    try:
                        motion_result = analyze_motion(
                            path,
                            media,
                            frame_provider=read_frame,
                            gray_frame_provider=read_gray_frame,
                            subject_boxes_by_frame=subject_boxes,
                            cancel_check=stage_check,
                        )
                        body["motion"] = motion_result.as_motion_analysis()
                        capabilities["motion"] = {
                            "status": "completed",
                            "version": MOTION_ANALYZER_VERSION,
                        }
                    except AnalyzeError:
                        raise
                    except Exception:
                        capabilities["motion"] = {
                            "status": "failed",
                            "version": MOTION_ANALYZER_VERSION,
                        }

            if "quality" in requested_set:
                with _feature_stage(
                    "quality", telemetry, job, deadline, stage_budget
                ) as stage_check:
                    try:
                        if motion_result is None and "motion" not in requested_set:
                            motion_estimator = make_internal_motion_estimator(
                                media,
                                read_gray_frame,
                                subject_boxes_by_frame=subject_boxes,
                                cancel_check=stage_check,
                            )
                        quality = analyze_quality(
                            path,
                            media,
                            shots=internal_shots,
                            motion=motion_result,
                            compute_motion=motion_estimator,
                            subject_boxes_by_frame=subject_boxes,
                            frame_provider=read_frame,
                            cancel_check=stage_check,
                        )
                        body["quality"] = quality.as_quality_analysis()
                        capabilities["quality"] = {
                            "status": "completed",
                            "version": QUALITY_POLICY_VERSION,
                        }
                    except AnalyzeError:
                        raise
                    except Exception:
                        capabilities["quality"] = {
                            "status": "failed",
                            "version": QUALITY_POLICY_VERSION,
                        }

            if "exposure" in requested_set:
                with _feature_stage(
                    "exposure", telemetry, job, deadline, stage_budget
                ) as stage_check:
                    try:
                        exposure = analyze_exposure(
                            path,
                            media,
                            shots=internal_shots,
                            prior_facts=request.priorFacts,
                            fill_only=fill_only,
                            frame_provider=lambda index: frame_access.read_bgr_downscaled(
                                index,
                                max_pixels=65536,
                            ),
                            cancel_check=stage_check,
                        )
                        body["exposure"] = {"perShot": exposure.per_shot}
                        capabilities["exposure"] = {
                            "status": exposure.status,
                            "version": EXPOSURE_ANALYZER_VERSION,
                        }
                    except AnalyzeError:
                        raise
                    except Exception:
                        capabilities["exposure"] = {
                            "status": "failed",
                            "version": EXPOSURE_ANALYZER_VERSION,
                        }

            if "thumbnails" in requested_set:
                with _feature_stage(
                    "thumbnails", telemetry, job, deadline, stage_budget
                ) as stage_check:
                    try:
                        def upload_thumbnail(grant: ThumbnailGrant, payload: bytes) -> None:
                            with _feature_stage(
                                "upload_thumbnail",
                                telemetry,
                                job,
                                deadline,
                                stage_budget,
                            ) as upload_stage:
                                uploaded = upload_artifact(
                                    payload,
                                    grant.signed_put_url,
                                    settings=settings,
                                    expires_at=grant.expires_at,
                                    mime_type=THUMBNAIL_MIME_TYPE,
                                    timeout_sec=min(
                                        settings.media_analysis_download_timeout_sec,
                                        upload_stage.remaining(),
                                    ),
                                    cancel_check=upload_stage,
                                )
                                telemetry.add_uploaded_bytes(int(uploaded["byteCount"]))

                        thumb = analyze_thumbnails(
                            path,
                            media,
                            shots=internal_shots,
                            prior_facts=request.priorFacts,
                            fill_only=fill_only,
                            thumbnail_grants=_thumbnail_grants(request),
                            upload_callback=upload_thumbnail,
                            frame_provider=read_frame,
                            cancel_check=stage_check,
                        )
                        body["thumbnailCandidates"] = thumb.candidates
                        capability: dict[str, Any] = {
                            "status": thumb.status,
                            "version": thumb.version,
                        }
                        if thumb.warning_codes:
                            capability["warningCodes"] = list(thumb.warning_codes)
                            warnings.extend(thumb.warning_codes)
                        capabilities["thumbnails"] = capability
                    except AnalyzeError:
                        raise
                    except Exception:
                        capabilities["thumbnails"] = {
                            "status": "failed",
                            "version": THUMBNAIL_ANALYZER_VERSION,
                        }

    outcomes: list[FeatureOutcome]
    if feature_pool is None:
        outcomes = [call() for _name, call in feature_calls]
    else:
        outcomes = [future.result() for _name, future in background_features]
    for feature_body, feature_capabilities, feature_warnings in outcomes:
        body.update(feature_body)
        capabilities.update(feature_capabilities)
        warnings.extend(feature_warnings)

    if request.analysisResolution:
        body["analysisTransform"] = analysis_transform(
            analysis_width=request.analysisResolution.width,
            analysis_height=request.analysisResolution.height,
            canonical_width=media.width,
            canonical_height=media.height,
        )

    statuses = {item["status"] for item in capabilities.values()}
    if statuses == {"completed"}:
        overall = "completed"
    elif "failed" in statuses:
        overall = "partial" if "completed" in statuses else "failed"
    else:
        overall = "partial"

    canonical = {
        "fps": fps.as_dict(),
        "timeBase": Rational(fps.denominator, fps.numerator).as_dict()
        if fps.numerator
        else {"numerator": 1, "denominator": 30},
        "frameCount": media.frame_count,
        "durationSec": duration_sec(media.frame_count, fps),
        "width": media.width,
        "height": media.height,
        "mimeType": "video/mp4",
        "codec": "h264" if media.codec in {"h264", "avc1"} else media.codec,
        "rotationApplied": True,
        "sampleAspectRatio": "1:1",
        "colorSpace": "bt709",
        "videoRange": "limited",
        "hasAudio": media.has_audio,
        "audioCodec": media.audio_codec,
        "audioSampleRate": media.audio_sample_rate,
        "audioChannelCount": media.audio_channel_count,
        "audioBitrateKbps": None,
    }

    provenance = {
        "visualAnalyzerVersion": __version__,
        "decodePipelineVersion": DECODE_PIPELINE_VERSION,
        "samplingPolicyVersion": SAMPLING_POLICY_VERSION,
        **analyze_build_provenance(),
    }
    if "subjects" in requested_set:
        provenance["trackerVersion"] = TRACKER_VERSION
        entry = manifest.by_name("yolox-tiny")
        if entry:
            provenance["subjectModel"] = subject_model_provenance(entry)
    if "faces" in requested_set:
        provenance["faceTrackerVersion"] = FACE_TRACKER_VERSION
        if "subjects" in requested_set or (
            request.priorFacts and request.priorFacts.subjects is not None
        ):
            provenance["faceAssociationVersion"] = FACE_ASSOCIATION_VERSION
        entry = manifest.by_name("yunet")
        if entry:
            provenance["faceModel"] = model_provenance(
                entry,
                runtime="stub" if entry.stub else "opencv-dnn",
                preprocessing=YUNET_PREPROCESSING_VERSION,
            )
    if "ocr" in requested_set:
        provenance["ocrSamplerVersion"] = OCR_SAMPLER_VERSION
        provenance["ocrMergeVersion"] = OCR_MERGE_VERSION
        entry = manifest.by_name("PP-OCRv5_mobile_det")
        if entry:
            provenance["ocrModel"] = model_provenance(
                entry,
                runtime="stub" if entry.stub else "onnxruntime-cpu",
                preprocessing=preprocessing_version(),
            )
    if "shots" in requested_set:
        provenance["shotAnalyzerVersion"] = SHOT_ANALYZER_VERSION
        provenance["shotClassifierVersion"] = SHOT_CLASSIFIER_VERSION
    if "motion" in requested_set:
        provenance["motionAnalyzerVersion"] = MOTION_ANALYZER_VERSION
        provenance["motionWorkingMaxDimension"] = (
            settings.media_analysis_motion_max_dimension
        )
    if "quality" in requested_set:
        provenance["qualityPolicyVersion"] = QUALITY_POLICY_VERSION
    if AUDIO_FEATURES & requested_set:
        provenance["pcmExtractionVersion"] = PCM_EXTRACTION_VERSION
    if "audio" in requested_set:
        vad_entry = manifest.by_name(SILERO_VAD_MODEL_NAME)
        provenance.update(
            audio_provenance(
                vad_entry=vad_entry,
                vad_runtime="stub"
                if vad_entry is not None and vad_entry.stub
                else "onnxruntime-cpu",
            )
        )
    if "waveform" in requested_set:
        provenance["waveformAnalyzerVersion"] = WAVEFORM_ANALYZER_VERSION
    if "exposure" in requested_set:
        provenance.update(exposure_policy_metadata())
    if "thumbnails" in requested_set:
        provenance.update(thumbnail_policy_metadata())
    if request.priorFacts and request.priorFacts.policyVersions:
        policy = request.priorFacts.policyVersions
        if policy.qualityPolicyVersion:
            provenance["priorQualityPolicyVersion"] = policy.qualityPolicyVersion
        if policy.motionAnalyzerVersion:
            provenance["priorMotionAnalyzerVersion"] = policy.motionAnalyzerVersion
        if policy.exposureAnalyzerVersion:
            provenance["priorExposureAnalyzerVersion"] = policy.exposureAnalyzerVersion
        if policy.audioAnalyzerVersion:
            provenance["priorAudioAnalyzerVersion"] = policy.audioAnalyzerVersion
        if policy.samplingPolicyVersion:
            provenance["priorSamplingPolicyVersion"] = policy.samplingPolicyVersion

    return {
        "schemaVersion": 1,
        "overallStatus": overall,
        "requestedFeatures": requested,
        "canonicalMedia": canonical,
        "capabilities": capabilities,
        **body,
        "provenance": provenance,
        "warningCodes": sorted(set(warnings)),
    }


def _compute_matte(
    request: AnalyzeRequest,
    media: Any,
    path: Path,
    runtime: RuntimeState,
    settings: Settings,
    job: Job,
    deadline: float,
    work_dir: Path,
    telemetry: JobTelemetry,
) -> dict[str, Any]:
    requested = list(request.features)
    capabilities: dict[str, Any] = {}
    body: dict[str, Any] = {}
    warnings: list[str] = []
    stage_budget = settings.media_analysis_stage_timeout_sec

    def cancel_check() -> None:
        _check_job(job, deadline)

    manifest = runtime.manifest
    modnet_entry = manifest.by_name("modnet")

    if runtime.modnet_session is None:
        raise AnalyzeError(MODEL_NOT_READY, "Required models are not loaded")

    canonical = CanonicalMediaSpec(
        fps=media.fps,
        time_base=Rational(media.fps.denominator, media.fps.numerator)
        if media.fps.numerator
        else Rational(1, 30),
        frame_count=media.frame_count,
        width=media.width,
        height=media.height,
    )

    def upload_matte(
        body_bytes: bytes,
        content_type: str,
        grant: Any,
        *,
        timeout_sec: float,
        cancel: Any,
    ) -> None:
        upload_artifact(
            body_bytes,
            grant.signed_put_url,
            settings=settings,
            expires_at=grant.expires_at,
            mime_type=content_type,
            timeout_sec=min(settings.media_analysis_download_timeout_sec, timeout_sec),
            cancel_check=cancel,
        )

    try:
        with _feature_stage(
            "person_matte", telemetry, job, deadline, stage_budget
        ) as stage_check:
            with BoundedFrameAccess(
                path,
                config=FrameAccessConfig(
                    max_cached_frames=1,
                    max_cached_bytes=settings.media_analysis_frame_cache_bytes,
                    max_spill_bytes=0,  # Matte is a single sequential consumer.
                ),
                cancel_check=stage_check,
            ) as frame_access:

                def frame_stream():
                    for index in range(media.frame_count):
                        stage_check()
                        yield index, frame_access.read_bgr(index)

                def upload_timed(body_bytes: bytes, content_type: str, grant: Any) -> None:
                    with _feature_stage(
                        "upload_matte",
                        telemetry,
                        job,
                        deadline,
                        stage_budget,
                    ) as upload_stage:
                        upload_matte(
                            body_bytes,
                            content_type,
                            grant,
                            timeout_sec=upload_stage.remaining(),
                            cancel=upload_stage,
                        )
                        telemetry.add_uploaded_bytes(len(body_bytes))

                candidate = run_person_matte_stage1(
                    frames=frame_stream(),
                    canonical=canonical,
                    matte_target=_matte_target_dict(request),
                    output_grants=_output_grants_dict(request),
                    prior_facts=_prior_facts_dict(request),
                    session=runtime.modnet_session,
                    upload=upload_timed,
                    work_dir=work_dir,
                    cancel_check=stage_check,
                    deadline=min(deadline, stage_check.stage_deadline),
                    keyframe_interval=settings.media_analysis_matte_keyframe_interval,
                )
                telemetry.unique_frames_decoded = frame_access.unique_decodes
                telemetry.frame_cache_hits = frame_access.cache_hits
                telemetry.record_stage("decode", frame_access.decode_duration_ms)
        body["personMatte"] = candidate.as_dict()
        capabilities["person_matte"] = {
            "status": "completed",
            "version": MATTE_TEMPORAL_POLICY_VERSION,
        }
        if runtime.reference_mode:
            warnings.append("MATTE_REFERENCE_MODE")
    except AnalyzeError:
        raise
    except Exception:
        capabilities["person_matte"] = {
            "status": "failed",
            "version": MATTE_TEMPORAL_POLICY_VERSION,
        }
        overall = "failed"
    else:
        overall = "completed"

    lock = None
    lock_path = settings.media_analysis_model_dir / "manifest.lock.json"
    if not lock_path.is_file():
        lock_path = Path("models/manifest.lock.json")
    if lock_path.is_file():
        try:
            lock = load_lock(lock_path, profile=settings.media_analysis_image)
        except ValueError:
            lock = None

    provenance = {
        "visualAnalyzerVersion": __version__,
        "decodePipelineVersion": DECODE_PIPELINE_VERSION,
        "referenceMode": runtime.reference_mode,
        "productionInferenceReady": runtime.production_inference_ready,
        **analyze_build_provenance(),
    }
    provenance.update(
        matte_provenance_block(
            modnet_entry=modnet_entry,
            lock=lock,
            runtime=(
                ",".join(runtime.execution_providers)
                if runtime.execution_providers
                else "onnxruntime"
            ),
        )
    )
    if runtime.reference_mode:
        provenance["matteReferenceModeNote"] = (
            "Reference/test mode enabled; stub pipeline is not production-ready"
        )

    return {
        "schemaVersion": 1,
        "overallStatus": overall,
        "requestedFeatures": requested,
        "canonicalMedia": {
            "fps": media.fps.as_dict(),
            "timeBase": Rational(media.fps.denominator, media.fps.numerator).as_dict()
            if media.fps.numerator
            else {"numerator": 1, "denominator": 30},
            "frameCount": media.frame_count,
            "durationSec": duration_sec(media.frame_count, media.fps),
            "width": media.width,
            "height": media.height,
            "mimeType": "video/mp4",
            "codec": "h264" if media.codec in {"h264", "avc1"} else media.codec,
            "rotationApplied": True,
            "sampleAspectRatio": "1:1",
            "colorSpace": "bt709",
            "videoRange": "limited",
            "hasAudio": media.has_audio,
            "audioCodec": media.audio_codec,
            "audioSampleRate": media.audio_sample_rate,
            "audioChannelCount": media.audio_channel_count,
            "audioBitrateKbps": None,
        },
        "capabilities": capabilities,
        **body,
        "provenance": provenance,
        "warningCodes": sorted(set(warnings)),
    }
