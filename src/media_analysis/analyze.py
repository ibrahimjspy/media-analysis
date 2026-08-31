from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from media_analysis import __version__
from media_analysis.config import IMPLEMENTED_CPU, MATTE_FEATURES, Settings
from media_analysis.decode import (
    DECODE_PIPELINE_VERSION,
    canonicalize,
    enforce_limits,
    needs_canonical_audio,
    probe,
)
from media_analysis.errors import (
    CANCELLED,
    FEATURE_UNAVAILABLE,
    INVALID_REQUEST,
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
from media_analysis.frame_access import BoundedFrameAccess
from media_analysis.frames import Rational, analysis_transform, duration_sec
from media_analysis.jobs import Job
from media_analysis.models_manifest import SILERO_VAD_MODEL_NAME
from media_analysis.person_matte.constants import MATTE_TEMPORAL_POLICY_VERSION
from media_analysis.person_matte.pipeline import run_person_matte_stage1
from media_analysis.person_matte.provenance import matte_provenance_block
from media_analysis.person_matte.types import CanonicalMediaSpec
from media_analysis.runtime import RuntimeState
from media_analysis.schemas import AnalyzeRequest
from media_analysis.source import fetch_source
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
    _check_job(job, deadline)

    payload = fetch_source(
        request.source.signedGetUrl,
        settings=settings,
        expires_at=request.source.expiresAt,
        expected_sha256=request.source.sha256,
        timeout_sec=min(settings.media_analysis_download_timeout_sec, _remaining(deadline)),
        cancel_check=lambda: _check_job(job, deadline),
    )
    _check_job(job, deadline)

    with tempfile.TemporaryDirectory(prefix="media-analysis-") as tmp:
        src = Path(tmp) / "source.bin"
        src.write_bytes(payload)
        work = src
        preserve_audio = needs_canonical_audio(
            canonicalize=request.canonicalize,
            features=request.features,
        )
        if request.canonicalize:
            dest = Path(tmp) / "canonical.mp4"
            media = canonicalize(
                src,
                dest,
                timeout_sec=_remaining(deadline),
                preserve_audio=preserve_audio,
            )
            work = dest
        else:
            media = probe(src, timeout_sec=_remaining(deadline))
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
        _check_job(job, deadline)
        result = _compute(
            request,
            media,
            work,
            runtime,
            settings,
            job,
            deadline,
            work_dir=Path(tmp),
        )
        if (
            request.canonicalize
            and request.outputGrants is not None
            and request.outputGrants.canonicalMp4 is not None
        ):
            _check_job(job, deadline)
            grant = request.outputGrants.canonicalMp4
            artifact = upload_artifact(
                work.read_bytes(),
                grant.signedPutUrl,
                settings=settings,
                expires_at=grant.expiresAt,
                mime_type="video/mp4",
                timeout_sec=min(
                    settings.media_analysis_download_timeout_sec,
                    _remaining(deadline),
                ),
                cancel_check=lambda: _check_job(job, deadline),
            )
            result["canonicalMedia"].update(artifact)
        return result


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
) -> dict[str, Any]:
    if settings.media_analysis_image.startswith("matte"):
        return _compute_matte(request, media, path, runtime, settings, job, deadline, work_dir)

    manifest = runtime.manifest
    fps = media.fps
    requested = list(request.features)
    requested_set = set(requested)
    capabilities: dict[str, Any] = {}
    body: dict[str, Any] = {}
    warnings: list[str] = []

    def cancel_check() -> None:
        _check_job(job, deadline)

    prior_shots = _prior_shot_dicts(request)
    fill_only = prior_shots is not None and "shots" not in requested_set
    internal_shots: list[dict[str, Any]] | None = None

    needs_shot_timeline = bool(SHOT_TIMELINE_FEATURES & requested_set) or prior_shots is not None

    if "shots" in requested_set:
        _check_job(job, deadline)
        try:
            internal_shots = analyze_shots(path, media)
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
        _check_job(job, deadline)
        internal_shots = analyze_shots(path, media)

    if "subjects" in requested_set:
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
                    cancel_check=cancel_check,
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

    _check_job(job, deadline)
    if "faces" in requested_set:
        entry = manifest.by_name("yunet")
        try:
            if entry is None:
                raise RuntimeError("face model is absent")
            if entry.stub:
                faces = empty_face_analysis()
            else:
                if internal_shots is None or runtime.face_detector is None:
                    raise RuntimeError("face dependencies are unavailable")
                faces = analyze_faces(
                    path,
                    media,
                    shots=internal_shots,
                    subjects=body.get("subjects"),
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
                    cancel_check=cancel_check,
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

    _check_job(job, deadline)
    if "ocr" in requested_set:
        entry = manifest.by_name("PP-OCRv5_mobile_det")
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
                    shots=internal_shots,
                    model_path=settings.media_analysis_model_dir / entry.file,
                    session=runtime.ocr_session,
                    cancel_check=cancel_check,
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

    subject_tracks = body.get("subjects")
    if subject_tracks is None and request.priorFacts and request.priorFacts.subjects:
        subject_tracks = [
            track.model_dump(mode="python", by_alias=True)
            for track in request.priorFacts.subjects
        ]
    subject_boxes = _subject_boxes_by_frame(subject_tracks)

    visual_requested = VISUAL_SHARED & requested_set
    if visual_requested:
        _check_job(job, deadline)
        with BoundedFrameAccess(path, cancel_check=cancel_check) as frame_access:
            read_frame = frame_access.read_bgr
            motion_result = None
            motion_estimator = None

            if "motion" in requested_set:
                try:
                    motion_result = analyze_motion(
                        path,
                        media,
                        frame_provider=read_frame,
                        subject_boxes_by_frame=subject_boxes,
                        cancel_check=cancel_check,
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
                try:
                    if motion_result is None and "motion" not in requested_set:
                        motion_estimator = make_internal_motion_estimator(
                            media,
                            read_frame,
                            subject_boxes_by_frame=subject_boxes,
                            cancel_check=cancel_check,
                        )
                    quality = analyze_quality(
                        path,
                        media,
                        shots=internal_shots,
                        motion=motion_result,
                        compute_motion=motion_estimator,
                        subject_boxes_by_frame=subject_boxes,
                        frame_provider=read_frame,
                        cancel_check=cancel_check,
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
                try:
                    exposure = analyze_exposure(
                        path,
                        media,
                        shots=internal_shots,
                        prior_facts=request.priorFacts,
                        fill_only=fill_only,
                        frame_provider=read_frame,
                        cancel_check=cancel_check,
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
                try:
                    def upload_thumbnail(grant: ThumbnailGrant, payload: bytes) -> None:
                        upload_artifact(
                            payload,
                            grant.signed_put_url,
                            settings=settings,
                            expires_at=grant.expires_at,
                            mime_type=THUMBNAIL_MIME_TYPE,
                            timeout_sec=min(
                                settings.media_analysis_download_timeout_sec,
                                _remaining(deadline),
                            ),
                            cancel_check=cancel_check,
                        )

                    thumb = analyze_thumbnails(
                        path,
                        media,
                        shots=internal_shots,
                        prior_facts=request.priorFacts,
                        fill_only=fill_only,
                        thumbnail_grants=_thumbnail_grants(request),
                        upload_callback=upload_thumbnail,
                        frame_provider=read_frame,
                        cancel_check=cancel_check,
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

    if AUDIO_FEATURES & requested_set:
        _check_job(job, deadline)
        shared_pcm = extract_analysis_pcm(
            path,
            has_audio=media.has_audio,
            source_duration_sec=media.duration,
            source_sample_rate=media.audio_sample_rate,
            source_channel_count=media.audio_channel_count,
            timeout_sec=_remaining(deadline),
            cancel_check=cancel_check,
        )
        vad_session = _resolve_vad_session(runtime, manifest, settings)

        if "audio" in requested_set:
            try:
                audio = analyze_audio(
                    shared_pcm,
                    source_path=path if media.has_audio else None,
                    fps=fps,
                    vad_session=vad_session,
                    require_vad=True,
                    timeout_sec=_remaining(deadline),
                    cancel_check=cancel_check,
                )
                body["audio"] = audio.to_payload()
                audio_warnings = list(audio.warning_codes)
                if audio_warnings:
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
            try:
                waveform_payload, waveform_warnings = analyze_waveform(shared_pcm)
                body["waveform"] = waveform_payload
                if waveform_warnings:
                    warnings.extend(waveform_warnings)
                capabilities["waveform"] = waveform_capability(
                    status="completed",
                    warning_codes=waveform_warnings or None,
                )
            except AnalyzeError:
                raise
            except Exception:
                capabilities["waveform"] = waveform_capability(status="failed")

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
    }
    if "subjects" in requested_set:
        provenance["trackerVersion"] = TRACKER_VERSION
        entry = manifest.by_name("yolox-tiny")
        if entry:
            provenance["subjectModel"] = subject_model_provenance(entry)
    if "faces" in requested_set:
        provenance["faceTrackerVersion"] = FACE_TRACKER_VERSION
        if "subjects" in requested_set:
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
) -> dict[str, Any]:
    requested = list(request.features)
    capabilities: dict[str, Any] = {}
    body: dict[str, Any] = {}
    warnings: list[str] = []

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

    def upload_matte(body_bytes: bytes, content_type: str, grant: Any) -> None:
        upload_artifact(
            body_bytes,
            grant.signed_put_url,
            settings=settings,
            expires_at=grant.expires_at,
            mime_type=content_type,
            timeout_sec=min(settings.media_analysis_download_timeout_sec, _remaining(deadline)),
            cancel_check=cancel_check,
        )

    try:
        with BoundedFrameAccess(path, cancel_check=cancel_check) as frame_access:

            def frame_stream():
                for index in range(media.frame_count):
                    cancel_check()
                    yield index, frame_access.read_bgr(index)

            candidate = run_person_matte_stage1(
                frames=frame_stream(),
                canonical=canonical,
                matte_target=_matte_target_dict(request),
                output_grants=_output_grants_dict(request),
                prior_facts=_prior_facts_dict(request),
                session=runtime.modnet_session,
                upload=upload_matte,
                work_dir=work_dir,
                cancel_check=cancel_check,
                deadline=deadline,
            )
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
    }
    provenance.update(
        matte_provenance_block(
            modnet_entry=modnet_entry,
            lock=lock,
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
