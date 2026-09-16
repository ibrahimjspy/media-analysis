"""Kind-specific orchestration after shared download and SHA verification."""

from __future__ import annotations

import hashlib
import io
from typing import Any

from media_analysis.audio_decode import AUDIO_DECODE_VERSION, canonical_audio, probe_audio
from media_analysis.build_provenance import analyze_build_provenance
from media_analysis.errors import DECODE_FAILED, AnalyzeError
from media_analysis.features import image as image_features
from media_analysis.features.audio import analyze_audio, audio_provenance
from media_analysis.features.audio_dsp import pcm_has_measurable_signal
from media_analysis.features.audio_pcm import PCM_EXTRACTION_VERSION, extract_analysis_pcm
from media_analysis.features.rhythm import RHYTHM_VERSION, analyze_rhythm
from media_analysis.features.waveform import WAVEFORM_ANALYZER_VERSION, analyze_waveform
from media_analysis.image_decode import IMAGE_DECODE_VERSION, decode_image
from media_analysis.models_manifest import SILERO_VAD_MODEL_NAME, sha256_file
from media_analysis.native_schemas import AudioResult, ImageResult
from media_analysis.upload import upload_artifact

MODEL_FEATURES = {
    "subjects": ("yolox-tiny", "subject_detector"),
    "faces": ("yunet", "face_detector"),
    "ocr": ("PP-OCRv5_mobile_det", "ocr_session"),
}


def run_native(
    request, src, fingerprint, *, settings, runtime, job, deadline, telemetry, media_cache, work_dir
) -> dict[str, Any]:
    # Imported here to keep shared guards owned by the existing execution layer.
    from media_analysis.analyze import _feature_stage, _resolve_vad_session, _result_cache_key

    def stage(name):
        return _feature_stage(
            f"{request.mediaKind}.{name}",
            telemetry,
            job,
            deadline,
            settings.media_analysis_stage_timeout_sec,
        )

    key = _result_cache_key(request, runtime, settings, fingerprint)
    cached = media_cache.get_json("native-result", key)
    if cached is not None and cached.get("overallStatus") != "completed":
        cached = None
    bodies, capabilities, warnings, provenance = {}, {}, [], analyze_build_provenance()
    prepared_image = None
    playback = src
    pcm = None
    if request.mediaKind == "image":
        with stage("decode"):
            prepared_image = decode_image(src, request, settings)
        canonical = prepared_image.metadata
        telemetry.decoded_pixels = canonical["width"] * canonical["height"]
        provenance.update(
            {
                "imageDecodeVersion": IMAGE_DECODE_VERSION,
                "imageMeasurementVersion": image_features.IMAGE_MEASUREMENT_VERSION,
            }
        )
    else:
        with stage("probe") as guard:
            media = probe_audio(src, settings, guard)
        with stage("pcm") as guard:
            pcm = extract_analysis_pcm(
                src,
                has_audio=True,
                source_duration_sec=media.duration,
                source_sample_rate=media.audio_sample_rate,
                source_channel_count=media.audio_channel_count,
                max_duration_sec=settings.media_analysis_max_audio_duration_sec,
                timeout_sec=guard.remaining(),
                cancel_check=guard,
            )
        if not pcm.has_audio:
            raise AnalyzeError(DECODE_FAILED, "Audio stream decoded no samples")
        if request.canonicalize:
            playback = work_dir / "canonical.wav"
            with stage("canonicalize") as guard:
                canonical_audio(src, playback, media, settings, guard)
        canonical = {
            "kind": "audio",
            "durationSec": pcm.duration_sec,
            "sourceDurationSec": media.duration,
            "codec": "pcm_s16le" if request.canonicalize else media.audio_codec,
            "sampleRate": media.audio_sample_rate,
            "channels": media.audio_channel_count,
            "sourceCodec": media.audio_codec,
            "sourceStartTimeSec": media.start_time,
            "analysisSampleRate": pcm.sample_rate,
            "analysisSampleCount": int(pcm.samples.size),
            "analysisChannels": 1,
            "canonicalizationVersion": AUDIO_DECODE_VERSION,
            "clock": {
                "origin": "first-decoded-playback-sample",
                "analysisToPlaybackOffsetSec": 0.0,
                "decoderDelayPolicy": "ffmpeg-applies-container-skip-samples-and-padding",
                "sourcePtsOriginSec": media.start_time,
            },
        }
        if request.canonicalize:
            canonical["mimeType"] = "audio/wav"
            canonical["sha256"] = sha256_file(playback)
            canonical["byteCount"] = playback.stat().st_size
        provenance["audioDecodeVersion"] = AUDIO_DECODE_VERSION
        provenance["pcmExtractionVersion"] = PCM_EXTRACTION_VERSION
    canonical["sourceSha256"] = fingerprint
    if cached is not None:
        bodies = {k: v for k, v in cached.items() if k in request.features}
        capabilities = cached["capabilities"]
        warnings = cached["warningCodes"]
        provenance = cached["provenance"]
        telemetry.result_cache_hit = True
    else:
        saliency = None
        for feature in request.features:
            with stage(feature) as guard:
                try:
                    if request.mediaKind == "image":
                        bgr = prepared_image.bgr
                        if feature in MODEL_FEATURES:
                            name, attribute = MODEL_FEATURES[feature]
                            entry = runtime.manifest.by_name(name)
                            if entry is not None:
                                provenance[f"{feature}Model"] = {
                                    "name": entry.name,
                                    "sha256": entry.sha256,
                                    "version": entry.version,
                                    "referenceMode": entry.stub,
                                }
                            if getattr(runtime, attribute) is None:
                                capabilities[feature] = {
                                    "status": "unavailable",
                                    "warningCodes": [f"{feature.upper()}_UNAVAILABLE"],
                                }
                                warnings.append(f"{feature.upper()}_UNAVAILABLE")
                                continue
                            bodies[feature] = {
                                "regions": image_features.regions(feature, bgr, runtime)
                            }
                        elif feature in {"quality", "exposure"}:
                            bodies[feature] = getattr(image_features, feature)(bgr)
                        elif feature in {"saliency", "focus"}:
                            if saliency is None:
                                saliency = image_features.saliency(bgr)
                            bodies[feature] = (
                                saliency
                                if feature == "saliency"
                                else image_features.focus(saliency)
                            )
                            provenance["saliencyVersion"] = image_features.SALIENCY_VERSION
                            if feature == "focus":
                                provenance["focusVersion"] = image_features.FOCUS_VERSION
                        elif feature == "thumbnails":
                            bodies[feature] = []  # Rendering/delivery occurs on every request.
                    elif feature == "audio":
                        vad = _resolve_vad_session(runtime, runtime.manifest, settings)
                        measured = analyze_audio(
                            pcm,
                            source_path=src,
                            vad_session=vad,
                            require_vad=False,
                            timeout_sec=guard.remaining(),
                            cancel_check=guard,
                        )
                        bodies[feature] = measured.to_payload()
                        bodies[feature]["signalState"] = (
                            "present" if pcm_has_measurable_signal(pcm) else "silence"
                        )
                        bodies[feature]["speechState"] = (
                            "reference"
                            if runtime.reference_mode and vad is not None
                            else ("measured" if vad is not None else "unknown")
                        )
                        warnings.extend(measured.warning_codes)
                        provenance.update(
                            audio_provenance(
                                vad_entry=runtime.manifest.by_name(SILERO_VAD_MODEL_NAME),
                                vad_runtime="stub" if runtime.reference_mode else "onnxruntime-cpu",
                            )
                        )
                        if vad is None:
                            capabilities[feature] = {
                                "status": "partial",
                                "warningCodes": ["VAD_UNAVAILABLE"],
                            }
                            warnings.append("VAD_UNAVAILABLE")
                            continue
                    elif feature == "waveform":
                        bodies[feature], notices = analyze_waveform(pcm)
                        warnings.extend(notices)
                        provenance["waveformAnalyzerVersion"] = WAVEFORM_ANALYZER_VERSION
                    elif feature == "rhythm":
                        options = request.rhythmOptions
                        bodies[feature] = analyze_rhythm(
                            pcm,
                            min_bpm=options.minBpm if options else 50,
                            max_bpm=options.maxBpm if options else 200,
                            cancel_check=guard,
                        )
                        provenance["rhythmVersion"] = RHYTHM_VERSION
                    capabilities[feature] = {"status": "completed"}
                except AnalyzeError:
                    raise
                except Exception:
                    bodies.pop(feature, None)
                    warnings.append(f"{feature.upper()}_FAILED")
                    capabilities[feature] = {
                        "status": "failed",
                        "warningCodes": [f"{feature.upper()}_FAILED"],
                    }
    statuses = {item["status"] for item in capabilities.values()}
    overall = (
        "completed"
        if statuses == {"completed"}
        else ("partial" if statuses & {"completed", "partial"} else "failed")
    )
    result = {
        "schemaVersion": 2,
        "mediaKind": request.mediaKind,
        "overallStatus": overall,
        "requestedFeatures": request.features,
        "canonicalMedia": canonical,
        "capabilities": capabilities,
        **bodies,
        "provenance": provenance,
        "warningCodes": sorted(set(warnings)),
    }
    # Save measurements before delivery. Failed uploads can retry without inference.
    if cached is None and overall == "completed":
        media_cache.put_json("native-result", key, result)

    def deliver(data, grant, mime):
        with stage("upload") as guard:
            artifact = upload_artifact(
                data,
                grant.signedPutUrl,
                settings=settings,
                expires_at=grant.expiresAt,
                mime_type=mime,
                timeout_sec=min(guard.remaining(), settings.media_analysis_download_timeout_sec),
                cancel_check=guard,
            )
            telemetry.add_uploaded_bytes(artifact["byteCount"])
            return artifact

    grants = request.outputGrants
    if prepared_image:
        with stage("encode"):
            image_bytes = prepared_image.png()
        canonical["sha256"] = hashlib.sha256(image_bytes).hexdigest()
        canonical["byteCount"] = len(image_bytes)
        if grants and grants.canonicalImage:
            canonical.update(deliver(image_bytes, grants.canonicalImage, "image/png"))
        if "thumbnails" in request.features:
            from PIL import Image

            thumb = prepared_image.canonical.copy()
            thumb.thumbnail((480, 480), Image.Resampling.LANCZOS)
            target = io.BytesIO()
            thumb.save(target, format="JPEG", quality=85)
            artifact = deliver(target.getvalue(), grants.thumbnails[0], "image/jpeg")
            result["thumbnails"] = [
                {"index": 0, "width": thumb.width, "height": thumb.height, **artifact}
            ]
    elif grants and grants.canonicalAudio:
        canonical.update(deliver(playback.read_bytes(), grants.canonicalAudio, "audio/wav"))
    result["telemetry"] = {
        **telemetry.as_dict(),
        "mediaKind": request.mediaKind,
        "decodedSamples": int(pcm.samples.size) if pcm else 0,
    }
    result_type = ImageResult if request.mediaKind == "image" else AudioResult
    return result_type.model_validate(result).model_dump(exclude_unset=True)
