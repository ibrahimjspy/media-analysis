"""One discovery model shared by /capabilities and /ready."""

from __future__ import annotations

from media_analysis.config import AUDIO_FEATURES, CPU_FEATURES, FEATURES_BY_KIND, MATTE_FEATURES

FEATURE_MODELS = {
    "subjects": "yolox-tiny",
    "faces": "yunet",
    "ocr": "PP-OCRv5_mobile_det",
    "audio": "silero-vad",
    "person_matte": "modnet",
}


def configured_features(settings):
    role = settings.worker_role
    if role == "matte":
        features = MATTE_FEATURES
    elif role == "audio":
        features = AUDIO_FEATURES
    elif role == "ocr":
        features = frozenset({"ocr", "shots"})
    elif role == "general":
        features = CPU_FEATURES - {"ocr"}
    else:
        features = CPU_FEATURES
    if settings.media_analysis_enabled_features is not None:
        features = features & set(settings.media_analysis_enabled_features)
    return features


def capability_document(settings, runtime):
    from media_analysis.audio_decode import AUDIO_CODECS, AUDIO_DECODE_VERSION
    from media_analysis.features.image import FOCUS_VERSION, SALIENCY_VERSION
    from media_analysis.features.rhythm import RHYTHM_VERSION
    from media_analysis.image_decode import IMAGE_DECODE_VERSION, IMAGE_FORMATS

    configured = configured_features(settings)
    kinds = {}
    for kind, features in FEATURES_BY_KIND.items():
        kinds[kind] = {"configured": bool(features & configured), "features": {}}
        for feature in sorted(features):
            model = FEATURE_MODELS.get(feature)
            model_ready = model is None or model in runtime.loaded_models
            if kind == "image" and model:
                attribute = {
                    "subjects": "subject_detector",
                    "faces": "face_detector",
                    "ocr": "ocr_session",
                }.get(feature)
                model_ready = attribute is not None and getattr(runtime, attribute) is not None
            if feature == "visual_regions":
                from media_analysis.features.visual_regions import available

                model_ready = available(settings.media_analysis_visual_enabled)
            kinds[kind]["features"][feature] = {
                "implemented": True,
                "configured": feature in configured,
                "requiredModels": [model] if model else [],
                "modelReady": model_ready,
                "available": feature in configured and runtime.ready and model_ready,
                "referenceMode": runtime.reference_mode,
                # This endpoint is not an output-quality certification.
                "productionQualified": False,
            }
    return {
        "schemaVersion": 1,
        "mediaKinds": kinds,
        "inputPolicy": {
            "maxBytes": settings.media_analysis_max_bytes,
            "image": {
                "formats": sorted(IMAGE_FORMATS),
                "animated": False,
                "maxChannelBitDepth": 8,
                "heifColorPolicy": "8-bit SDR sRGB or valid ICC; HDR rejected",
                "maxDecodedPixels": settings.media_analysis_max_image_pixels,
                "maxAnalysisDimension": settings.media_analysis_image_max_dimension,
            },
            "audio": {
                "codecs": sorted(AUDIO_CODECS),
                "maxDurationSec": settings.media_analysis_max_audio_duration_sec,
                "maxChannels": settings.media_analysis_max_audio_channels,
                "maxSampleRate": settings.media_analysis_max_audio_sample_rate,
            },
            "video": {
                "decoder": "installed-ffmpeg",
                "maxSourceWidth": settings.media_analysis_max_width,
                "maxSourceHeight": settings.media_analysis_max_height,
                "maxSourceDurationSec": settings.media_analysis_max_duration_sec,
                "maxDecodedPixels": settings.media_analysis_max_decoded_pixels,
                "limitsApplyBeforeNormalization": True,
            },
        },
        "canonicalArtifacts": {"image": "image/png", "audio": "audio/wav", "video": "video/mp4"},
        "algorithms": {
            "imageDecode": IMAGE_DECODE_VERSION,
            "audioDecode": AUDIO_DECODE_VERSION,
            "saliency": SALIENCY_VERSION,
            "focus": FOCUS_VERSION,
            "visualRegions": "home-tour-visual-1:12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
            "rhythm": RHYTHM_VERSION,
        },
        "matte": {
            "targetModes": ["all_people"],
            "segmentedRanges": False,
            "productionQualified": False,
        },
    }
