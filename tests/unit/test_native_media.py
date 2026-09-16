from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms
from pydantic import ValidationError

from media_analysis.config import Settings
from media_analysis.errors import AnalyzeError
from media_analysis.features import image as measurements
from media_analysis.features.audio_pcm import AnalysisPcm
from media_analysis.features.rhythm import analyze_rhythm
from media_analysis.features.yunet import RawFaceDetection
from media_analysis.image_decode import decode_image
from media_analysis.request_hash import request_hash
from media_analysis.runtime import load_runtime
from media_analysis.schemas import AnalyzeRequest

pytestmark = pytest.mark.unit


def request(kind="image", features=None, **kwargs):
    return AnalyzeRequest(
        idempotencyKey="native",
        mediaKind=kind,
        source={
            "signedGetUrl": "http://localhost/test",
            "expiresAt": "2099-01-01T00:00:00Z",
            "sha256": "a" * 64,
        },
        features=features or ["quality"],
        **kwargs,
    )


@pytest.mark.parametrize(
    "kind,features", [("image", ["motion"]), ("audio", ["faces"]), ("video", ["focus"])]
)
def test_kind_rejects_incompatible_features(kind, features):
    with pytest.raises(ValidationError):
        request(kind, features)


def test_legacy_exact_digest_and_explicit_video():
    from tests.unit.test_request_hash import _base

    payload = _base()
    assert (
        request_hash(payload) == "425c0134ed1980983df903c39e0e803bd3eb39456b2b434efdd99d1b7ed14c46"
    )
    assert request_hash({**payload, "mediaKind": "video"}) == request_hash(payload)


def test_hash_new_kind_settings_and_grants():
    base = request().model_dump()
    assert request_hash(base) != request_hash({**base, "mediaKind": "audio"})
    assert request_hash(base) != request_hash(
        {**base, "imageOptions": {"alphaBackground": "black"}}
    )
    grant = {"signedPutUrl": "http://localhost/out?a", "expiresAt": "2099-01-01T00:00:00Z"}
    first = {**base, "outputGrants": {"canonicalImage": grant}}
    second = {
        **base,
        "outputGrants": {"canonicalImage": {**grant, "signedPutUrl": "http://localhost/out?b"}},
    }
    assert request_hash(first) == request_hash(second)
    assert request_hash(first) != request_hash(base)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_orientation_transform_matches_actual_pixels(tmp_path, orientation):
    pixels = np.zeros((8, 12, 3), np.uint8)
    pixels[1, 2] = [255, 0, 0]
    source = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = orientation
    path = tmp_path / "oriented.png"
    source.save(path, exif=exif)
    result = decode_image(path, request(), Settings(_env_file=None))
    matrix = np.array(result.metadata["originalToCanonicalNormalized"])
    transformed = matrix @ np.array([2.5 / 12, 1.5 / 8, 1])
    actual = np.asarray(result.canonical)
    y, x = np.unravel_index(actual[:, :, 0].argmax(), actual.shape[:2])
    assert [(x + 0.5) / actual.shape[1], (y + 0.5) / actual.shape[0]] == pytest.approx(
        transformed[:2]
    )
    assert result.canonical.getexif().get(274) is None


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "HEIF"])
def test_advertised_image_formats(tmp_path, fmt):
    path = tmp_path / "source"
    Image.new("RGB", (40, 20), "red").save(path, format=fmt)
    result = decode_image(path, request(), Settings(_env_file=None))
    assert result.metadata["width"] == 40
    assert result.metadata["sourceFormat"] == fmt


def test_alpha_and_icc_policy(tmp_path):
    path = tmp_path / "alpha.png"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGBA", (20, 10), (255, 0, 0, 0)).save(path, icc_profile=profile)
    result = decode_image(path, request(), Settings(_env_file=None))
    assert np.asarray(result.canonical).min() == 255
    result = decode_image(
        path, request(imageOptions={"alphaBackground": "black"}), Settings(_env_file=None)
    )
    assert np.asarray(result.canonical).max() == 0


def test_image_bomb_animation_and_bad_profile(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (30, 20)).save(path)
    with pytest.raises(AnalyzeError, match="pixel limit"):
        decode_image(path, request(), Settings(_env_file=None, media_analysis_max_image_pixels=100))
    Image.new("RGB", (30, 20)).save(
        path, save_all=True, append_images=[Image.new("RGB", (30, 20), "red")]
    )
    with pytest.raises(AnalyzeError, match="Animated"):
        decode_image(path, request(), Settings(_env_file=None))
    Image.new("RGB", (30, 20)).save(path, icc_profile=b"bad-profile")
    with pytest.raises(AnalyzeError, match="profile"):
        decode_image(path, request(), Settings(_env_file=None))


def test_image_quality_saliency_and_no_center_fallback():
    blank = np.zeros((128, 128, 3), np.uint8)
    assert measurements.saliency(blank)["regions"] == []
    assert measurements.focus(measurements.saliency(blank))["candidates"] == []
    edge = blank.copy()
    edge[30:70, 100:125] = 255
    found = measurements.saliency(edge)
    assert found["regions"][0]["point"]["x"] > 0.7
    assert found["scoreType"] == "heuristic"
    blurred = cv2.GaussianBlur(edge, (21, 21), 5)
    assert (
        measurements.quality(edge)["laplacianVariance"]
        > measurements.quality(blurred)["laplacianVariance"]
    )
    assert measurements.quality(blank)["darkClipFraction"] == 1
    assert measurements.quality(np.full_like(blank, 255))["brightClipFraction"] == 1


def test_faces_use_canonical_normalized_coordinates_without_tracks():
    class Detector:
        def set_input_size(self, w, h):
            assert (w, h) == (100, 50)

        def detect(self, frame):
            return [RawFaceDetection(50, 10, 20, 20, 0.9)]

    regions = measurements.regions(
        "faces", np.zeros((50, 100, 3), np.uint8), SimpleNamespace(face_detector=Detector())
    )
    assert regions[0]["box"] == {
        "x": 0.5,
        "y": 0.2,
        "width": pytest.approx(0.2),
        "height": pytest.approx(0.4),
    }
    assert "trackId" not in regions[0]


def click_pcm(times, duration=12, rate=16000):
    samples = np.zeros(int(duration * rate), np.float32)
    # Independent pulse generator, not the detector's hop grid.
    for time in times:
        start = round(time * rate)
        samples[start : start + 100] = np.hanning(200)[100:].astype(np.float32) * 0.8
    return AnalysisPcm(samples, rate, duration, True)


@pytest.mark.parametrize("bpm,phase", [(60, 0.137), (120, 0.273), (180, 1.017)])
def test_rhythm_click_count_and_timestamp_error(bpm, phase):
    expected = np.arange(phase, 11.5, 60 / bpm)
    result = analyze_rhythm(click_pcm(expected))
    actual = [event["timeSec"] for event in result["beats"]]
    assert len(actual) == len(expected)  # missing/extra events independently checked
    assert np.max(np.abs(np.array(actual) - expected)) <= 0.011
    assert result["bpm"] == pytest.approx(bpm, rel=0.02)
    assert all(event["sampleIndex"] / 16000 == event["timeSec"] for event in result["beats"])


def test_rhythm_silence_absence_irregular_missing_and_tempo_change():
    silence = click_pcm([])
    assert analyze_rhythm(silence)["reasons"] == ["SILENCE"]
    assert analyze_rhythm(replace(silence, has_audio=False))["reasons"] == ["AUDIO_ABSENT"]
    assert analyze_rhythm(click_pcm([0.1, 0.6, 1.7, 2.0, 4.8]))["beats"] == []
    times = np.r_[np.arange(0.3, 4, 0.5), np.arange(5, 11, 0.75)]
    result = analyze_rhythm(click_pcm(times))
    assert len(result["segments"]) == 2
    assert result["bpm"] is None
    times = np.delete(np.arange(0.3, 11.5, 0.5), 10)
    result = analyze_rhythm(click_pcm(times))
    assert len(result["beats"]) == len(times)
    assert all(abs(event["timeSec"] - 5.3) > 0.1 for event in result["beats"])


def test_audio_runtime_does_not_require_visual_models(tmp_path):
    cfg = Settings(
        _env_file=None,
        media_analysis_model_dir=tmp_path,
        media_analysis_worker_role="audio",
        media_analysis_enabled_features=["rhythm", "waveform"],
    )
    runtime = load_runtime(cfg)
    assert runtime.ready
    assert runtime.loaded_models == ()


def test_redelivery_registry_preserves_active_exclusion():
    from media_analysis.jobs import JobRegistry

    registry = JobRegistry()
    job, cached = registry.begin("key", "hash")
    assert cached is None
    registry.complete(job, {"overallStatus": "completed"})
    replay, cached = registry.begin("key", "hash", redeliver=True)
    assert replay is not job
    assert cached is None
    with pytest.raises(ValueError, match="active"):
        registry.begin("key", "hash", redeliver=True)
    with pytest.raises(ValueError, match="different"):
        registry.begin("key", "changed", redeliver=True)
    registry.fail(replay)
    retry, cached = registry.begin("key", "hash", redeliver=True)
    assert retry.active


@pytest.mark.parametrize(
    "field,value",
    [("duration", "nan"), ("duration", "-1"), ("sample_rate", "0"), ("channels", "0")],
)
def test_invalid_audio_metadata_is_decode_failure(tmp_path, monkeypatch, field, value):
    import json

    import media_analysis.audio_decode as module

    stream = {
        "codec_type": "audio",
        "codec_name": "flac",
        "duration": "1",
        "sample_rate": "16000",
        "channels": 1,
        field: value,
    }
    monkeypatch.setattr(
        module,
        "run_bounded_subprocess",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [stream]})),
    )
    with pytest.raises(AnalyzeError) as caught:
        module.probe_audio(
            tmp_path / "source", Settings(_env_file=None), SimpleNamespace(remaining=lambda: 10)
        )
    assert caught.value.code == "DECODE_FAILED"


def test_result_cache_pins_algorithm_identity(settings, monkeypatch):
    import media_analysis.features.rhythm as rhythm
    from media_analysis.analyze import _result_cache_key

    runtime = load_runtime(settings)
    req = request("audio", ["rhythm"])
    old = _result_cache_key(req, runtime, settings, "a" * 64)
    monkeypatch.setattr(rhythm, "RHYTHM_VERSION", "new-policy")
    assert _result_cache_key(req, runtime, settings, "a" * 64) != old


def test_no_face_detection_is_completed_empty_at_primitive():
    class Detector:
        def set_input_size(self, w, h):
            pass

        def detect(self, frame):
            return []

    assert (
        measurements.regions(
            "faces", np.zeros((10, 10, 3), np.uint8), SimpleNamespace(face_detector=Detector())
        )
        == []
    )


def test_sixteen_bit_png_is_not_silently_clipped(tmp_path):
    path = tmp_path / "high-depth.png"
    Image.fromarray(np.full((10, 10), 40000, dtype=np.uint16)).save(path)
    with pytest.raises(AnalyzeError, match="8-bit"):
        decode_image(path, request(), Settings(_env_file=None))


def test_nonfinite_decoded_audio_fails(tmp_path, monkeypatch):
    import media_analysis.features.audio_pcm as module

    monkeypatch.setattr(module.shutil, "which", lambda name: "/ffmpeg")
    monkeypatch.setattr(
        module,
        "run_bounded_subprocess",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=np.array([np.nan], np.float32).tobytes(), stderr=b""
        ),
    )
    with pytest.raises(AnalyzeError, match="non-finite"):
        module.extract_analysis_pcm(tmp_path / "source", has_audio=True)


@pytest.mark.parametrize("kind", ["image", "audio", "video"])
def test_recorded_worker_contract_branches(kind):
    import json
    from pathlib import Path

    from pydantic import TypeAdapter

    from media_analysis.native_schemas import NativeResult

    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "native_contracts" / f"{kind}.json").read_text()
    )
    req = AnalyzeRequest.model_validate(fixture["request"])
    assert req.mediaKind == kind
    result = fixture["result"]
    if kind == "video":
        assert result["schemaVersion"] == 1
        assert "mediaKind" not in result
        assert {"fps", "frameCount", "width", "height"} <= result["canonicalMedia"].keys()
        assert {"shots", "capabilities", "provenance", "requestedFeatures"} <= result.keys()
    else:
        TypeAdapter(NativeResult).validate_python(result)
        assert not {"fps", "frameCount"} & result["canonicalMedia"].keys()


@pytest.mark.parametrize(
    "feature,model,loader,attribute",
    [
        ("subjects", "yolox-tiny", "build_yolox_detector", "subject_detector"),
        ("faces", "yunet", "load_yunet_detector", "face_detector"),
        ("ocr", "PP-OCRv5_mobile_det", "create_det_session", "ocr_session"),
        ("audio", "silero-vad", "load_silero_vad_session", "vad_session"),
    ],
)
def test_configured_runtime_only_initializes_requested_models(
    settings, monkeypatch, feature, model, loader, attribute
):
    import json

    import media_analysis.runtime as runtime_module

    path = settings.media_analysis_model_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["models"] = [entry for entry in manifest["models"] if entry["name"] == model]
    manifest["models"][0]["stub"] = False
    path.write_text(json.dumps(manifest))
    calls = []
    detector = SimpleNamespace(detect=lambda frame: [])

    def load(path):
        calls.append(path)
        return detector

    monkeypatch.setattr(runtime_module, loader, load)
    monkeypatch.setattr(runtime_module, "run_det_inference", lambda *args: [])
    monkeypatch.setattr(runtime_module, "warmup_silero_vad", lambda *args: None)
    cfg = settings.model_copy(
        update={
            "media_analysis_enabled_features": [feature],
            "media_analysis_allow_stub_models": False,
        }
    )
    state = runtime_module.load_runtime(cfg)
    assert state.ready, state.errors
    assert state.loaded_models == (model,)
    assert getattr(state, attribute) is detector
    assert len(calls) == 1


def test_new_result_types_reject_temporal_image_and_audio_fields():
    from pydantic import ValidationError

    from media_analysis.native_schemas import DetectionRegion, Onset

    with pytest.raises(ValidationError):
        DetectionRegion(score=0.8, scoreType="raw_model", trackId="invented")
    with pytest.raises(ValidationError):
        Onset(sec=1, strength=0.5, sourceFrameApprox=30)
