from __future__ import annotations

import hashlib
import io
import subprocess
import threading
import wave
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from tests.unit.test_native_media import click_pcm

import media_analysis.analyze as analyze_module
from media_analysis.app import create_app
from media_analysis.features import image as image_features
from media_analysis.jobs import registry

pytestmark = [pytest.mark.e2e, pytest.mark.ffmpeg]
EXPIRY = "2099-01-01T00:00:00Z"


@contextmanager
def media_server(data):
    uploads = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_PUT(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            uploads.append((self.path, body, self.headers["Content-Type"]))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", uploads
    finally:
        server.shutdown()
        thread.join()


def payload(url, data, kind, features, **kwargs):
    return {
        "idempotencyKey": "native",
        "mediaKind": kind,
        "features": features,
        "source": {
            "signedGetUrl": url,
            "expiresAt": EXPIRY,
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        **kwargs,
    }


def png_bytes():
    output = io.BytesIO()
    image = Image.new("RGB", (64, 32), "black")
    image.paste("white", (40, 10, 60, 25))
    image.save(output, format="PNG")
    return output.getvalue()


def wav_bytes():
    pcm = click_pcm(np.arange(0.273, 5.8, 0.5), duration=6)
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        stereo = np.column_stack([pcm.samples, pcm.samples * 0.5])
        writer.writeframes((stereo * 32767).astype("<i2").tobytes())
    return output.getvalue()


def test_image_http_artifacts_replay_and_restart(settings, auth_headers, tmp_path, monkeypatch):
    cfg = settings.model_copy(update={"media_analysis_cache_dir": tmp_path / "cache"})
    data = png_bytes()
    calls = []
    original = image_features.quality

    def measured(frame):
        calls.append(1)
        return original(frame)

    monkeypatch.setattr(image_features, "quality", measured)
    with media_server(data) as (url, uploads), TestClient(create_app(cfg)) as client:

        def grant(path):
            return {"signedPutUrl": url + path, "expiresAt": EXPIRY}

        request = payload(
            url,
            data,
            "image",
            ["quality", "exposure", "focus", "saliency", "thumbnails"],
            canonicalize=True,
            outputGrants={
                "canonicalImage": grant("/one"),
                "thumbnails": [{"index": 0, **grant("/thumb-one")}],
            },
        )
        first = client.post("/analyze", json=request, headers=auth_headers)
        assert first.status_code == 200, first.text
        result = first.json()
        assert result["overallStatus"] == "completed", result
        assert result["canonicalMedia"]["width"] == 64
        assert "fps" not in result["canonicalMedia"]
        assert "sourceFrame" not in result["thumbnails"][0]
        request["outputGrants"] = {
            "canonicalImage": grant("/two"),
            "thumbnails": [{"index": 0, **grant("/thumb-two")}],
        }
        second = client.post("/analyze", json=request, headers=auth_headers)
        assert second.status_code == 200, second.text
        assert second.json()["telemetry"]["resultCacheHit"]
        assert len(calls) == 1
        assert [item[0] for item in uploads] == ["/one", "/thumb-one", "/two", "/thumb-two"]
        assert hashlib.sha256(uploads[0][1]).hexdigest() == result["canonicalMedia"]["sha256"]
        assert uploads[0][2] == "image/png"
        assert uploads[1][2] == "image/jpeg"
        registry.clear()  # Caller retry after worker-local job state is lost.
        third = client.post("/analyze", json=request, headers=auth_headers)
        assert third.status_code == 200
        assert len(calls) == 1
        assert len(uploads) == 6


def test_standalone_audio_real_decode_clock_and_stereo_artifact(settings, auth_headers, tmp_path):
    data = wav_bytes()
    cfg = settings.model_copy(update={"media_analysis_cache_dir": tmp_path / "cache"})
    with media_server(data) as (url, uploads), TestClient(create_app(cfg)) as client:
        request = payload(
            url,
            data,
            "audio",
            ["audio", "waveform", "rhythm"],
            canonicalize=True,
            outputGrants={"canonicalAudio": {"signedPutUrl": url + "/out", "expiresAt": EXPIRY}},
        )
        response = client.post("/analyze", json=request, headers=auth_headers)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["overallStatus"] == "completed", result
        assert result["canonicalMedia"]["analysisSampleCount"] == 96000
        assert result["canonicalMedia"]["durationSec"] == 6
        assert not {"width", "height", "fps", "frameCount"} & result["canonicalMedia"].keys()
        assert all("sourceFrame" not in event for event in result["audio"]["onsets"])
        expected = np.arange(0.273, 5.8, 0.5)
        beats = result["rhythm"]["beats"]
        assert len(beats) == len(expected)
        assert (
            max(abs(event["timeSec"] - time) for event, time in zip(beats, expected, strict=True))
            < 0.011
        )
        with wave.open(io.BytesIO(uploads[0][1])) as reader:
            assert reader.getnchannels() == 2
            assert reader.getnframes() == 96000
        assert uploads[0][2] == "audio/wav"


def test_image_unavailable_model_is_not_completed_empty(settings, auth_headers):
    data = png_bytes()
    with media_server(data) as (url, _), TestClient(create_app(settings)) as client:
        response = client.post(
            "/analyze",
            headers=auth_headers,
            json=payload(url, data, "image", ["quality", "faces", "ocr"]),
        )
        result = response.json()
        assert response.status_code == 200, result
        assert result["overallStatus"] == "partial"
        assert result["capabilities"]["faces"]["status"] == "unavailable"
        assert "faces" not in result
        assert "ocr" not in result


@pytest.mark.parametrize(
    "kind,data,features",
    [
        ("audio", png_bytes(), ["waveform"]),
        ("image", wav_bytes(), ["quality"]),
        ("video", png_bytes(), ["shots"]),
    ],
)
def test_decoded_kind_mismatch_fails(settings, auth_headers, kind, data, features):
    with media_server(data) as (url, _), TestClient(create_app(settings)) as client:
        response = client.post(
            "/analyze", headers=auth_headers, json=payload(url, data, kind, features)
        )
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "DECODE_FAILED"


def test_bad_hash_and_missing_hash_fail(settings, auth_headers):
    data = png_bytes()
    with media_server(data) as (url, _), TestClient(create_app(settings)) as client:
        request = payload(url, data, "image", ["quality"])
        request["source"]["sha256"] = "0" * 64
        response = client.post("/analyze", json=request, headers=auth_headers)
        assert response.json()["code"] == "CHECKSUM_MISMATCH"
        del request["source"]["sha256"]
        response = client.post("/analyze", json=request, headers=auth_headers)
        assert response.json()["code"] == "INVALID_REQUEST"


@pytest.mark.parametrize("has_audio", [True, False])
def test_video_rhythm_only_without_canonicalization(settings, auth_headers, tmp_path, has_audio):
    path = tmp_path / "video.mp4"
    args = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=s=32x32:d=2:r=30"]
    if has_audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=400:duration=2", "-c:a", "aac"]
    subprocess.run([*args, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    data = path.read_bytes()
    with media_server(data) as (url, _), TestClient(create_app(settings)) as client:
        response = client.post(
            "/analyze",
            headers=auth_headers,
            json=payload(url, data, "video", ["rhythm"], canonicalize=False),
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["capabilities"]["rhythm"]["status"] == "completed"
        assert result["canonicalMedia"]["hasAudio"] == has_audio
        assert ("AUDIO_ABSENT" in result["rhythm"]["reasons"]) != has_audio
        assert "audio" not in result


def test_discovery_matches_readiness_and_configured_audio(settings, auth_headers, tmp_path):
    cfg = settings.model_copy(
        update={
            "media_analysis_worker_role": "audio",
            "media_analysis_enabled_features": ["rhythm", "waveform"],
            "media_analysis_model_dir": tmp_path,
        }
    )
    with TestClient(create_app(cfg)) as client:
        ready = client.get("/ready").json()
        assert ready["ready"]
        assert ready["loadedModels"] == []
        discovery = client.get("/capabilities").json()
        assert ready["capabilities"] == discovery
        assert discovery["mediaKinds"]["audio"]["features"]["rhythm"]["available"]
        assert not discovery["mediaKinds"]["image"]["configured"]


def test_native_cancel_before_decode(settings, tmp_path):
    from media_analysis.errors import AnalyzeError
    from media_analysis.jobs import Job
    from media_analysis.runtime import load_runtime
    from media_analysis.schemas import AnalyzeRequest

    job = Job("cancel", "digest")
    job.cancel.set()
    request = AnalyzeRequest.model_validate(
        payload("http://localhost/never", b"", "image", ["quality"])
    )
    with pytest.raises(AnalyzeError, match="cancelled"):
        analyze_module.run_analyze(
            request, settings=settings, runtime=load_runtime(settings), job=job
        )


@pytest.mark.parametrize(
    "codec,extension",
    [
        ("pcm_s16le", "wav"),
        ("pcm_s24le", "wav"),
        ("pcm_s32le", "wav"),
        ("pcm_f32le", "wav"),
        ("flac", "flac"),
        ("libmp3lame", "mp3"),
        ("aac", "m4a"),
        ("libopus", "ogg"),
        ("libvorbis", "ogg"),
        ("alac", "m4a"),
    ],
)
def test_advertised_audio_codecs_and_decoder_delay(
    settings, auth_headers, tmp_path, codec, extension
):
    original = tmp_path / "original.wav"
    original.write_bytes(wav_bytes())
    encoded = tmp_path / f"encoded.{extension}"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(original), "-c:a", codec, str(encoded)],
        check=True,
    )
    data = encoded.read_bytes()
    with media_server(data) as (url, _), TestClient(create_app(settings)) as client:
        response = client.post(
            "/analyze",
            headers=auth_headers,
            json=payload(url, data, "audio", ["rhythm", "waveform"]),
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["overallStatus"] == "completed", result
        beats = result["rhythm"]["beats"]
        expected = np.arange(0.273, 5.8, 0.5)
        assert len(beats) == len(expected)
        assert (
            max(abs(event["timeSec"] - time) for event, time in zip(beats, expected, strict=True))
            < 0.025
        )


def test_failed_upload_retries_measurements_and_expired_renewal_fails(
    settings, auth_headers, tmp_path, monkeypatch
):
    data = png_bytes()
    cfg = settings.model_copy(update={"media_analysis_cache_dir": tmp_path / "cache"})
    calls = []
    original = image_features.quality

    def measured(frame):
        calls.append(1)
        return original(frame)

    monkeypatch.setattr(image_features, "quality", measured)
    with media_server(data) as (url, uploads), TestClient(create_app(cfg)) as client:
        request = payload(
            url,
            data,
            "image",
            ["quality"],
            canonicalize=True,
            outputGrants={
                "canonicalImage": {
                    "signedPutUrl": url + "/out",
                    "expiresAt": "2000-01-01T00:00:00Z",
                }
            },
        )
        response = client.post("/analyze", json=request, headers=auth_headers)
        assert response.status_code == 502
        request["outputGrants"]["canonicalImage"]["expiresAt"] = EXPIRY
        response = client.post("/analyze", json=request, headers=auth_headers)
        assert response.status_code == 200, response.text
        assert response.json()["telemetry"]["resultCacheHit"]
        assert len(calls) == 1
        assert len(uploads) == 1
        request["outputGrants"]["canonicalImage"]["expiresAt"] = "2000-01-01T00:00:00Z"
        response = client.post("/analyze", json=request, headers=auth_headers)
        assert response.status_code == 502  # Cached success must not claim new delivery.
        assert len(uploads) == 1


def test_video_canonical_grant_replayed_to_current_destination(
    settings, auth_headers, source_server, tmp_path
):
    data = source_server["path"].read_bytes()
    cfg = settings.model_copy(update={"media_analysis_cache_dir": tmp_path / "cache"})
    with media_server(data) as (url, uploads), TestClient(create_app(cfg)) as client:
        request = payload(
            url,
            data,
            "video",
            ["shots"],
            canonicalize=True,
            outputGrants={"canonicalMp4": {"signedPutUrl": url + "/first", "expiresAt": EXPIRY}},
        )
        first = client.post("/analyze", json=request, headers=auth_headers)
        assert first.status_code == 200, first.text
        request["outputGrants"]["canonicalMp4"]["signedPutUrl"] = url + "/second"
        second = client.post("/analyze", json=request, headers=auth_headers)
        assert second.status_code == 200, second.text
        assert second.json()["telemetry"]["resultCacheHit"]
        assert [item[0] for item in uploads] == ["/first", "/second"]
        assert uploads[0][1] == uploads[1][1]


def test_limits_enforced_again_on_native_cache_hit(settings, auth_headers, tmp_path):
    from media_analysis.app import reset_manifest

    data = png_bytes()
    cfg = settings.model_copy(update={"media_analysis_cache_dir": tmp_path / "cache"})
    with media_server(data) as (url, _):
        request = payload(url, data, "image", ["quality"])
        with TestClient(create_app(cfg)) as client:
            assert client.post("/analyze", json=request, headers=auth_headers).status_code == 200
        registry.clear()
        reset_manifest()
        cfg = cfg.model_copy(update={"media_analysis_max_image_pixels": 100})
        with TestClient(create_app(cfg)) as client:
            response = client.post("/analyze", json=request, headers=auth_headers)
            assert response.status_code == 413


def test_cached_source_checksum_is_reverified(settings, auth_headers, tmp_path):
    data = png_bytes()
    cfg = settings.model_copy(update={"media_analysis_cache_dir": tmp_path / "cache"})
    with media_server(data) as (url, _), TestClient(create_app(cfg)) as client:
        request = payload(url, data, "image", ["quality"])
        assert client.post("/analyze", json=request, headers=auth_headers).status_code == 200
        source_path = next((tmp_path / "cache" / "source").glob("*.media"))
        source_path.write_bytes(b"corrupted")
        request["idempotencyKey"] = "retry"
        response = client.post("/analyze", json=request, headers=auth_headers)
        assert response.status_code == 422
        assert response.json()["code"] == "CHECKSUM_MISMATCH"


def test_native_detector_empty_success_failure_and_coordinates(settings, auth_headers, monkeypatch):
    from dataclasses import replace

    import media_analysis.app as app_module
    from media_analysis.features.yunet import RawFaceDetection
    from media_analysis.runtime import load_runtime

    class FaceDetector:
        def set_input_size(self, w, h):
            pass

        def detect(self, frame):
            return [RawFaceDetection(32, 8, 16, 16, 0.8)]

    class SubjectDetector:
        def detect_normalized(self, frame):
            return [], None

    state = replace(
        load_runtime(settings), face_detector=FaceDetector(), subject_detector=SubjectDetector()
    )
    monkeypatch.setattr(app_module, "_runtime", state)
    data = png_bytes()
    cfg = settings.model_copy(update={"media_analysis_cache_enabled": False})
    with media_server(data) as (url, _), TestClient(create_app(cfg)) as client:
        request = payload(url, data, "image", ["subjects", "faces"])
        response = client.post("/analyze", json=request, headers=auth_headers)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["overallStatus"] == "completed"
        assert result["subjects"] == {"regions": []}
        assert result["faces"]["regions"][0]["box"] == {
            "x": 0.5,
            "y": 0.25,
            "width": 0.25,
            "height": 0.5,
        }

        def broken(frame):
            raise RuntimeError("detector broke")

        state.face_detector.detect = broken
        request["idempotencyKey"] = "broken"
        response = client.post("/analyze", json=request, headers=auth_headers)
        result = response.json()
        assert result["overallStatus"] == "partial"
        assert result["capabilities"]["faces"]["status"] == "failed"
        assert "faces" not in result


def test_native_audio_duration_bound(settings, auth_headers):
    data = wav_bytes()
    cfg = settings.model_copy(update={"media_analysis_max_audio_duration_sec": 1})
    with media_server(data) as (url, _), TestClient(create_app(cfg)) as client:
        response = client.post(
            "/analyze", headers=auth_headers, json=payload(url, data, "audio", ["waveform"])
        )
        assert response.status_code == 413, response.text


def test_native_stage_timeout_cannot_cache_success(settings, auth_headers, tmp_path, monkeypatch):
    import time

    data = png_bytes()
    cfg = settings.model_copy(
        update={
            "media_analysis_cache_dir": tmp_path / "cache",
            "media_analysis_stage_timeout_sec": 0.2,
        }
    )
    original = image_features.quality

    def slow(frame):
        time.sleep(0.25)
        return original(frame)

    monkeypatch.setattr(image_features, "quality", slow)
    with media_server(data) as (url, _), TestClient(create_app(cfg)) as client:
        response = client.post(
            "/analyze", headers=auth_headers, json=payload(url, data, "image", ["quality"])
        )
        assert response.status_code == 504, response.text
        assert not list((tmp_path / "cache" / "native-result").glob("*.json"))


def test_heif_cannot_enter_video_path(settings, auth_headers):
    data = io.BytesIO()
    Image.new("RGB", (64, 32), "red").save(data, format="HEIF")
    data = data.getvalue()
    with media_server(data) as (url, _), TestClient(create_app(settings)) as client:
        response = client.post(
            "/analyze", headers=auth_headers, json=payload(url, data, "video", ["shots"])
        )
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "DECODE_FAILED"


def test_visual_regions_native_boundary(settings, auth_headers, tmp_path, monkeypatch):
    """New optional evidence survives HTTP serialization with canonical geometry."""
    from media_analysis.features import visual_regions

    cfg = settings.model_copy(update={"media_analysis_cache_dir": tmp_path / "visual-cache"})
    data = png_bytes()
    monkeypatch.setattr(visual_regions, "available", lambda enabled: True)
    monkeypatch.setattr(
        visual_regions,
        "detect",
        lambda frame, guard: {
            "policyVersion": "home-tour-visual-1",
            "modelRevision": "test",
            "regions": [
                {
                    "id": "region-0",
                    "label": "sofa",
                    "score": 0.7,
                    "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                }
            ],
        },
    )
    with media_server(data) as (url, uploads), TestClient(create_app(cfg)) as client:
        request = payload(url, data, "image", ["quality", "visual_regions"])
        response = client.post("/analyze", json=request, headers=auth_headers)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["capabilities"]["visual_regions"]["status"] == "completed"
        assert result["visual_regions"]["regions"][0]["box"]["x"] == 0.1


def test_visual_regions_unavailable_is_not_empty_success(
    settings, auth_headers, tmp_path, monkeypatch
):
    """A missing optional detector preserves quality and reports unavailable."""
    from media_analysis.features import visual_regions

    cfg = settings.model_copy(update={"media_analysis_cache_dir": tmp_path / "visual-off"})
    monkeypatch.setattr(visual_regions, "available", lambda enabled: False)
    data = png_bytes()
    with media_server(data) as (url, uploads), TestClient(create_app(cfg)) as client:
        response = client.post(
            "/analyze",
            json=payload(url, data, "image", ["quality", "visual_regions"]),
            headers=auth_headers,
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["overallStatus"] == "partial"
        assert result["capabilities"]["visual_regions"]["status"] == "unavailable"
        assert result.get("visual_regions") is None


def test_neural_failure_retains_legacy_rhythm(settings, auth_headers, tmp_path, monkeypatch):
    """Optional inference failure is partial evidence, never erased legacy data."""
    from dataclasses import replace

    import media_analysis.app as app_module

    original = app_module.load_runtime

    class BrokenPredictor:
        def analyze(self, *args):
            raise ValueError("test model failure")

    monkeypatch.setattr(
        app_module,
        "load_runtime",
        lambda cfg: replace(original(cfg), neural_beats=BrokenPredictor()),
    )
    cfg = settings.model_copy(update={"media_analysis_cache_dir": tmp_path / "neural-fail"})
    data = wav_bytes()
    with media_server(data) as (url, uploads), TestClient(create_app(cfg)) as client:
        request = payload(url, data, "audio", ["rhythm"], rhythmOptions={"neuralBeats": True})
        result = client.post("/analyze", json=request, headers=auth_headers)
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["capabilities"]["rhythm"]["status"] == "partial"
        assert body["rhythm"]["neural"]["status"] == "failed"
        assert body["rhythm"]["sampleRate"] == 16000
        assert body["rhythm"]["beats"]  # The metronome remains measured.
        assert body["rhythm"]["neural"]["beats"] == []


def test_neural_serialization_empty_and_omitted(settings, auth_headers, tmp_path, monkeypatch):
    """Completed empty neural output is explicit; unrequested output is omitted."""
    from dataclasses import replace

    import media_analysis.app as app_module
    from media_analysis.features.beat_this import outcome

    original = app_module.load_runtime

    class EmptyPredictor:
        def analyze(self, *args):
            return outcome("completed")

    monkeypatch.setattr(
        app_module,
        "load_runtime",
        lambda cfg: replace(original(cfg), neural_beats=EmptyPredictor()),
    )
    cfg = settings.model_copy(update={"media_analysis_cache_dir": tmp_path / "neural-empty"})
    data = wav_bytes()
    with media_server(data) as (url, uploads), TestClient(create_app(cfg)) as client:
        first = payload(url, data, "audio", ["rhythm"])
        normal = client.post("/analyze", json=first, headers=auth_headers).json()
        assert "neural" not in normal["rhythm"]
        first["idempotencyKey"] = "neural"
        first["rhythmOptions"] = {"neuralBeats": True}
        response = client.post("/analyze", json=first, headers=auth_headers)
        assert response.status_code == 200, response.text
        assert response.json()["rhythm"]["neural"]["status"] == "completed"
        assert response.json()["rhythm"]["neural"]["beats"] == []
