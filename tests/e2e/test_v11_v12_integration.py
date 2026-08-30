from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient
from tests.conftest import TEST_KEY, write_stub_models

from media_analysis.app import create_app, reset_manifest
from media_analysis.config import Settings, reset_settings
from media_analysis.jobs import registry


class _PutHandler(BaseHTTPRequestHandler):
    uploads: list[tuple[str, bytes, str]] = []

    def do_PUT(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).uploads.append((self.path, body, self.headers.get("Content-Type", "")))
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        return


@pytest.fixture
def put_server(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setenv("MEDIA_ANALYSIS_ALLOWED_HOSTS", "127.0.0.1,localhost")
    reset_settings()
    _PutHandler.uploads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PutHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield {
        "url": f"http://{host}:{port}/upload?X-Amz-Signature=fake",
        "uploads": _PutHandler.uploads,
    }
    server.shutdown()
    thread.join(timeout=2)


def _shot(start: int, end: int) -> dict:
    return {
        "startFrame": start,
        "endFrameExclusive": end,
        "startSecApprox": start / 30.0,
        "endSecApprox": end / 30.0,
        "boundaryKind": "start" if start == 0 else "hard_cut",
        "classification": "other",
    }


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_combined_v11_v12_features_complete(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json={
            "idempotencyKey": "v11-v12-combined",
            "source": {"signedGetUrl": source_server["url"], "expiresAt": future_expiry},
            "features": ["shots", "motion", "quality", "exposure", "waveform"],
            "analysisResolution": {"width": 160, "height": 120},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["overallStatus"] == "completed"
    assert body["motion"]["samples"]
    assert body["quality"]["perShot"]
    assert body["exposure"]["perShot"]
    assert "min" in body["waveform"]
    assert body["capabilities"]["waveform"]["status"] == "completed"


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_fill_only_exposure_uses_prior_facts(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    first = client.post(
        "/analyze",
        headers=auth_headers,
        json={
            "idempotencyKey": "fill-shots",
            "source": {"signedGetUrl": source_server["url"], "expiresAt": future_expiry},
            "features": ["shots"],
            "analysisResolution": {"width": 160, "height": 120},
        },
    )
    shots = first.json()["shots"]
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json={
            "idempotencyKey": "fill-exposure",
            "source": {"signedGetUrl": source_server["url"], "expiresAt": future_expiry},
            "features": ["exposure"],
            "priorFacts": {"shots": shots},
            "analysisResolution": {"width": 160, "height": 120},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "shots" not in body
    assert body["exposure"]["perShot"]
    assert body["capabilities"]["exposure"]["status"] == "completed"


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_fresh_per_shot_feature_runs_internal_shot_detection(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "media_analysis.analyze.analyze_shots",
        lambda _path, _media: [_shot(0, 15), _shot(15, 30)],
    )
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json={
            "idempotencyKey": "fresh-exposure-internal-shots",
            "source": {"signedGetUrl": source_server["url"], "expiresAt": future_expiry},
            "features": ["exposure"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "shots" not in body
    assert len(body["exposure"]["perShot"]) == 2


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_thumbnails_uploads_once_with_grant(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    put_server: dict,
    future_expiry: str,
) -> None:
    shots = [_shot(0, 30)]
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json={
            "idempotencyKey": "thumb-upload",
            "source": {"signedGetUrl": source_server["url"], "expiresAt": future_expiry},
            "features": ["thumbnails"],
            "priorFacts": {"shots": shots},
            "outputGrants": {
                "thumbnails": [
                    {
                        "index": 0,
                        "signedPutUrl": put_server["url"],
                        "expiresAt": future_expiry,
                    }
                ]
            },
            "analysisResolution": {"width": 160, "height": 120},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["thumbnailCandidates"]
    assert len(put_server["uploads"]) == 1
    _, payload, content_type = put_server["uploads"][0]
    assert content_type == "image/jpeg"
    assert body["thumbnailCandidates"][0]["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_silent_video_audio_and_waveform_warn_audio_absent(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json={
            "idempotencyKey": "silent-audio",
            "source": {"signedGetUrl": source_server["url"], "expiresAt": future_expiry},
            "features": ["audio", "waveform"],
            "analysisResolution": {"width": 160, "height": 120},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["audio"]["durationSec"] == 0.0
    assert body["waveform"]["min"] == []
    assert "AUDIO_ABSENT" in body["warningCodes"]


@pytest.mark.e2e
def test_fill_only_exposure_without_prior_shots_is_invalid(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json={
            "idempotencyKey": "bad-fill",
            "source": {"signedGetUrl": source_server["url"], "expiresAt": future_expiry},
            "features": ["exposure"],
            "priorFacts": {
                "subjects": [
                    {
                        "trackId": "0-1",
                        "samples": [
                            {
                                "sourceFrame": 0,
                                "presentationTimeSecApprox": 0.0,
                                "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                                "sampleKind": "detected",
                                "scoreType": "raw_model",
                                "occluded": False,
                            }
                        ],
                    }
                ]
            },
        },
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


@pytest.fixture
def matte_client(model_dir, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    matte_dir = model_dir.parent / "matte-models"
    matte_dir.mkdir(parents=True, exist_ok=True)
    blob = b"stub-modnet"
    (matte_dir / "modnet.onnx").write_bytes(blob)
    (matte_dir / "manifest.json").write_text(
        (
            '{"models":[{"name":"modnet","file":"modnet.onnx","sha256":"'
            + hashlib.sha256(blob).hexdigest()
            + '","license":"Apache-2.0","version":"stub","stub":true}]}'
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEDIA_ANALYSIS_KEY", TEST_KEY)
    monkeypatch.setenv("MEDIA_ANALYSIS_ALLOWED_HOSTS", "127.0.0.1,localhost")
    monkeypatch.setenv("MEDIA_ANALYSIS_MODEL_DIR", str(matte_dir))
    monkeypatch.setenv("MEDIA_ANALYSIS_IMAGE", "matte-cpu")
    monkeypatch.setenv("MEDIA_ANALYSIS_ALLOW_STUB_MODELS", "true")
    monkeypatch.setenv("MEDIA_ANALYSIS_MATTE_REFERENCE_MODE", "1")
    reset_settings()
    reset_manifest()
    registry.clear()
    return TestClient(create_app(Settings()))


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_matte_reference_mode_completes_with_upload(
    matte_client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    put_server: dict,
    future_expiry: str,
) -> None:
    probe = source_server
    shots_response = matte_client.post(
        "/analyze",
        headers=auth_headers,
        json={
            "idempotencyKey": "matte-shots-src",
            "source": {"signedGetUrl": probe["url"], "expiresAt": future_expiry},
            "features": ["person_matte"],
            "matteTarget": {"mode": "all_people"},
            "priorFacts": {
                "shots": [
                    {
                        "startFrame": 0,
                        "endFrameExclusive": 30,
                        "startSecApprox": 0.0,
                        "endSecApprox": 1.0,
                        "boundaryKind": "start",
                        "classification": "other",
                    }
                ]
            },
            "outputGrants": {
                "matteAssets": [
                    {
                        "label": "full",
                        "signedPutUrl": put_server["url"],
                        "expiresAt": future_expiry,
                    }
                ]
            },
        },
    )
    assert shots_response.status_code == 200, shots_response.text
    body = shots_response.json()
    assert body["personMatte"]["delivery"]["sha256"]
    assert len(put_server["uploads"]) == 1
    assert body["provenance"]["referenceMode"] is True
    assert "MATTE_REFERENCE_MODE" in body["warningCodes"]


@pytest.mark.e2e
def test_matte_without_reference_mode_is_not_ready(
    model_dir,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matte_dir = model_dir.parent / "matte-no-ref"
    write_stub_models(model_dir)
    matte_dir.mkdir(parents=True, exist_ok=True)
    blob = b"stub-modnet"
    (matte_dir / "modnet.onnx").write_bytes(blob)
    (matte_dir / "manifest.json").write_text(
        (
            '{"models":[{"name":"modnet","file":"modnet.onnx","sha256":"'
            + hashlib.sha256(blob).hexdigest()
            + '","license":"Apache-2.0","version":"stub","stub":true}]}'
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEDIA_ANALYSIS_KEY", TEST_KEY)
    monkeypatch.setenv("MEDIA_ANALYSIS_MODEL_DIR", str(matte_dir))
    monkeypatch.setenv("MEDIA_ANALYSIS_IMAGE", "matte-cpu")
    monkeypatch.setenv("MEDIA_ANALYSIS_MATTE_REFERENCE_MODE", "0")
    reset_settings()
    reset_manifest()
    client = TestClient(create_app(Settings()))
    assert client.get("/ready").json()["ready"] is False
