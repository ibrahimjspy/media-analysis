import pytest
from fastapi.testclient import TestClient

from media_analysis.app import create_app, reset_manifest


def _analyze_body(url: str, features: list[str], key: str, expiry: str) -> dict:
    return {
        "idempotencyKey": key,
        "canonicalize": False,
        "features": features,
        "source": {"signedGetUrl": url, "expiresAt": expiry},
        "analysisResolution": {"width": 160, "height": 120},
    }


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_canonicalize_true_returns_measured_canonical_media(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    body = _analyze_body(source_server["url"], ["shots"], "canon", future_expiry)
    body["canonicalize"] = True
    response = client.post("/analyze", headers=auth_headers, json=body)
    assert response.status_code == 200, response.text
    media = response.json()["canonicalMedia"]
    assert media["frameCount"] >= 25
    assert media["fps"]["numerator"] == 30
    assert response.json()["provenance"]["decodePipelineVersion"]


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_shots_only_does_not_emit_subjects(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json=_analyze_body(source_server["url"], ["shots"], "shots-only", future_expiry),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schemaVersion"] == 1
    assert body["overallStatus"] == "completed"
    assert body["requestedFeatures"] == ["shots"]
    assert "subjects" not in body
    assert "faces" not in body
    assert "reservedRegions" not in body
    assert body["capabilities"]["shots"]["status"] == "completed"
    shots = body["shots"]
    assert shots
    assert shots[0]["startFrame"] == 0
    assert shots[-1]["endFrameExclusive"] == body["canonicalMedia"]["frameCount"]
    assert body["canonicalMedia"]["frameCount"] >= 25
    assert body["canonicalMedia"]["width"] == 320
    assert body["canonicalMedia"]["height"] == 240
    assert body["provenance"]["decodePipelineVersion"]
    assert body["provenance"]["shotAnalyzerVersion"]
    assert "subjectModel" not in body["provenance"]
    assert body["analysisTransform"]["uniformScale"] == 0.5


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_empty_person_video_completes_subjects_as_empty_list(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json=_analyze_body(
            source_server["url"],
            ["subjects", "faces", "ocr", "shots"],
            "v1-all",
            future_expiry,
        ),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["subjects"] == []
    assert body["faces"] == []
    assert body["reservedRegions"] == []
    assert body["capabilities"]["subjects"]["status"] == "completed"
    assert body["capabilities"]["ocr"]["status"] == "completed"
    assert "OCR_EMPTY" in body["warningCodes"]
    assert body["provenance"]["subjectModel"]["artifactSha256"]
    assert body["provenance"]["subjectModel"]["runtimeProvider"] == "stub"
    assert shots_cover_timeline(body)


def shots_cover_timeline(body: dict) -> bool:
    shots = body["shots"]
    assert shots[0]["startFrame"] == 0
    assert shots[-1]["endFrameExclusive"] == body["canonicalMedia"]["frameCount"]
    for left, right in zip(shots, shots[1:], strict=False):
        assert left["endFrameExclusive"] == right["startFrame"]
    return True


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_fill_subjects_then_subjects_and_faces_is_valid(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    first = client.post(
        "/analyze",
        headers=auth_headers,
        json=_analyze_body(source_server["url"], ["subjects"], "fill-1", future_expiry),
    )
    second = client.post(
        "/analyze",
        headers=auth_headers,
        json=_analyze_body(source_server["url"], ["subjects", "faces"], "fill-2", future_expiry),
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert "faces" not in first.json()
    assert second.json()["faces"] == []
    assert second.json()["subjects"] == []


@pytest.mark.e2e
def test_expired_source_does_not_fetch(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
) -> None:
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json=_analyze_body(
            source_server["url"],
            ["shots"],
            "expired",
            "2020-01-01T00:00:00Z",
        ),
    )
    assert response.status_code == 400
    assert response.json()["code"] == "SOURCE_EXPIRED"
    assert source_server["handler"].last_path == ""


@pytest.mark.e2e
def test_disallowed_host(
    client: TestClient,
    auth_headers: dict[str, str],
    future_expiry: str,
) -> None:
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json=_analyze_body("https://evil.example/clip.mp4", ["shots"], "ssrf", future_expiry),
    )
    assert response.json()["code"] == "SOURCE_FETCH_FAILED"
    assert "evil.example" not in response.text


@pytest.mark.e2e
@pytest.mark.ffmpeg
@pytest.mark.parametrize(
    "resolution",
    [
        {"width": 160, "height": 100},
        {"width": 640, "height": 480},
    ],
)
def test_analysis_resolution_must_be_downscaled_without_distortion(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
    resolution: dict[str, int],
) -> None:
    body = _analyze_body(source_server["url"], ["shots"], "bad-resolution", future_expiry)
    body["analysisResolution"] = resolution
    response = client.post("/analyze", headers=auth_headers, json=body)
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_source_sha_mismatch(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    body = _analyze_body(source_server["url"], ["shots"], "bad-sha", future_expiry)
    body["source"]["sha256"] = "0" * 64
    response = client.post("/analyze", headers=auth_headers, json=body)
    assert response.json()["code"] == "CHECKSUM_MISMATCH"


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_idempotent_replay_returns_same_result(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    payload = _analyze_body(source_server["url"], ["shots"], "same-key", future_expiry)
    first = client.post("/analyze", headers=auth_headers, json=payload)
    payload["source"]["signedGetUrl"] = source_server["url"] + "&retry=1"
    second = client.post("/analyze", headers=auth_headers, json=payload)
    assert first.status_code == 200
    assert second.json() == first.json()


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_same_key_different_features_is_invalid(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    client.post(
        "/analyze",
        headers=auth_headers,
        json=_analyze_body(source_server["url"], ["shots"], "clash", future_expiry),
    )
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json=_analyze_body(source_server["url"], ["subjects"], "clash", future_expiry),
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


@pytest.mark.e2e
def test_cancel_unknown_key(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post("/analyze/does-not-exist/cancel", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"cancelled": False, "idempotencyKey": "does-not-exist"}


@pytest.mark.e2e
def test_cancel_requires_key(client: TestClient) -> None:
    response = client.post("/analyze/k/cancel")
    assert response.json()["code"] == "UNAUTHORIZED"


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_motion_completes_on_analysis_cpu(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json=_analyze_body(source_server["url"], ["motion"], "motion-v11", future_expiry),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["overallStatus"] == "completed"
    assert "motion" in body
    assert body["capabilities"]["motion"]["status"] == "completed"
    assert body["provenance"]["motionAnalyzerVersion"]


@pytest.mark.e2e
def test_analyze_without_ready_models(
    model_dir,
    settings,
    auth_headers: dict[str, str],
    future_expiry: str,
) -> None:
    (model_dir / "yolox_tiny.onnx").write_bytes(b"broken")
    reset_manifest()
    client = TestClient(create_app(settings))
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json=_analyze_body("http://127.0.0.1/clip.mp4", ["shots"], "not-ready", future_expiry),
    )
    assert response.json()["code"] == "MODEL_NOT_READY"


@pytest.mark.e2e
def test_job_deadline_is_enforced_before_download(
    settings,
    auth_headers: dict[str, str],
    future_expiry: str,
) -> None:
    timed_settings = settings.model_copy(update={"media_analysis_job_timeout_sec": 0})
    reset_manifest()
    try:
        with TestClient(create_app(timed_settings)) as timed_client:
            response = timed_client.post(
                "/analyze",
                headers=auth_headers,
                json=_analyze_body(
                    "http://127.0.0.1/clip.mp4",
                    ["shots"],
                    "deadline",
                    future_expiry,
                ),
            )
        assert response.status_code == 504
        assert response.json()["code"] == "TIMEOUT"
    finally:
        reset_manifest()


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_source_download_limit_is_enforced_while_streaming(
    settings,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    limited_settings = settings.model_copy(update={"media_analysis_max_bytes": 1})
    reset_manifest()
    try:
        with TestClient(create_app(limited_settings)) as limited_client:
            response = limited_client.post(
                "/analyze",
                headers=auth_headers,
                json=_analyze_body(
                    source_server["url"],
                    ["shots"],
                    "download-limit",
                    future_expiry,
                ),
            )
        assert response.status_code == 413
        assert response.json()["code"] == "LIMIT_EXCEEDED"
    finally:
        reset_manifest()
