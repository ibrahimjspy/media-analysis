from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.conftest import TEST_KEY

from media_analysis.app import create_app, reset_manifest
from media_analysis.config import reset_settings


@pytest.mark.e2e
def test_health_is_unauthenticated(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"status": "ok", "version": "1.2.0"}


@pytest.mark.e2e
def test_ready_true_after_stub_models_match(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["productionInferenceReady"] is False
    assert body["productionBlockers"] == ["stub model artifacts are not production-ready"]
    assert body["referenceMode"] is True
    assert body["warmupComplete"] is True
    assert body["loadedModels"] == [
        "yolox-tiny",
        "yunet",
        "PP-OCRv5_mobile_det",
        "silero-vad",
    ]
    assert body["workerRole"] == "combined"


@pytest.mark.e2e
def test_metrics_exposes_stage_percentiles_without_auth(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "stageLatencyPercentiles" in response.json()


@pytest.mark.e2e
def test_ready_false_when_checksum_breaks(
    model_dir: Path,
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (model_dir / "yolox_tiny.onnx").write_bytes(b"broken")
    reset_manifest()
    client = TestClient(create_app(settings))
    response = client.get("/ready")
    assert response.json()["ready"] is False


@pytest.mark.e2e
def test_worker_boot_fails_when_model_checksum_breaks(
    model_dir: Path,
    settings,
) -> None:
    (model_dir / "yolox_tiny.onnx").write_bytes(b"broken")
    reset_manifest()
    app = create_app(settings)
    with pytest.raises(RuntimeError, match="model runtime failed to initialize"):
        with TestClient(app):
            pass


@pytest.mark.e2e
def test_analyze_without_key_is_401_json(client: TestClient, future_expiry: str) -> None:
    response = client.post(
        "/analyze",
        json={
            "idempotencyKey": "k",
            "features": ["shots"],
            "source": {"signedGetUrl": "http://127.0.0.1/x", "expiresAt": future_expiry},
        },
    )
    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body == {
        "error": "Missing or invalid X-Media-Analysis-Key",
        "code": "UNAUTHORIZED",
    }
    assert "<html" not in response.text.lower()


@pytest.mark.e2e
def test_analyze_wrong_key_is_401(client: TestClient, future_expiry: str) -> None:
    response = client.post(
        "/analyze",
        headers={"X-Media-Analysis-Key": "nope"},
        json={
            "idempotencyKey": "k",
            "features": ["shots"],
            "source": {"signedGetUrl": "http://127.0.0.1/x", "expiresAt": future_expiry},
        },
    )
    assert response.json()["code"] == "UNAUTHORIZED"


@pytest.mark.e2e
def test_unknown_feature_is_400(
    client: TestClient,
    auth_headers: dict[str, str],
    future_expiry: str,
) -> None:
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json={
            "idempotencyKey": "k",
            "features": ["telepathy"],
            "source": {"signedGetUrl": "http://127.0.0.1/x", "expiresAt": future_expiry},
        },
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


@pytest.mark.e2e
def test_person_matte_on_cpu_is_400(
    client: TestClient, auth_headers: dict[str, str], future_expiry: str
) -> None:
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json={
            "idempotencyKey": "k",
            "features": ["person_matte"],
            "source": {"signedGetUrl": "http://127.0.0.1/x", "expiresAt": future_expiry},
        },
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


@pytest.mark.e2e
def test_missing_models_dir_ready_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_ANALYSIS_KEY", TEST_KEY)
    monkeypatch.setenv("MEDIA_ANALYSIS_MODEL_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("MEDIA_ANALYSIS_IMAGE", "analysis-cpu")
    reset_settings()
    reset_manifest()
    client = TestClient(create_app())
    assert client.get("/ready").json()["ready"] is False
