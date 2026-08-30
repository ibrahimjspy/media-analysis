from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from media_analysis.app import create_app, reset_manifest
from media_analysis.config import Settings
from media_analysis.tools.vendor_models import vendor

LOCK = Path("models/manifest.lock.json")


@pytest.mark.production
@pytest.mark.real_models
@pytest.mark.ffmpeg
def test_prod_real_models_run_through_analyze_http(
    tmp_path: Path,
    source_server: dict,
    future_expiry: str,
) -> None:
    model_dir = tmp_path / "models"
    vendor("analysis-cpu", LOCK, model_dir, stub=False, force=True)
    settings = Settings(
        media_analysis_key="real-model-test-key",
        media_analysis_allowed_hosts="127.0.0.1",
        media_analysis_model_dir=model_dir,
        media_analysis_allow_stub_models=False,
    )
    reset_manifest()

    with TestClient(create_app(settings)) as client:
        ready = client.get("/ready").json()
        assert ready["ready"] is True
        assert ready["productionInferenceReady"] is False
        assert ready["productionBlockers"] == ["owned PP-OCR export parity gate is closed"]
        assert ready["referenceMode"] is False
        assert ready["warmupComplete"] is True
        assert ready["loadedModels"] == [
            "yolox-tiny",
            "yunet",
            "PP-OCRv5_mobile_det",
            "silero-vad",
        ]

        response = client.post(
            "/analyze",
            headers={"X-Media-Analysis-Key": "real-model-test-key"},
            json={
                "idempotencyKey": "real-model-red-clip",
                "source": {
                    "signedGetUrl": source_server["url"],
                    "expiresAt": future_expiry,
                    "sha256": source_server["sha256"],
                },
                "features": ["subjects", "faces", "ocr", "shots"],
                "analysisResolution": {"width": 160, "height": 120},
            },
        )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["overallStatus"] == "completed"
    assert result["subjects"] == []
    assert result["faces"] == []
    assert result["reservedRegions"] == []
    assert result["shots"]
    assert {
        name: capability["status"]
        for name, capability in result["capabilities"].items()
    } == {
        "subjects": "completed",
        "faces": "completed",
        "ocr": "completed",
        "shots": "completed",
    }
    assert result["provenance"]["subjectModel"]["runtimeProvider"] == "onnxruntime-cpu"
    assert result["provenance"]["faceModel"]["runtimeProvider"] == "opencv-dnn"
    assert result["provenance"]["ocrModel"]["runtimeProvider"] == "onnxruntime-cpu"
