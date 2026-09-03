from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.fixtures.generate import (
    write_ocr_text_mp4,
    write_rotated_mp4,
    write_speech_tone_mp4,
    write_vfr_mp4,
)

from media_analysis.app import create_app, reset_manifest
from media_analysis.config import Settings
from media_analysis.tools.vendor_models import vendor

LOCK = Path("models/manifest.lock.json")


def _client(tmp_path: Path) -> TestClient:
    model_dir = tmp_path / "models"
    vendor("analysis-cpu", LOCK, model_dir, stub=False, force=True)
    settings = Settings(
        media_analysis_key="real-model-test-key",
        media_analysis_allowed_hosts="127.0.0.1",
        media_analysis_model_dir=model_dir,
        media_analysis_allow_stub_models=False,
    )
    reset_manifest()
    return TestClient(create_app(settings))


def _analyze(client: TestClient, source_server: dict, video: Path, features: list[str], key: str):
    source_server["handler"].file_path = video
    return client.post(
        "/analyze",
        headers={"X-Media-Analysis-Key": "real-model-test-key"},
        json={
            "idempotencyKey": key,
            "canonicalize": "vfr" in key,
            "features": features,
            "source": {
                "signedGetUrl": source_server["url"],
                "expiresAt": "2099-01-01T00:00:00Z",
            },
        },
    )


@pytest.mark.production
@pytest.mark.real_models
@pytest.mark.ffmpeg
def test_prod_real_models_run_generated_goldens(
    tmp_path: Path,
    source_server: dict,
) -> None:
    with _client(tmp_path) as client:
        ocr = _analyze(
            client,
            source_server,
            write_ocr_text_mp4(tmp_path / "ocr.mp4"),
            ["ocr", "shots"],
            "real-ocr",
        )
        speech = _analyze(
            client,
            source_server,
            write_speech_tone_mp4(tmp_path / "speech.mp4"),
            ["audio", "waveform"],
            "real-speech",
        )
        vfr = _analyze(
            client,
            source_server,
            write_vfr_mp4(tmp_path / "vfr.mp4"),
            ["shots"],
            "real-vfr",
        )
        rotated = _analyze(
            client,
            source_server,
            write_rotated_mp4(tmp_path / "rotated.mp4"),
            ["shots"],
            "real-rotated",
        )

    for response in (ocr, speech, vfr, rotated):
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["overallStatus"] == "completed"
        assert body["telemetry"]["bytesDownloaded"] > 0
        assert body["provenance"]["ffmpegBuild"]["ffmpegVersion"] != "unavailable"

    assert vfr.json()["canonicalMedia"]["fps"] == {"numerator": 30, "denominator": 1}
    assert speech.json()["canonicalMedia"]["hasAudio"] is True
    assert len(speech.json()["audio"]["speech"]) > 0
    # Recall thresholds are not calibrated yet; the clip must remain measurable.
    assert "reservedRegions" in ocr.json()
