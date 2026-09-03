from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.fixtures.generate import write_rotated_mp4, write_speech_tone_mp4, write_vfr_mp4

from media_analysis.app import create_app, reset_manifest


def _body(url: str, features: list[str], key: str, expiry: str) -> dict:
    return {
        "idempotencyKey": key,
        "canonicalize": False,
        "features": features,
        "source": {"signedGetUrl": url, "expiresAt": expiry},
        "analysisResolution": {"width": 160, "height": 120},
    }


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_analyze_reports_telemetry_and_build_provenance(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    response = client.post(
        "/analyze",
        headers=auth_headers,
        json=_body(source_server["url"], ["shots", "motion"], "telemetry", future_expiry),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    telemetry = body["telemetry"]
    assert telemetry["bytesDownloaded"] > 0
    assert telemetry["framesDecoded"] == body["canonicalMedia"]["frameCount"]
    assert telemetry["decodedPixels"] == 320 * 240 * body["canonicalMedia"]["frameCount"]
    assert telemetry["stageTimingsMs"]["download"] >= 0
    assert telemetry["stageTimingsMs"]["probe"] >= 0
    assert telemetry["stageTimingsMs"]["shots"] >= 0
    assert telemetry["stageTimingsMs"]["motion"] >= 0
    assert body["provenance"]["ffmpegBuild"]["ffmpegVersion"] != "unavailable"
    assert body["provenance"]["ffmpegBuild"]["ffprobeVersion"] != "unavailable"
    assert body["provenance"]["runtimeBuild"]["onnxruntimeVersion"]
    assert body["provenance"]["runtimeBuild"]["opencvVersion"]


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_decoded_pixel_limit_is_enforced_after_probe(
    settings,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
) -> None:
    limited = settings.model_copy(update={"media_analysis_max_decoded_pixels": 1})
    reset_manifest()
    try:
        with TestClient(create_app(limited)) as client:
            response = client.post(
                "/analyze",
                headers=auth_headers,
                json=_body(source_server["url"], ["shots"], "pixels", future_expiry),
            )
        assert response.status_code == 413
        assert response.json()["code"] == "LIMIT_EXCEEDED"
    finally:
        reset_manifest()


@pytest.mark.e2e
def test_stage_budget_is_enforced_before_download(
    settings,
    auth_headers: dict[str, str],
    future_expiry: str,
) -> None:
    limited = settings.model_copy(update={"media_analysis_stage_timeout_sec": 0})
    reset_manifest()
    try:
        with TestClient(create_app(limited)) as client:
            response = client.post(
                "/analyze",
                headers=auth_headers,
                json=_body("http://127.0.0.1/clip.mp4", ["shots"], "stage", future_expiry),
            )
        assert response.status_code == 504
        assert response.json()["code"] == "TIMEOUT"
    finally:
        reset_manifest()


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_vfr_source_canonicalizes_to_cfr(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
    tmp_path: Path,
) -> None:
    video = write_vfr_mp4(tmp_path / "vfr.mp4")
    source_server["handler"].file_path = video
    payload = _body(source_server["url"], ["shots"], "vfr", future_expiry)
    payload["canonicalize"] = True
    response = client.post("/analyze", headers=auth_headers, json=payload)
    assert response.status_code == 200, response.text
    media = response.json()["canonicalMedia"]
    assert media["fps"] == {"numerator": 30, "denominator": 1}
    assert media["frameCount"] >= 30
    assert "canonicalize" in response.json()["telemetry"]["stageTimingsMs"]


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_rotated_and_speech_goldens_complete(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
    tmp_path: Path,
) -> None:
    source_server["handler"].file_path = write_rotated_mp4(tmp_path / "rotated.mp4")
    rotated = client.post(
        "/analyze",
        headers=auth_headers,
        json=_body(source_server["url"], ["shots"], "rotated", future_expiry),
    )
    assert rotated.status_code == 200, rotated.text
    assert rotated.json()["canonicalMedia"]["rotationApplied"] is True

    source_server["handler"].file_path = write_speech_tone_mp4(tmp_path / "speech.mp4")
    speech = client.post(
        "/analyze",
        headers=auth_headers,
        json=_body(source_server["url"], ["waveform"], "speech", future_expiry),
    )
    assert speech.status_code == 200, speech.text
    assert speech.json()["canonicalMedia"]["hasAudio"] is True
    assert speech.json()["capabilities"]["waveform"]["status"] == "completed"


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_people_and_ocr_goldens_complete_on_stub_models(
    client: TestClient,
    auth_headers: dict[str, str],
    source_server: dict,
    future_expiry: str,
    tmp_path: Path,
) -> None:
    from tests.fixtures.generate import write_ocr_text_mp4, write_people_shapes_mp4

    source_server["handler"].file_path = write_people_shapes_mp4(tmp_path / "people.mp4")
    people = client.post(
        "/analyze",
        headers=auth_headers,
        json=_body(
            source_server["url"],
            ["subjects", "faces", "shots"],
            "people",
            future_expiry,
        ),
    )
    assert people.status_code == 200, people.text
    assert people.json()["subjects"] == []
    assert people.json()["faces"] == []

    source_server["handler"].file_path = write_ocr_text_mp4(tmp_path / "ocr.mp4")
    ocr = client.post(
        "/analyze",
        headers=auth_headers,
        json=_body(source_server["url"], ["ocr", "shots"], "ocr-text", future_expiry),
    )
    assert ocr.status_code == 200, ocr.text
    assert ocr.json()["reservedRegions"] == []
    assert "OCR_EMPTY" in ocr.json()["warningCodes"]
