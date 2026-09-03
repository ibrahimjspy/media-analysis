from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.fixtures.generate import write_image_loop_mp4

from media_analysis.app import create_app, reset_manifest
from media_analysis.config import Settings
from media_analysis.features.ppocr import create_det_session, run_det_inference
from media_analysis.tools.model_lock import load_lock
from media_analysis.tools.ppocr_recall import (
    assert_recall_floors,
    polygons_as_boxes,
    render_latin_banner,
)
from media_analysis.tools.vendor_models import vendor

LOCK = Path("models/manifest.lock.json")


@pytest.fixture(scope="module")
def vendored_ocr(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dest = tmp_path_factory.mktemp("ocr-models")
    vendor("analysis-cpu", LOCK, dest, stub=False, force=True)
    return dest


@pytest.mark.production
@pytest.mark.real_models
def test_prod_real_ocr_detects_synthetic_latin_banner(vendored_ocr: Path) -> None:
    lock = load_lock(LOCK, profile="analysis-cpu")
    entry = next(item for item in lock.models if item.name == "PP-OCRv5_mobile_det")
    session = create_det_session(vendored_ocr / entry.file)
    image, expected = render_latin_banner("HELLO")
    boxes = polygons_as_boxes(run_det_inference(session, image, limit_side_len=320))
    assert_recall_floors(boxes, [expected])


@pytest.mark.production
@pytest.mark.real_models
@pytest.mark.ffmpeg
def test_prod_real_ocr_pipeline_recalls_top_banner_text(
    vendored_ocr: Path,
    tmp_path: Path,
    source_server: dict,
) -> None:
    settings = Settings(
        media_analysis_key="real-model-test-key",
        media_analysis_allowed_hosts="127.0.0.1",
        media_analysis_model_dir=vendored_ocr,
        media_analysis_allow_stub_models=False,
    )
    reset_manifest()
    image, expected = render_latin_banner("HELLO")
    video = write_image_loop_mp4(tmp_path / "ocr-banner.mp4", image)
    source_server["handler"].file_path = video

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/analyze",
            headers={"X-Media-Analysis-Key": "real-model-test-key"},
            json={
                "idempotencyKey": "real-ocr-recall",
                "features": ["ocr", "shots"],
                "source": {
                    "signedGetUrl": source_server["url"],
                    "expiresAt": "2099-01-01T00:00:00Z",
                },
            },
        )
    assert response.status_code == 200, response.text
    regions = response.json()["reservedRegions"]
    predicted = [sample["box"] for region in regions for sample in region.get("samples", [])]
    assert_recall_floors(predicted, [expected])
