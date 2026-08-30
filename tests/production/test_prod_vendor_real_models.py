from __future__ import annotations

import json
from pathlib import Path

import pytest

from media_analysis.tools.model_lock import load_lock
from media_analysis.tools.vendor_models import vendor

LOCK = Path("models/manifest.lock.json")


@pytest.mark.production
@pytest.mark.real_models
def test_prod_vendor_real_weights_match_lock(tmp_path: Path) -> None:
    """Opt-in: downloads weights and verifies SHA-256 + size only (not production validation)."""
    lock = load_lock(LOCK, profile="analysis-cpu")
    vendor("analysis-cpu", LOCK, tmp_path, stub=False, force=True)
    for entry in lock.models:
        path = tmp_path / entry.file
        assert path.is_file()
        assert path.stat().st_size == entry.size_bytes


@pytest.mark.production
@pytest.mark.unit
def test_prod_ppocr_lock_documents_community_export_not_official() -> None:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    ppocr = next(
        item
        for item in payload["profiles"]["analysis-cpu"]["models"]
        if item["name"] == "PP-OCRv5_mobile_det"
    )
    prov = ppocr["provenance"]
    assert prov["artifactKind"] == "community-export"
    assert prov["ownedExportRecorded"] is False
    assert "PaddlePaddle/PaddleOCR" in prov["upstreamOfficialProject"]
    assert prov["exportRecipe"] == "docs/ppocr-owned-export.md"
    gates = payload["profiles"]["analysis-cpu"]["gates"]
    assert gates["ownedPpOcrExportRecorded"] is False


@pytest.mark.production
@pytest.mark.unit
def test_prod_yunet_lock_uses_immutable_opencv_zoo_commit() -> None:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    yunet = next(
        item for item in payload["profiles"]["analysis-cpu"]["models"] if item["name"] == "yunet"
    )
    assert "f12e12798e83" in yunet["url"]
    assert "/main/" not in yunet["url"]
    assert yunet["provenance"]["upstreamRevision"] == "f12e12798e83"
