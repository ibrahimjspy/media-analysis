from __future__ import annotations

import json
from pathlib import Path

import pytest

from media_analysis.tools.model_lock import load_lock

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "models" / "manifest.lock.json"


@pytest.mark.production
@pytest.mark.unit
def test_prod_analysis_cpu_lock_has_no_matte_weights() -> None:
    lock = load_lock(LOCK_PATH, profile="analysis-cpu")
    names = {item.name for item in lock.models}
    assert "modnet" not in names
    assert names == {"yolox-tiny", "yunet", "PP-OCRv5_mobile_det", "silero-vad"}


@pytest.mark.production
@pytest.mark.unit
def test_prod_lock_entries_have_pinned_metadata() -> None:
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert payload["lockVersion"] == 2
    for item in payload["profiles"]["analysis-cpu"]["models"]:
        if item.get("stub"):
            assert item["url"].startswith("stub://")
        else:
            assert item["url"].startswith("https://")
        assert len(item["sha256"]) == 64
        assert item["revision"]
        assert item["license"] in {"Apache-2.0", "MIT"}
        assert item["sizeBytes"] > 0
        assert "provenance" in item
        assert item["provenance"]["artifactKind"]


@pytest.mark.production
@pytest.mark.unit
def test_prod_lock_sha256_are_lowercase_hex() -> None:
    lock = load_lock(LOCK_PATH, profile="analysis-cpu")
    for item in lock.models:
        assert item.sha256 == item.sha256.lower()
        assert all(ch in "0123456789abcdef" for ch in item.sha256)


@pytest.mark.production
@pytest.mark.unit
def test_prod_silero_lock_uses_verified_upstream_commit() -> None:
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    profile = payload["profiles"]["analysis-cpu"]
    silero = next(item for item in profile["models"] if item["name"] == "silero-vad")
    revision = "bfdc0193023f121ea5b3cc7b176dbed570a68a59"
    assert revision in silero["url"]
    assert silero["revision"].endswith(revision)
    assert silero["sha256"] == "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
    assert silero["sizeBytes"] == 2_327_524
    assert silero.get("stub") is not True
    assert silero["provenance"]["artifactKind"] == "upstream-commit"
    assert profile["gates"]["sileroVadProductionEnabled"] is True
