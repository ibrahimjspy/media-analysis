from __future__ import annotations

from pathlib import Path

import pytest

from media_analysis.features.silero_vad import (
    load_silero_vad_session,
    warmup_silero_vad,
)
from media_analysis.tools.model_compat import (
    check_ppocr_db_runtime_compat,
    check_yolox_runtime_compat,
    check_yunet_opencv_runtime_compat,
)
from media_analysis.tools.model_lock import load_lock
from media_analysis.tools.vendor_models import vendor

LOCK = Path("models/manifest.lock.json")


@pytest.fixture(scope="module")
def vendored_real_models(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dest = tmp_path_factory.mktemp("real-models")
    vendor("analysis-cpu", LOCK, dest, stub=False, force=True)
    return dest


@pytest.mark.production
@pytest.mark.real_models
def test_prod_real_model_yolox_runtime_compat(vendored_real_models: Path) -> None:
    """Opt-in runtime compatibility for official YOLOX raw output layout (not recall/parity)."""
    lock = load_lock(LOCK, profile="analysis-cpu")
    entry = next(item for item in lock.models if item.name == "yolox-tiny")
    check_yolox_runtime_compat(vendored_real_models / entry.file)


@pytest.mark.production
@pytest.mark.real_models
def test_prod_real_model_yunet_opencv_runtime_compat(vendored_real_models: Path) -> None:
    """Opt-in FaceDetectorYN load + dummy detect (not recall/parity)."""
    lock = load_lock(LOCK, profile="analysis-cpu")
    entry = next(item for item in lock.models if item.name == "yunet")
    check_yunet_opencv_runtime_compat(vendored_real_models / entry.file)


@pytest.mark.production
@pytest.mark.real_models
def test_prod_real_model_ppocr_db_runtime_compat(vendored_real_models: Path) -> None:
    """Opt-in DB probability-map compat for interim community export (not recall/parity)."""
    lock = load_lock(LOCK, profile="analysis-cpu")
    entry = next(item for item in lock.models if item.name == "PP-OCRv5_mobile_det")
    assert entry.provenance is not None
    assert entry.provenance.is_community_export
    check_ppocr_db_runtime_compat(vendored_real_models / entry.file)


@pytest.mark.production
@pytest.mark.real_models
def test_prod_real_model_silero_v62_runtime_compat(vendored_real_models: Path) -> None:
    """Opt-in Silero v6.2 load + stateful zero-window warmup."""
    lock = load_lock(LOCK, profile="analysis-cpu")
    entry = next(item for item in lock.models if item.name == "silero-vad")
    session = load_silero_vad_session(vendored_real_models / entry.file)
    warmup_silero_vad(session)
