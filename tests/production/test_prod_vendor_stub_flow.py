from __future__ import annotations

from pathlib import Path

import pytest

from media_analysis.models_manifest import verify_models
from media_analysis.tools.vendor_models import vendor


@pytest.mark.production
@pytest.mark.unit
def test_prod_vendor_stub_writes_manifest_and_passes_verify(tmp_path: Path) -> None:
    vendor("analysis-cpu", Path("models/manifest.lock.json"), tmp_path, stub=True, force=False)
    manifest = tmp_path / "manifest.json"
    assert manifest.is_file()
    state = verify_models(tmp_path, image="analysis-cpu")
    assert state.ready
    assert state.loaded == (
        "yolox-tiny",
        "yunet",
        "PP-OCRv5_mobile_det",
        "silero-vad",
    )


@pytest.mark.production
@pytest.mark.unit
def test_prod_vendor_stub_refuses_force_over_mismatch_without_force(tmp_path: Path) -> None:
    vendor("analysis-cpu", Path("models/manifest.lock.json"), tmp_path, stub=True, force=False)
    (tmp_path / "yolox_tiny.onnx").write_bytes(b"tampered")
    with pytest.raises(SystemExit):
        vendor("analysis-cpu", Path("models/manifest.lock.json"), tmp_path, stub=True, force=False)


@pytest.mark.production
@pytest.mark.unit
def test_prod_vendor_stub_force_repairs_tampered_file(tmp_path: Path) -> None:
    vendor("analysis-cpu", Path("models/manifest.lock.json"), tmp_path, stub=True, force=False)
    (tmp_path / "yolox_tiny.onnx").write_bytes(b"tampered")
    vendor("analysis-cpu", Path("models/manifest.lock.json"), tmp_path, stub=True, force=True)
    assert verify_models(tmp_path, image="analysis-cpu").ready
