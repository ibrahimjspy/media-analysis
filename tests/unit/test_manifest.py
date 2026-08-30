import hashlib
import json
from pathlib import Path

import pytest
from tests.conftest import write_stub_models

from media_analysis.models_manifest import verify_models


@pytest.mark.unit
def test_matching_stub_manifest_is_ready(tmp_path: Path) -> None:
    write_stub_models(tmp_path)
    state = verify_models(tmp_path, image="analysis-cpu")
    assert state.ready
    assert state.loaded == (
        "yolox-tiny",
        "yunet",
        "PP-OCRv5_mobile_det",
        "silero-vad",
    )


@pytest.mark.unit
def test_sha_mismatch_fails_boot(tmp_path: Path) -> None:
    write_stub_models(tmp_path)
    (tmp_path / "yolox_tiny.onnx").write_bytes(b"tampered")
    state = verify_models(tmp_path, image="analysis-cpu")
    assert not state.ready
    assert any("sha256 mismatch" in error for error in state.errors)


@pytest.mark.unit
def test_cpu_image_rejects_matte_weights(tmp_path: Path) -> None:
    write_stub_models(tmp_path)
    blob = b"modnet-stub"
    (tmp_path / "modnet.onnx").write_bytes(blob)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["models"].append(
        {
            "name": "modnet",
            "file": "modnet.onnx",
            "sha256": hashlib.sha256(blob).hexdigest(),
            "license": "Apache-2.0",
            "stub": True,
        }
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    state = verify_models(tmp_path, image="analysis-cpu")
    assert not state.ready
    assert any("matte" in error for error in state.errors)


@pytest.mark.unit
def test_missing_manifest_is_not_ready(tmp_path: Path) -> None:
    state = verify_models(tmp_path, image="analysis-cpu")
    assert not state.ready
