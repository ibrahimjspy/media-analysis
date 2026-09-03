import json
import tarfile
from pathlib import Path

import pytest

from media_analysis.tools.export_ppocr import (
    extract_tar,
    find_inference_dir,
    load_export_lock,
    resolve_model_files,
    sha256_file,
    write_recipe,
)


@pytest.mark.unit
def test_export_lock_pins_official_inference_and_thresholds() -> None:
    lock = load_export_lock()
    upstream = lock["upstreamInference"]
    assert upstream["url"].startswith("https://")
    assert "PP-OCRv5_mobile_det_infer" in upstream["url"]
    assert len(upstream["sha256"]) == 64
    assert upstream["sizeBytes"] == 4_935_680
    assert lock["toolchain"]["platform"] == "linux/amd64"
    assert lock["toolchain"]["paddle2onnx"]
    assert lock["parity"]["maxAbsProb"] <= 0.001
    assert lock["parity"]["minBoxIou"] >= 0.9
    assert lock["recall"]["minRecall"] >= 0.8


@pytest.mark.unit
def test_find_inference_dir_and_json_layout(tmp_path: Path) -> None:
    infer = tmp_path / "PP-OCRv5_mobile_det_infer"
    infer.mkdir()
    (infer / "inference.json").write_text("{}", encoding="utf-8")
    (infer / "inference.pdiparams").write_bytes(b"params")
    found = find_inference_dir(tmp_path)
    assert found == infer
    assert resolve_model_files(found) == ("inference.json", "inference.pdiparams")


@pytest.mark.unit
def test_find_inference_dir_supports_legacy_pdmodel(tmp_path: Path) -> None:
    infer = tmp_path / "legacy"
    infer.mkdir()
    (infer / "inference.pdmodel").write_bytes(b"graph")
    (infer / "inference.pdiparams").write_bytes(b"params")
    assert resolve_model_files(find_inference_dir(tmp_path)) == (
        "inference.pdmodel",
        "inference.pdiparams",
    )


@pytest.mark.unit
def test_extract_tar_replaces_stale_directory(tmp_path: Path) -> None:
    dest = tmp_path / "infer"
    dest.mkdir()
    (dest / "stale-model.pdiparams").write_bytes(b"old-weights")

    source = tmp_path / "fresh"
    source.mkdir()
    (source / "inference.pdiparams").write_bytes(b"new-weights")
    archive = tmp_path / "models.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(source / "inference.pdiparams", arcname="fresh/inference.pdiparams")

    extract_tar(archive, dest)
    assert not (dest / "stale-model.pdiparams").exists()
    assert (dest / "fresh" / "inference.pdiparams").read_bytes() == b"new-weights"


@pytest.mark.unit
def test_write_recipe_and_sha256(tmp_path: Path) -> None:
    blob = tmp_path / "model.onnx"
    blob.write_bytes(b"onnx-bytes")
    recipe = write_recipe(
        tmp_path / "model.recipe.json",
        {"onnxSha256": sha256_file(blob), "artifactKind": "owned-export"},
    )
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    assert payload["artifactKind"] == "owned-export"
    assert len(payload["onnxSha256"]) == 64
