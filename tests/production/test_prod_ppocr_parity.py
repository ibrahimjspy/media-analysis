from __future__ import annotations

from pathlib import Path

import pytest

from media_analysis.features.ppocr import create_det_session
from media_analysis.tools.export_ppocr import load_export_lock
from media_analysis.tools.model_lock import load_lock
from media_analysis.tools.ppocr_parity import (
    PARITY_MAX_ABS_PROB,
    PARITY_MIN_BOX_IOU,
    assert_box_parity,
    assert_prob_map_parity,
    boxes_from_prob_map,
    load_parity_reference,
)
from media_analysis.tools.vendor_models import vendor

LOCK = Path("models/manifest.lock.json")
PARITY_REF = Path("tests/fixtures/ppocr_parity/reference.npz")


@pytest.mark.production
@pytest.mark.unit
def test_owned_export_gate_requires_recorded_parity_reference() -> None:
    lock = load_lock(LOCK, profile="analysis-cpu")
    if lock.gates is None or not lock.gates.owned_ppocr_export_recorded:
        pytest.skip("owned PP-OCR export gate is still closed")
    assert PARITY_REF.is_file(), "owned export must commit a Paddle/ONNX reference tensor"


@pytest.mark.production
@pytest.mark.real_models
def test_prod_onnx_matches_recorded_paddle_reference(
    tmp_path: Path,
) -> None:
    if not PARITY_REF.is_file():
        pytest.skip("Paddle reference tensors are recorded only after an owned export")
    export_lock = load_export_lock()
    dest = tmp_path / "models"
    vendor("analysis-cpu", LOCK, dest, stub=False, force=True)
    lock = load_lock(LOCK, profile="analysis-cpu")
    entry = next(item for item in lock.models if item.name == "PP-OCRv5_mobile_det")
    session = create_det_session(dest / entry.file)
    reference = load_parity_reference(PARITY_REF)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_prob = session.run([output_name], {input_name: reference["input"]})[0]
    min_box_iou = float(export_lock["parity"]["minBoxIou"] or PARITY_MIN_BOX_IOU)
    assert_prob_map_parity(
        reference["paddleProb"],
        onnx_prob,
        max_abs=float(export_lock["parity"]["maxAbsProb"] or PARITY_MAX_ABS_PROB),
    )
    onnx_boxes = boxes_from_prob_map(onnx_prob)
    assert_box_parity(
        boxes_from_prob_map(reference["paddleProb"]),
        onnx_boxes,
        min_iou=min_box_iou,
    )
    if reference["paddleBoxes"]:
        assert_box_parity(reference["paddleBoxes"], onnx_boxes, min_iou=min_box_iou)
