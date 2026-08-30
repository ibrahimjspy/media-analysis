"""Unit tests for YOLOX letterbox, decode, and NMS (fake ONNX sessions only)."""

from __future__ import annotations

import numpy as np
import pytest

from media_analysis.features.yolox import (
    COCO_PERSON_CLASS_ID,
    INPUT_SIZE,
    LETTERBOX_FILL,
    YOLOX_PREPROCESSING,
    Detection,
    LetterboxMeta,
    YoloxDetector,
    decode_yolox_outputs,
    filter_detections,
    letterbox_preprocess,
    map_boxes_to_normalized,
    nms_xyxy,
)


class FakeOnnxSession:
    def __init__(self, output: np.ndarray) -> None:
        self._output = output
        self.last_input: np.ndarray | None = None

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.last_input = next(iter(input_feed.values()))
        return [self._output]


def _single_person_output(
    *,
    cx: float,
    cy: float,
    w: float,
    h: float,
    obj: float,
    person_score: float,
    grid_index: int = 0,
) -> np.ndarray:
    """Build a minimal YOLOX-style row at one grid cell (pre-decode logits)."""
    rows = 3549
    out = np.zeros((1, rows, 85), dtype=np.float32)
    row = out[0, grid_index]
    row[0] = cx
    row[1] = cy
    row[2] = np.log(max(w / 8.0, 1e-6))
    row[3] = np.log(max(h / 8.0, 1e-6))
    row[4] = obj
    row[5 + COCO_PERSON_CLASS_ID] = person_score
    return out


@pytest.mark.unit
def test_letterbox_preprocess_matches_official_megvii_semantics() -> None:
    """BGR, top-left placement, CHW float32, no RGB swap, no /255."""
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)  # BGR
    tensor, meta = letterbox_preprocess(image)
    assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == np.float32
    # BGR channel order preserved (no RGB swap); values are raw 0..255, not /255.
    assert float(tensor[0, 0, 0, 0]) == pytest.approx(10.0)
    assert float(tensor[0, 1, 0, 0]) == pytest.approx(20.0)
    assert float(tensor[0, 2, 0, 0]) == pytest.approx(30.0)
    assert meta.orig_width == 160
    assert meta.orig_height == 120
    assert meta.scale == pytest.approx(min(INPUT_SIZE / 120, INPUT_SIZE / 160))
    assert meta.pad_left == 0.0
    assert meta.pad_top == 0.0
    new_h = int(120 * meta.scale)
    # Padding is bottom/right only; first row below content stays 114.
    assert float(tensor[0, 0, new_h, 0]) == pytest.approx(LETTERBOX_FILL)


@pytest.mark.unit
def test_decode_nms_keeps_person_class_only() -> None:
    output = _single_person_output(cx=10, cy=10, w=40, h=80, obj=0.9, person_score=0.95)
    output[0, 1, 5 + 1] = 0.99
    output[0, 1, 4] = 0.99
    boxes, scores, class_ids = decode_yolox_outputs(output)
    detections = filter_detections(boxes, scores, class_ids, conf_thresh=0.1, nms_iou=0.45)
    assert len(detections) == 1
    assert detections[0].class_id == COCO_PERSON_CLASS_ID
    assert detections[0].score == pytest.approx(0.9 * 0.95, rel=1e-3)


@pytest.mark.unit
def test_nms_suppresses_overlapping_boxes() -> None:
    boxes = np.array(
        [
            [0, 0, 50, 50],
            [5, 5, 55, 55],
            [200, 200, 260, 260],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms_xyxy(boxes, scores, iou_threshold=0.5)
    assert list(keep) == [0, 2]


@pytest.mark.unit
def test_map_boxes_to_normalized_uses_scale_only() -> None:
    meta = LetterboxMeta(scale=2.0, orig_width=100, orig_height=100)
    boxes = np.array([[20, 40, 60, 80]], dtype=np.float32)
    normalized = map_boxes_to_normalized(boxes, meta)
    assert normalized[0]["x"] == pytest.approx(0.1)
    assert normalized[0]["y"] == pytest.approx(0.2)
    assert normalized[0]["width"] == pytest.approx(0.2)
    assert normalized[0]["height"] == pytest.approx(0.2)


@pytest.mark.unit
def test_map_boxes_to_normalized_clamps_to_unit_interval() -> None:
    meta = LetterboxMeta(scale=1.0, orig_width=100, orig_height=100)
    boxes = np.array([[-10, -10, 120, 120]], dtype=np.float32)
    normalized = map_boxes_to_normalized(boxes, meta)
    assert normalized[0]["x"] == 0.0
    assert normalized[0]["y"] == 0.0
    assert normalized[0]["width"] == pytest.approx(1.0)
    assert normalized[0]["height"] == pytest.approx(1.0)


@pytest.mark.unit
def test_yolox_detector_uses_injected_fake_session() -> None:
    output = _single_person_output(cx=20, cy=20, w=32, h=64, obj=1.0, person_score=0.99)
    session = FakeOnnxSession(output)
    detector = YoloxDetector(session=session, conf_thresh=0.05)
    image = np.full((240, 320, 3), 114, dtype=np.uint8)
    detections, meta = detector.detect(image)
    assert session.last_input is not None
    assert session.last_input.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert meta.orig_width == 320
    assert len(detections) >= 1
    assert all(isinstance(item, Detection) for item in detections)


@pytest.mark.unit
def test_empty_decode_returns_no_detections() -> None:
    output = np.zeros((1, 3549, 85), dtype=np.float32)
    boxes, scores, class_ids = decode_yolox_outputs(output)
    assert filter_detections(boxes, scores, class_ids) == []


@pytest.mark.unit
def test_provenance_preprocessing_constant() -> None:
    assert YOLOX_PREPROCESSING == "letterbox-416"
