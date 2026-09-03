import numpy as np
import pytest

from media_analysis.tools.ppocr_parity import (
    assert_box_parity,
    assert_prob_map_parity,
    box_iou,
    detection_scores,
    max_abs_diff,
)


@pytest.mark.unit
def test_identical_boxes_have_unit_iou() -> None:
    box = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}
    assert box_iou(box, box) == pytest.approx(1.0)


@pytest.mark.unit
def test_disjoint_boxes_have_zero_iou() -> None:
    left = {"x": 0.0, "y": 0.0, "width": 0.2, "height": 0.2}
    right = {"x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2}
    assert box_iou(left, right) == 0.0


@pytest.mark.unit
def test_detection_scores_count_unmatched_as_errors() -> None:
    expected = [{"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1}]
    predicted = [
        {"x": 0.11, "y": 0.11, "width": 0.2, "height": 0.1},
        {"x": 0.7, "y": 0.7, "width": 0.1, "height": 0.1},
    ]
    scores = detection_scores(predicted, expected, iou_threshold=0.5)
    assert scores["truePositives"] == 1
    assert scores["falsePositives"] == 1
    assert scores["falseNegatives"] == 0
    assert scores["recall"] == 1.0
    assert scores["precision"] == pytest.approx(0.5)


@pytest.mark.unit
def test_prob_map_parity_accepts_small_noise() -> None:
    paddle = np.full((1, 1, 8, 8), 0.5, dtype=np.float32)
    onnx = paddle + 1e-4
    assert max_abs_diff(paddle, onnx) < 1e-3
    assert_prob_map_parity(paddle, onnx)


@pytest.mark.unit
def test_prob_map_parity_rejects_large_drift() -> None:
    paddle = np.zeros((4, 4), dtype=np.float32)
    onnx = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(AssertionError, match="max abs"):
        assert_prob_map_parity(paddle, onnx)


@pytest.mark.unit
def test_prob_map_parity_rejects_nan_and_infinite_maps() -> None:
    paddle = np.full((4, 4), 0.5, dtype=np.float32)
    nan_map = paddle.copy()
    nan_map[0, 0] = np.nan
    inf_map = paddle.copy()
    inf_map[0, 0] = np.inf
    with pytest.raises(AssertionError, match="NaN or infinite"):
        assert_prob_map_parity(paddle, nan_map)
    with pytest.raises(AssertionError, match="NaN or infinite"):
        assert_prob_map_parity(inf_map, paddle)


@pytest.mark.unit
def test_box_parity_requires_one_to_one_match() -> None:
    boxes = [{"x": 0.2, "y": 0.2, "width": 0.3, "height": 0.1}]
    assert_box_parity(boxes, boxes)
    with pytest.raises(AssertionError, match="1:1"):
        assert_box_parity(boxes, [])


@pytest.mark.unit
def test_box_parity_enforces_configured_min_iou() -> None:
    paddle = [{"x": 0.10, "y": 0.10, "width": 0.20, "height": 0.10}]
    onnx = [{"x": 0.13, "y": 0.10, "width": 0.20, "height": 0.10}]
    assert_box_parity(paddle, onnx, min_iou=0.5)
    with pytest.raises(AssertionError, match="IoU>=0.9"):
        assert_box_parity(paddle, onnx, min_iou=0.9)
