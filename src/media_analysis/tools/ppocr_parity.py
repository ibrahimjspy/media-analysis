"""Paddle-to-ONNX detection parity helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from media_analysis.features.ppocr import DetPolygon, polygon_to_box

PARITY_MAX_ABS_PROB = 1e-3
PARITY_MIN_BOX_IOU = 0.9
PARITY_POLICY_VERSION = "ocr-parity-paddle-onnx-1.0.0"


def box_iou(left: dict[str, float], right: dict[str, float]) -> float:
    lx0, ly0 = left["x"], left["y"]
    lx1, ly1 = lx0 + left["width"], ly0 + left["height"]
    rx0, ry0 = right["x"], right["y"]
    rx1, ry1 = rx0 + right["width"], ry0 + right["height"]
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = left["width"] * left["height"] + right["width"] * right["height"] - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def polygons_to_boxes(polygons: list[DetPolygon]) -> list[dict[str, float]]:
    return [polygon_to_box(item.points) for item in polygons]


def greedy_match(
    predicted: list[dict[str, float]],
    expected: list[dict[str, float]],
    *,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Return (true_positives, false_positives, false_negatives)."""
    used: set[int] = set()
    true_positives = 0
    for pred in predicted:
        best_index = -1
        best_iou = iou_threshold
        for index, truth in enumerate(expected):
            if index in used:
                continue
            score = box_iou(pred, truth)
            if score >= best_iou:
                best_iou = score
                best_index = index
        if best_index >= 0:
            used.add(best_index)
            true_positives += 1
    false_positives = len(predicted) - true_positives
    false_negatives = len(expected) - true_positives
    return true_positives, false_positives, false_negatives


def detection_scores(
    predicted: list[dict[str, float]],
    expected: list[dict[str, float]],
    *,
    iou_threshold: float,
) -> dict[str, float]:
    true_positives, false_positives, false_negatives = greedy_match(
        predicted,
        expected,
        iou_threshold=iou_threshold,
    )
    recall_den = true_positives + false_negatives
    precision_den = true_positives + false_positives
    return {
        "truePositives": float(true_positives),
        "falsePositives": float(false_positives),
        "falseNegatives": float(false_negatives),
        "recall": (true_positives / recall_den) if recall_den else 1.0,
        "precision": (true_positives / precision_den) if precision_den else 1.0,
    }


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise AssertionError(f"{name} contains NaN or infinite values")


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"probability map shape mismatch: {left.shape} vs {right.shape}")
    _require_finite("left probability map", left)
    _require_finite("right probability map", right)
    return float(np.max(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def _spatial_hw(prob: np.ndarray) -> tuple[int, int]:
    array = np.asarray(prob)
    if array.ndim < 2:
        raise ValueError(f"probability map rank {array.ndim} is too small")
    return int(array.shape[-2]), int(array.shape[-1])


def boxes_from_prob_map(prob: np.ndarray) -> list[dict[str, float]]:
    from media_analysis.features.ppocr import DetPreprocessMeta, postprocess_db_map

    height, width = _spatial_hw(prob)
    meta = DetPreprocessMeta(
        src_height=height,
        src_width=width,
        ratio=1.0,
        resized_height=height,
        resized_width=width,
    )
    return polygons_to_boxes(postprocess_db_map(prob, meta))


def assert_prob_map_parity(
    paddle_prob: np.ndarray,
    onnx_prob: np.ndarray,
    *,
    max_abs: float = PARITY_MAX_ABS_PROB,
) -> None:
    _require_finite("paddle probability map", paddle_prob)
    _require_finite("onnx probability map", onnx_prob)
    delta = max_abs_diff(paddle_prob, onnx_prob)
    if delta > max_abs:
        raise AssertionError(f"Paddle/ONNX probability map max abs {delta} exceeds {max_abs}")


def assert_box_parity(
    paddle_boxes: list[dict[str, float]],
    onnx_boxes: list[dict[str, float]],
    *,
    min_iou: float = PARITY_MIN_BOX_IOU,
) -> dict[str, float]:
    scores = detection_scores(onnx_boxes, paddle_boxes, iou_threshold=min_iou)
    if scores["recall"] < 1.0 or scores["precision"] < 1.0:
        raise AssertionError(
            "Paddle/ONNX boxes failed 1:1 match at "
            f"IoU>={min_iou}: {scores}"
        )
    return scores


def load_parity_reference(path: Any) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=False)
    return {
        "input": payload["input"],
        "paddleProb": payload["paddle_prob"],
        "paddleBoxes": payload["paddle_boxes"].tolist() if "paddle_boxes" in payload else [],
    }
