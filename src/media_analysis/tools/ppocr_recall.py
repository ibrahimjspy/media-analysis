"""Synthetic OCR detection recall fixtures and scoring."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from media_analysis.tools.ppocr_parity import detection_scores

RECALL_IOU = 0.5
RECALL_MIN = 0.8
PRECISION_MIN = 0.5
RECALL_POLICY_VERSION = "ocr-recall-synthetic-1.0.0"
DEFAULT_TEXT = "HELLO"


def render_latin_banner(
    text: str = DEFAULT_TEXT,
    *,
    width: int = 640,
    height: int = 360,
) -> tuple[np.ndarray, dict[str, float]]:
    """White Latin text in the top 20% so the OCR sampler can see it."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    scale = 1.6
    thickness = 3
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    roi_bottom = int(height * 0.20) - 2
    x = max(4, (width - text_w) // 2)
    y = min(roi_bottom - baseline, text_h + 6)
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    box = {
        "x": x / width,
        "y": (y - text_h) / height,
        "width": text_w / width,
        "height": (text_h + baseline) / height,
    }
    if box["y"] + box["height"] > 0.20:
        raise ValueError("banner must stay inside the top 20% OCR sampler ROI")
    return image, box


def score_detections(
    predicted: list[dict[str, float]],
    expected: list[dict[str, float]],
    *,
    iou_threshold: float = RECALL_IOU,
) -> dict[str, float]:
    return detection_scores(predicted, expected, iou_threshold=iou_threshold)


def assert_recall_floors(
    predicted: list[dict[str, float]],
    expected: list[dict[str, float]],
    *,
    min_recall: float = RECALL_MIN,
    min_precision: float = PRECISION_MIN,
    iou_threshold: float = RECALL_IOU,
) -> dict[str, float]:
    scores = score_detections(predicted, expected, iou_threshold=iou_threshold)
    if scores["recall"] < min_recall or scores["precision"] < min_precision:
        raise AssertionError(
            "OCR synthetic recall failed "
            f"(recall={scores['recall']:.3f} precision={scores['precision']:.3f}; "
            f"floors recall>={min_recall} precision>={min_precision})"
        )
    return scores


def polygons_as_boxes(polygons: Any) -> list[dict[str, float]]:
    from media_analysis.features.ppocr import polygon_to_box

    boxes: list[dict[str, float]] = []
    for item in polygons:
        points = item.points if hasattr(item, "points") else item
        boxes.append(polygon_to_box(tuple(points)))
    return boxes
