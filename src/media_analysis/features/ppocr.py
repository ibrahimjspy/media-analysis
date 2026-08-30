"""PP-OCRv5 mobile DB-style text detection (ONNX Runtime, detection only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

PREPROCESSING_VERSION = "ppocrv5-db-det-1.0.0"
DEFAULT_LIMIT_SIDE_LEN = 960
DEFAULT_THRESH = 0.3
DEFAULT_BOX_THRESH = 0.6
DEFAULT_UNCLIP_RATIO = 1.5
DEFAULT_MIN_SIZE = 3
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OrtSessionLike(Protocol):
    def run(self, output_names: list[str], input_feed: dict[str, Any]) -> list[Any]: ...

    def get_inputs(self) -> list[Any]: ...

    def get_outputs(self) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class DetPreprocessMeta:
    src_height: int
    src_width: int
    ratio: float
    resized_height: int
    resized_width: int


@dataclass(frozen=True, slots=True)
class DetPolygon:
    points: tuple[tuple[float, float], ...]
    score: float


def resize_for_det(
    image: np.ndarray,
    *,
    limit_side_len: int = DEFAULT_LIMIT_SIDE_LEN,
) -> tuple[np.ndarray, DetPreprocessMeta]:
    """Resize for DB detector; longest side capped, dimensions rounded to multiples of 32."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("detector expects HxWx3 BGR image")
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError("image dimensions must be positive")

    ratio = 1.0
    if max(src_h, src_w) > limit_side_len:
        ratio = float(limit_side_len) / max(src_h, src_w)
    resized_h = max(int(round(src_h * ratio / 32) * 32), 32)
    resized_w = max(int(round(src_w * ratio / 32) * 32), 32)
    ratio_h = resized_h / src_h
    ratio_w = resized_w / src_w
    resized = cv2.resize(image, (resized_w, resized_h))
    meta = DetPreprocessMeta(
        src_height=src_h,
        src_width=src_w,
        ratio=(ratio_h + ratio_w) / 2.0,
        resized_height=resized_h,
        resized_width=resized_w,
    )
    return resized, meta


def normalize_det_input(image_bgr: np.ndarray) -> np.ndarray:
    """ImageNet-normalized NCHW float32 batch."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("detector expects HxWx3 BGR image")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]


def preprocess_det_image(
    image_bgr: np.ndarray,
    *,
    limit_side_len: int = DEFAULT_LIMIT_SIDE_LEN,
) -> tuple[np.ndarray, DetPreprocessMeta]:
    resized, meta = resize_for_det(image_bgr, limit_side_len=limit_side_len)
    tensor = normalize_det_input(resized)
    return tensor, meta


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    """Order four points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def _box_score_fast(prob_map: np.ndarray, box: np.ndarray) -> float:
    h, w = prob_map.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [box.astype(np.int32)], 1)
    return float(cv2.mean(prob_map, mask)[0])


def _polygon_area(points: np.ndarray) -> float:
    contour = points.reshape(-1, 1, 2).astype(np.float32)
    return float(abs(cv2.contourArea(contour)))


def _polygon_perimeter(points: np.ndarray) -> float:
    contour = points.reshape(-1, 1, 2).astype(np.float32)
    return float(cv2.arcLength(contour, True))


def unclip_quad(box: np.ndarray, unclip_ratio: float) -> np.ndarray:
    """Standard DB polygon offset via pyclipper; preserve quads via minAreaRect."""
    try:
        import pyclipper
    except ImportError as exc:
        raise RuntimeError("pyclipper is required for DB unclip") from exc

    contour = box.reshape(-1, 2).astype(np.float64)
    area = _polygon_area(contour)
    length = _polygon_perimeter(contour)
    if length <= 0:
        return box.astype(np.float32)

    distance = area * unclip_ratio / length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(
        contour.tolist(),
        pyclipper.JT_ROUND,
        pyclipper.ET_CLOSEDPOLYGON,
    )
    expanded = offset.Execute(distance)
    if not expanded:
        return box.astype(np.float32)

    points = np.array(expanded[0], dtype=np.float32)
    if len(points) < 4:
        return box.astype(np.float32)
    if len(points) != 4:
        rect = cv2.minAreaRect(points)
        points = cv2.boxPoints(rect).astype(np.float32)
    return _order_quad_points(points)


def _map_to_src(points: np.ndarray, meta: DetPreprocessMeta) -> np.ndarray:
    scale_x = meta.src_width / meta.resized_width
    scale_y = meta.src_height / meta.resized_height
    mapped = points.copy()
    mapped[:, 0] *= scale_x
    mapped[:, 1] *= scale_y
    mapped[:, 0] = np.clip(mapped[:, 0], 0, meta.src_width)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, meta.src_height)
    return mapped


def clip_normalized_box(box: dict[str, float]) -> dict[str, float]:
    x0 = max(0.0, min(1.0, box["x"]))
    y0 = max(0.0, min(1.0, box["y"]))
    x1 = max(0.0, min(1.0, box["x"] + box["width"]))
    y1 = max(0.0, min(1.0, box["y"] + box["height"]))
    return {"x": x0, "y": y0, "width": max(0.0, x1 - x0), "height": max(0.0, y1 - y0)}


def polygon_to_box(points: tuple[tuple[float, float], ...]) -> dict[str, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return clip_normalized_box({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0})


def normalize_polygon(
    points: np.ndarray,
    meta: DetPreprocessMeta,
) -> tuple[tuple[float, float], ...]:
    mapped = _map_to_src(points, meta)
    return tuple(
        (float(x / meta.src_width), float(y / meta.src_height)) for x, y in mapped
    )


def postprocess_db_map(
    prob_map: np.ndarray,
    meta: DetPreprocessMeta,
    *,
    thresh: float = DEFAULT_THRESH,
    box_thresh: float = DEFAULT_BOX_THRESH,
    unclip_ratio: float = DEFAULT_UNCLIP_RATIO,
    min_size: int = DEFAULT_MIN_SIZE,
) -> list[DetPolygon]:
    """DB-style contour extraction from a single-channel probability map."""
    if prob_map.ndim == 4:
        prob_map = prob_map[0, 0]
    elif prob_map.ndim == 3:
        prob_map = prob_map[0]
    elif prob_map.ndim != 2:
        raise ValueError(f"unexpected probability map shape: {prob_map.shape}")

    binary = (prob_map > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    results: list[DetPolygon] = []

    for contour in contours:
        if contour is None or len(contour) < 4:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 4:
            continue
        points = approx.reshape(-1, 2)
        if len(points) > 4:
            rect = cv2.minAreaRect(points.astype(np.float32))
            points = cv2.boxPoints(rect)
        if len(points) != 4:
            continue
        mini_box = _order_quad_points(points.astype(np.float32))
        if min(float(np.ptp(mini_box[:, 0])), float(np.ptp(mini_box[:, 1]))) < min_size:
            continue
        score = _box_score_fast(prob_map, mini_box)
        if score < box_thresh:
            continue
        expanded = unclip_quad(mini_box, unclip_ratio)
        if len(expanded) != 4:
            continue
        expanded = _order_quad_points(expanded.astype(np.float32))
        if min(float(np.ptp(expanded[:, 0])), float(np.ptp(expanded[:, 1]))) < min_size:
            continue
        norm = normalize_polygon(expanded, meta)
        results.append(DetPolygon(points=norm, score=score))

    return results


def create_det_session(model_path: Path) -> Any:
    """Load vendored ONNX model via ONNX Runtime CPU."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for OCR detection") from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _session_io_names(session: OrtSessionLike) -> tuple[str, str]:
    return session.get_inputs()[0].name, session.get_outputs()[0].name


def run_det_inference(
    session: OrtSessionLike,
    image_bgr: np.ndarray,
    *,
    limit_side_len: int = DEFAULT_LIMIT_SIDE_LEN,
) -> list[DetPolygon]:
    tensor, meta = preprocess_det_image(image_bgr, limit_side_len=limit_side_len)
    input_name, output_name = _session_io_names(session)
    outputs = session.run([output_name], {input_name: tensor})
    if not outputs:
        raise ValueError("detector session returned no outputs")
    return postprocess_db_map(outputs[0], meta)
