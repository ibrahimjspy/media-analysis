"""YOLOX-Tiny ONNX person detection at 416×416 (official Megvii export layout).

Preprocessing matches ``yolox.data.data_augment.preproc`` used by the official
ONNX Runtime demo: BGR pixels, top-left letterbox in a 114-filled canvas, CHW
float32, no RGB channel swap, no divide-by-255.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

INPUT_SIZE = 416
COCO_PERSON_CLASS_ID = 0
LETTERBOX_FILL = 114.0
DEFAULT_INPUT_NAME = "images"
DEFAULT_OUTPUT_NAME = "output"
YOLOX_PREPROCESSING = "letterbox-416"
# Official Megvii preproc: BGR uint8 → top-left in 114 canvas → CHW float32 (no RGB swap, no /255).


@dataclass(frozen=True, slots=True)
class LetterboxMeta:
    """Mapping metadata from official YOLOX preproc (top-left letterbox, scale only)."""

    scale: float
    orig_width: int
    orig_height: int
    input_width: int = INPUT_SIZE
    input_height: int = INPUT_SIZE

    @property
    def pad_left(self) -> float:
        return 0.0

    @property
    def pad_top(self) -> float:
        return 0.0


# ByteTrack-style thresholds (person class only).
DEFAULT_CONF_THRESH = 0.25
DEFAULT_NMS_IOU_THRESH = 0.45
DEFAULT_TRACK_THRESH = 0.5
DEFAULT_LOW_DET_THRESH = 0.1


@dataclass(frozen=True, slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int = COCO_PERSON_CLASS_ID


class OnnxInferenceSession(Protocol):
    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]: ...


def letterbox_preprocess(
    image_bgr: np.ndarray,
    *,
    input_size: int = INPUT_SIZE,
) -> tuple[np.ndarray, LetterboxMeta]:
    """Official Megvii YOLOX preproc: BGR, top-left in 114 canvas, CHW float32, no /255."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected H×W×3 BGR image")
    orig_h, orig_w = image_bgr.shape[:2]
    scale = min(input_size / orig_h, input_size / orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded = np.ones((input_size, input_size, 3), dtype=np.uint8) * int(LETTERBOX_FILL)
    padded[:new_h, :new_w] = resized
    chw = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    tensor = chw[np.newaxis, ...]
    meta = LetterboxMeta(
        scale=scale,
        orig_width=orig_w,
        orig_height=orig_h,
        input_width=input_size,
        input_height=input_size,
    )
    return tensor, meta


def _build_grids(input_size: int) -> tuple[np.ndarray, np.ndarray]:
    strides = (8, 16, 32)
    grids: list[np.ndarray] = []
    expanded_strides: list[np.ndarray] = []
    for stride in strides:
        hsize = input_size // stride
        wsize = input_size // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((1, grid.shape[1], 1), stride, dtype=np.float32))
    return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)


def decode_yolox_outputs(
    outputs: np.ndarray,
    *,
    input_size: int = INPUT_SIZE,
    person_class_id: int = COCO_PERSON_CLASS_ID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode official YOLOX ONNX tensor to xyxy boxes, scores, class ids."""
    if outputs.ndim == 3:
        predictions = outputs[0]
    elif outputs.ndim == 2:
        predictions = outputs
    else:
        raise ValueError(f"unexpected YOLOX output rank: {outputs.ndim}")

    decoded = predictions.astype(np.float32, copy=True)
    grids, expanded_strides = _build_grids(input_size)
    decoded[:, :2] = (decoded[:, :2] + grids[0]) * expanded_strides[0]
    decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * expanded_strides[0]

    boxes_cxcywh = decoded[:, :4]
    obj_conf = decoded[:, 4]
    class_probs = decoded[:, 5:]
    if class_probs.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    class_ids = np.argmax(class_probs, axis=1)
    class_scores = class_probs[np.arange(class_probs.shape[0]), class_ids]
    scores = obj_conf * class_scores

    person_mask = class_ids == person_class_id
    boxes_cxcywh = boxes_cxcywh[person_mask]
    scores = scores[person_mask]
    class_ids = class_ids[person_mask]

    if boxes_cxcywh.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    cx, cy, bw, bh = boxes_cxcywh.T
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    return boxes_xyxy, scores.astype(np.float32), class_ids.astype(np.int64)


def nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float = DEFAULT_NMS_IOU_THRESH,
    max_detections: int = 100,
) -> np.ndarray:
    """Greedy NMS; returns indices kept in score-descending order."""
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou_xyxy(boxes[current], boxes[rest])
        order = rest[ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def _iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_box = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    area_boxes = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


def map_boxes_to_normalized(
    boxes_xyxy: np.ndarray,
    meta: LetterboxMeta,
) -> list[dict[str, float]]:
    """Map letterbox-space xyxy to normalized canonical 0..1 xywh (divide by scale only)."""
    if boxes_xyxy.size == 0:
        return []
    boxes = boxes_xyxy.astype(np.float32, copy=True) / meta.scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, meta.orig_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, meta.orig_height)

    out: list[dict[str, float]] = []
    for x1, y1, x2, y2 in boxes:
        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))
        x = float(x1) / meta.orig_width
        y = float(y1) / meta.orig_height
        out.append(
            {
                "x": _clamp01(x),
                "y": _clamp01(y),
                "width": _clamp01(w / meta.orig_width),
                "height": _clamp01(h / meta.orig_height),
            }
        )
    return out


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def filter_detections(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    conf_thresh: float = DEFAULT_CONF_THRESH,
    nms_iou: float = DEFAULT_NMS_IOU_THRESH,
) -> list[Detection]:
    if boxes_xyxy.size == 0:
        return []
    mask = scores >= conf_thresh
    boxes_xyxy = boxes_xyxy[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    if boxes_xyxy.size == 0:
        return []
    keep = nms_xyxy(boxes_xyxy, scores, iou_threshold=nms_iou)
    return [
        Detection(
            x1=float(boxes_xyxy[i, 0]),
            y1=float(boxes_xyxy[i, 1]),
            x2=float(boxes_xyxy[i, 2]),
            y2=float(boxes_xyxy[i, 3]),
            score=float(scores[i]),
            class_id=int(class_ids[i]),
        )
        for i in keep
    ]


@dataclass(slots=True)
class YoloxDetector:
    session: OnnxInferenceSession
    input_name: str = DEFAULT_INPUT_NAME
    output_name: str = DEFAULT_OUTPUT_NAME
    input_size: int = INPUT_SIZE
    conf_thresh: float = DEFAULT_CONF_THRESH
    nms_iou: float = DEFAULT_NMS_IOU_THRESH

    def detect(self, image_bgr: np.ndarray) -> tuple[list[Detection], LetterboxMeta]:
        tensor, meta = letterbox_preprocess(image_bgr, input_size=self.input_size)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})[0]
        boxes, scores, class_ids = decode_yolox_outputs(outputs, input_size=self.input_size)
        detections = filter_detections(
            boxes,
            scores,
            class_ids,
            conf_thresh=self.conf_thresh,
            nms_iou=self.nms_iou,
        )
        return detections, meta

    def detect_normalized(
        self,
        image_bgr: np.ndarray,
    ) -> tuple[list[tuple[Detection, dict[str, float]]], LetterboxMeta]:
        detections, meta = self.detect(image_bgr)
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
        norm_boxes = map_boxes_to_normalized(boxes, meta)
        return list(zip(detections, norm_boxes, strict=True)), meta


def create_yolox_session(model_path: Path) -> OnnxInferenceSession:
    """Load ONNX Runtime CPU session from a vendored artifact path (never downloads)."""
    import onnxruntime as ort

    if not model_path.is_file():
        raise FileNotFoundError(f"YOLOX model not found: {model_path}")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
