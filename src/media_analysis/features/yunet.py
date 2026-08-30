"""OpenCV YuNet face detector wrapper (face_detection_yunet_2023mar.onnx)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_CAPABILITY_VERSION = "yunet-2023mar"
YUNET_PREPROCESSING_VERSION = "yunet-native-resolution"

DEFAULT_SCORE_THRESHOLD = 0.6
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_TOP_K = 5000
DEFAULT_INPUT_SIZE = (320, 320)


@dataclass(frozen=True, slots=True)
class RawFaceDetection:
    x: float
    y: float
    width: float
    height: float
    score: float


class YuNetDetector(Protocol):
    def set_input_size(self, width: int, height: int) -> None: ...

    def detect(self, frame_bgr: np.ndarray) -> list[RawFaceDetection]: ...


@dataclass(slots=True)
class OpenCVYuNetDetector:
    """Wraps cv2.FaceDetectorYN with resize-aware setInputSize."""

    _detector: cv2.FaceDetectorYN
    _score_threshold: float
    _input_width: int
    _input_height: int

    def set_input_size(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("input size must be positive")
        if width != self._input_width or height != self._input_height:
            self._detector.setInputSize((width, height))
            self._input_width = width
            self._input_height = height

    def detect(self, frame_bgr: np.ndarray) -> list[RawFaceDetection]:
        height, width = frame_bgr.shape[:2]
        self.set_input_size(width, height)
        _, faces = self._detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return []
        out: list[RawFaceDetection] = []
        for row in faces:
            score = float(row[14])
            if score < self._score_threshold:
                continue
            out.append(
                RawFaceDetection(
                    x=float(row[0]),
                    y=float(row[1]),
                    width=float(row[2]),
                    height=float(row[3]),
                    score=score,
                )
            )
        return out


@dataclass
class FakeYuNetDetector:
    """Test double that never touches ONNX weights."""

    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    by_size: dict[tuple[int, int], list[RawFaceDetection]] | None = None
    sequence: list[list[RawFaceDetection]] | None = None
    input_sizes: list[tuple[int, int]] | None = None
    _call_index: int = 0

    def __post_init__(self) -> None:
        if self.input_sizes is None:
            self.input_sizes = []

    def set_input_size(self, width: int, height: int) -> None:
        assert self.input_sizes is not None
        self.input_sizes.append((width, height))

    def detect(self, frame_bgr: np.ndarray) -> list[RawFaceDetection]:
        height, width = frame_bgr.shape[:2]
        self.set_input_size(width, height)
        if self.sequence is not None:
            index = min(self._call_index, len(self.sequence) - 1)
            detections = list(self.sequence[index])
            self._call_index += 1
        elif self.by_size is not None:
            detections = list(self.by_size.get((width, height), []))
        else:
            detections = []
        return [item for item in detections if item.score >= self.score_threshold]


def load_yunet_detector(
    model_path: Path,
    *,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> OpenCVYuNetDetector:
    """Load YuNet via OpenCV FaceDetectorYN.create(model, config, input_size, ...)."""
    if not model_path.is_file():
        raise FileNotFoundError(f"YuNet model not found: {model_path}")
    width, height = input_size
    if width <= 0 or height <= 0:
        raise ValueError("input size must be positive")
    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (width, height),
        score_threshold,
        nms_threshold,
        top_k,
    )
    if detector is None:
        raise RuntimeError(f"FaceDetectorYN.create failed for {model_path}")
    return OpenCVYuNetDetector(
        _detector=detector,
        _score_threshold=score_threshold,
        _input_width=width,
        _input_height=height,
    )


def normalize_face_box(
    detection: RawFaceDetection,
    *,
    frame_width: int,
    frame_height: int,
) -> dict[str, float]:
    """Convert pixel-space YuNet output to a clipped 0..1 normalized box."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame size must be positive")
    box = {
        "x": detection.x / frame_width,
        "y": detection.y / frame_height,
        "width": detection.width / frame_width,
        "height": detection.height / frame_height,
    }
    return clip_normalized_box(box)


def clip_normalized_box(box: dict[str, float]) -> dict[str, float]:
    """Clip by endpoints so partially out-of-frame boxes shrink correctly."""
    x1 = max(0.0, min(1.0, box["x"]))
    y1 = max(0.0, min(1.0, box["y"]))
    x2 = max(0.0, min(1.0, box["x"] + box["width"]))
    y2 = max(0.0, min(1.0, box["y"] + box["height"]))
    if x2 <= x1 or y2 <= y1:
        return {"x": x1, "y": y1, "width": 0.0, "height": 0.0}
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
