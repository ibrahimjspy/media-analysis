from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from media_analysis.features.ppocr import postprocess_db_map, preprocess_det_image
from media_analysis.features.yolox import (
    DEFAULT_INPUT_NAME,
    DEFAULT_OUTPUT_NAME,
    INPUT_SIZE,
    decode_yolox_outputs,
    letterbox_preprocess,
)
from media_analysis.features.yunet import load_yunet_detector


@dataclass(frozen=True, slots=True)
class OnnxIoSpec:
    name: str
    rank: int
    dtype: str
    shape: tuple[int | str, ...]


def _numpy_dtype_name(type_str: str) -> str:
    mapping = {
        "tensor(float)": "float32",
        "tensor(float16)": "float16",
        "tensor(int64)": "int64",
        "tensor(int32)": "int32",
    }
    return mapping.get(type_str, type_str.removeprefix("tensor(").removesuffix(")"))


def _shape_tuple(raw: Any) -> tuple[int | str, ...]:
    if raw is None:
        return ()
    dims: list[int | str] = []
    for dim in raw:
        if dim is None:
            dims.append("?")
            continue
        try:
            dims.append(int(dim))
        except (TypeError, ValueError):
            dims.append("?")
    return tuple(dims)


def describe_onnx_io(session: Any) -> tuple[OnnxIoSpec, OnnxIoSpec]:
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_spec = OnnxIoSpec(
        name=input_meta.name,
        rank=len(input_meta.shape),
        dtype=_numpy_dtype_name(input_meta.type),
        shape=_shape_tuple(input_meta.shape),
    )
    output_spec = OnnxIoSpec(
        name=output_meta.name,
        rank=len(output_meta.shape),
        dtype=_numpy_dtype_name(output_meta.type),
        shape=_shape_tuple(output_meta.shape),
    )
    return input_spec, output_spec


def check_yolox_runtime_compat(model_path: Path) -> None:
    """Runtime compatibility: official raw YOLOX ONNX layout (not recall/parity)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_spec, output_spec = describe_onnx_io(session)

    if input_spec.name != DEFAULT_INPUT_NAME:
        raise AssertionError(f"YOLOX input name mismatch: {input_spec.name!r}")
    if input_spec.dtype != "float32":
        raise AssertionError(f"YOLOX input dtype mismatch: {input_spec.dtype}")
    if input_spec.rank != 4:
        raise AssertionError(f"YOLOX input rank mismatch: {input_spec.rank}")

    if output_spec.name != DEFAULT_OUTPUT_NAME:
        raise AssertionError(f"YOLOX output name mismatch: {output_spec.name!r}")
    if output_spec.dtype != "float32":
        raise AssertionError(f"YOLOX output dtype mismatch: {output_spec.dtype}")
    if output_spec.rank not in {2, 3}:
        raise AssertionError(f"YOLOX output rank mismatch: {output_spec.rank}")

    dummy_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    tensor, _meta = letterbox_preprocess(dummy_bgr, input_size=INPUT_SIZE)
    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)

    outputs = session.run([output_spec.name], {input_spec.name: tensor})[0]
    boxes, scores, class_ids = decode_yolox_outputs(outputs, input_size=INPUT_SIZE)
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert scores.ndim == 1
    assert class_ids.ndim == 1


def check_ppocr_db_runtime_compat(model_path: Path) -> None:
    """Runtime compatibility: DB probability-map detector (not recall/parity)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_spec, output_spec = describe_onnx_io(session)

    if input_spec.dtype != "float32":
        raise AssertionError(f"PP-OCR input dtype mismatch: {input_spec.dtype}")
    if input_spec.rank != 4:
        raise AssertionError(f"PP-OCR input rank mismatch: {input_spec.rank}")

    if output_spec.dtype != "float32":
        raise AssertionError(f"PP-OCR output dtype mismatch: {output_spec.dtype}")
    if output_spec.rank != 4:
        raise AssertionError(f"PP-OCR probability map rank mismatch: {output_spec.rank}")

    dummy_bgr = np.zeros((64, 128, 3), dtype=np.uint8)
    tensor, meta = preprocess_det_image(dummy_bgr, limit_side_len=128)
    assert tensor.dtype == np.float32
    assert tensor.ndim == 4

    prob_map = session.run([output_spec.name], {input_spec.name: tensor})[0]
    assert prob_map.dtype == np.float32
    assert prob_map.ndim == 4
    assert prob_map.shape[1] == 1

    # Viable dummy inference through DB postprocess (probability map → contours).
    polygons = postprocess_db_map(prob_map, meta)
    assert isinstance(polygons, list)


def check_yunet_opencv_runtime_compat(model_path: Path) -> None:
    """Runtime compatibility: OpenCV FaceDetectorYN load + dummy detect (not recall/parity)."""
    detector = load_yunet_detector(model_path, input_size=(320, 240))
    assert isinstance(detector._detector, cv2.FaceDetectorYN)

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    faces = detector.detect(frame)
    assert isinstance(faces, list)
