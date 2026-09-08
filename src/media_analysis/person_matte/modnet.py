"""MODNet ONNX preprocessing and postprocessing with injected ORT session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from media_analysis.errors import INVALID_REQUEST, AnalyzeError
from media_analysis.person_matte.constants import (
    MODNET_INPUT_NAME,
    MODNET_MAX_ALIGNED_DIM,
    MODNET_MAX_ASPECT_RATIO,
    MODNET_MIN_ALIGNED_DIM,
    MODNET_OUTPUT_NAME,
    MODNET_PREPROCESSING_VERSION,
    MODNET_REF_SIZE,
)


class OnnxInferenceSession(Protocol):
    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]: ...


@dataclass
class ReferenceModnetSession:
    """Deterministic reference stub for closed-gate matte images."""

    input_name: str = MODNET_INPUT_NAME
    output_name: str = MODNET_OUTPUT_NAME

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        tensor = input_feed[self.input_name]
        mean = float(tensor.mean())
        if mean < -0.5:
            value = 0.0
        elif mean > 0.5:
            value = 1.0
        else:
            value = 128.0 / 255.0
        _, _, height, width = tensor.shape
        alpha = np.full((1, 1, height, width), value, dtype=np.float32)
        return [alpha]


def _align_dim(value: int) -> int:
    if value < MODNET_MIN_ALIGNED_DIM:
        return MODNET_MIN_ALIGNED_DIM
    aligned = value - value % 32
    if aligned < MODNET_MIN_ALIGNED_DIM:
        aligned = MODNET_MIN_ALIGNED_DIM
    return aligned


def compute_modnet_input_size(
    im_h: int,
    im_w: int,
    ref_size: int = MODNET_REF_SIZE,
) -> tuple[int, int]:
    """Official MODNet ONNX sizing with safe 32-pixel alignment bounds."""
    if im_h <= 0 or im_w <= 0:
        raise AnalyzeError(INVALID_REQUEST, "MODNet input dimensions must be positive")
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        else:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    im_rw = _align_dim(im_rw)
    im_rh = _align_dim(im_rh)
    overflow = max(im_rw / MODNET_MAX_ALIGNED_DIM, im_rh / MODNET_MAX_ALIGNED_DIM, 1.0)
    if overflow > 1.0:
        im_rw = _align_dim(int(im_rw / overflow))
        im_rh = _align_dim(int(im_rh / overflow))
    aspect = max(im_rw / im_rh, im_rh / im_rw)
    if aspect > MODNET_MAX_ASPECT_RATIO:
        raise AnalyzeError(INVALID_REQUEST, "MODNet aspect ratio exceeds supported bounds")
    if im_rw > MODNET_MAX_ALIGNED_DIM or im_rh > MODNET_MAX_ALIGNED_DIM:
        raise AnalyzeError(INVALID_REQUEST, "MODNet tensor dimensions exceed supported bounds")
    return im_rw, im_rh


def scale_factors(im_h: int, im_w: int, ref_size: int = MODNET_REF_SIZE) -> tuple[float, float]:
    im_rw, im_rh = compute_modnet_input_size(im_h, im_w, ref_size)
    return im_rw / im_w, im_rh / im_h


def preprocess_modnet(image_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
    """BGR uint8 H×W×3 → NCHW float32 and original (h, w) for resize-back."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected H×W×3 BGR image")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    im_h, im_w = rgb.shape[:2]
    x_scale, y_scale = scale_factors(im_h, im_w)
    resized = cv2.resize(rgb, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) - 127.5) / 127.5
    chw = np.transpose(normalized, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0).astype(np.float32)
    return batch, im_h, im_w


def postprocess_modnet(
    output: np.ndarray,
    *,
    orig_height: int,
    orig_width: int,
) -> np.ndarray:
    """Model output → uint8 alpha H×W; resize in float space before quantization."""
    matte = np.squeeze(output).astype(np.float32)
    if matte.ndim != 2:
        raise ValueError("MODNet output must be 2D alpha")
    if matte.shape != (orig_height, orig_width):
        matte = cv2.resize(matte, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
    return np.clip(matte * 255.0, 0.0, 255.0).astype(np.uint8)


def infer_modnet_alpha(
    session: OnnxInferenceSession,
    image_bgr: np.ndarray,
    *,
    input_name: str = MODNET_INPUT_NAME,
    output_name: str = MODNET_OUTPUT_NAME,
) -> np.ndarray:
    tensor, orig_h, orig_w = preprocess_modnet(image_bgr)
    outputs = session.run([output_name], {input_name: tensor})
    return postprocess_modnet(outputs[0], orig_height=orig_h, orig_width=orig_w)


def load_modnet_session(
    model_path: Path,
    *,
    execution_provider: str = "auto",
    engine_cache_dir: Path | None = None,
    inference_threads: int = 1,
) -> Any:
    """Load MODNet with TensorRT/CUDA preference and FP16 engine optimization."""
    if not model_path.is_file():
        raise FileNotFoundError(f"MODNet model not found: {model_path}")
    if not 1 <= inference_threads <= 8:
        raise ValueError("matte inference_threads must be between 1 and 8")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for MODNet") from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = inference_threads
    available = set(ort.get_available_providers())
    normalized = execution_provider.strip().lower()
    if normalized not in {"auto", "tensorrt", "cuda", "cpu"}:
        raise ValueError("unknown matte execution provider")
    required = {
        "tensorrt": "TensorrtExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }.get(normalized)
    if required is not None and required not in available:
        raise RuntimeError(f"requested {normalized} execution provider is unavailable")

    providers: list[Any] = []
    if normalized in {"auto", "tensorrt"} and "TensorrtExecutionProvider" in available:
        cache_dir = engine_cache_dir or Path("/var/tmp/media-analysis/tensorrt")
        cache_dir.mkdir(parents=True, exist_ok=True)
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache_dir),
                },
            )
        )
    if normalized in {"auto", "tensorrt", "cuda"} and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if normalized == "cpu" or (normalized == "auto" and not providers):
        providers.append("CPUExecutionProvider")
    if normalized != "auto" and not providers:
        raise RuntimeError(f"requested {normalized} execution provider is unavailable")
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers,
    )
    # The ORT package can advertise an EP whose shared libraries fail to load.
    # Check the actual session as well as the package's compiled-in providers.
    if required is not None and required not in session.get_providers():
        raise RuntimeError(f"requested {normalized} execution provider failed to initialize")
    if required is not None:
        session.disable_fallback()
    return session


def preprocessing_version() -> str:
    return MODNET_PREPROCESSING_VERSION
