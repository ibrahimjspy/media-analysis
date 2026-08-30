"""Per-shot exposure and color measurements (v1.2)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from media_analysis.decode import ProbedMedia
from media_analysis.features.measure_common import (
    FrameProvider,
    PriorFactsInput,
    ShotsInput,
    representative_sample_frames,
    resolve_shots,
    shot_ranges,
)

# Provisional until benchmark; suffix marks benchmark-gated policy.
EXPOSURE_ANALYZER_VERSION = "bt709-linear-xyz-mccamy-provisional-1.0.0"
EXPOSURE_SAMPLING_POLICY_VERSION = "representative-5-per-shot-65536px-provisional-1.0.0"
SATURATION_COLOR_SPACE = "opencv_bgr_to_hsv_s_mean"
LUMA_COLOR_SPACE = "bt709_linear_rgb_after_srgb_eotf"
DOMINANT_COLOR_METHOD = "rgb_5bit_quantize_merge-provisional-1.0.0"
COLOR_TEMPERATURE_METHOD = "mccamy_from_bt709_xyz_xy-provisional-1.0.0"

# Provisional until benchmark.
COLOR_TEMP_MIN_SATURATION = 0.08
COLOR_TEMP_MIN_LUMA = 0.05
COLOR_TEMP_MAX_LUMA = 0.95
DOMINANT_COLOR_COUNT = 5
DOMINANT_MIN_FRACTION = 0.02
REPRESENTATIVE_SAMPLES_PER_SHOT = 5
MAX_LUMA_SAMPLE_PIXELS_PER_FRAME = 65536

# BT.709 luma on linear RGB in [0, 1] after sRGB/BT.709 inverse EOTF:
#   Y = 0.2126729*R + 0.7151522*G + 0.0721750*B
BT709_LUMA_COEFFICIENTS = (0.2126729, 0.7151522, 0.0721750)
BT709_XYZ_MATRIX = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
# Normalized contrast is the spread between luminance percentiles:
#   contrast = p95(Y) - p05(Y)
CONTRAST_FORMULA = "p95_luma_minus_p05_luma"


@dataclass(frozen=True, slots=True)
class ExposureShotMetrics:
    shot_index: int
    mean_luma: float
    p05_luma: float
    p95_luma: float
    contrast: float
    saturation: float
    estimated_color_temperature_k: int | None
    dominant_colors: list[dict[str, float | str]]


@dataclass(frozen=True, slots=True)
class ExposureAnalyzeResult:
    per_shot: list[dict[str, Any]]
    status: Literal["completed", "failed"] = "completed"
    version: str = EXPOSURE_ANALYZER_VERSION
    warning_codes: tuple[str, ...] = ()


def srgb_uint8_to_linear(value: np.ndarray) -> np.ndarray:
    """Inverse sRGB/BT.709 transfer on uint8 channels -> linear [0, 1]."""
    normalized = value.astype(np.float64) / 255.0
    return np.where(
        normalized <= 0.04045,
        normalized / 12.92,
        ((normalized + 0.055) / 1.055) ** 2.4,
    )


def bgr_uint8_to_linear_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = frame_bgr[:, :, ::-1]
    linear = srgb_uint8_to_linear(rgb)
    return linear.astype(np.float64)


def downsample_for_measurement(frame_bgr: np.ndarray, *, max_pixels: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    pixel_count = height * width
    if pixel_count <= max_pixels:
        return frame_bgr
    stride = int(np.ceil(np.sqrt(pixel_count / max_pixels)))
    return frame_bgr[::stride, ::stride]


def bt709_luma_from_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Return per-pixel BT.709 linear luma normalized to [0, 1]."""
    linear_rgb = bgr_uint8_to_linear_rgb(frame_bgr)
    r, g, b = linear_rgb[:, :, 0], linear_rgb[:, :, 1], linear_rgb[:, :, 2]
    return (
        BT709_LUMA_COEFFICIENTS[0] * r
        + BT709_LUMA_COEFFICIENTS[1] * g
        + BT709_LUMA_COEFFICIENTS[2] * b
    )


def sample_luma_values(frame_bgr: np.ndarray) -> np.ndarray:
    """Bounded per-frame luma samples; never materializes full-resolution Python floats."""
    sampled = downsample_for_measurement(
        frame_bgr,
        max_pixels=MAX_LUMA_SAMPLE_PIXELS_PER_FRAME,
    )
    return bt709_luma_from_bgr(sampled).ravel()


def mean_saturation_from_bgr(frame_bgr: np.ndarray) -> float:
    """Mean HSV saturation channel normalized to [0, 1]."""
    sampled = downsample_for_measurement(
        frame_bgr,
        max_pixels=MAX_LUMA_SAMPLE_PIXELS_PER_FRAME,
    )
    hsv = cv2.cvtColor(sampled, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]) / 255.0)


def linear_rgb_to_xy(linear_rgb: np.ndarray) -> tuple[float, float]:
    """BT.709 XYZ chromaticity from mean linear RGB."""
    mean_rgb = linear_rgb.reshape(-1, 3).mean(axis=0)
    xyz = BT709_XYZ_MATRIX @ mean_rgb
    total = float(xyz.sum())
    if total <= 1e-12:
        return 0.0, 0.0
    return float(xyz[0] / total), float(xyz[1] / total)


def estimate_color_temperature_k(frame_bgr: np.ndarray) -> int | None:
    """McCamy-style estimate from BT.709 XYZ chromaticity; null when unreliable."""
    linear_rgb = bgr_uint8_to_linear_rgb(
        downsample_for_measurement(frame_bgr, max_pixels=MAX_LUMA_SAMPLE_PIXELS_PER_FRAME)
    )
    luma = float(np.mean(bt709_luma_from_bgr(
        downsample_for_measurement(frame_bgr, max_pixels=MAX_LUMA_SAMPLE_PIXELS_PER_FRAME)
    )))
    saturation = mean_saturation_from_bgr(frame_bgr)
    if saturation < COLOR_TEMP_MIN_SATURATION:
        return None
    if luma < COLOR_TEMP_MIN_LUMA or luma > COLOR_TEMP_MAX_LUMA:
        return None
    mean_rgb = linear_rgb.reshape(-1, 3).mean(axis=0)
    if np.max(mean_rgb) - np.min(mean_rgb) < 0.02:
        return None
    x, y = linear_rgb_to_xy(linear_rgb)
    if y <= 1e-6:
        return None
    n = (x - 0.3320) / (0.1858 - y + 1e-9)
    cct = 449.0 * n**3 + 3525.0 * n**2 + 6813.5 * n + 5527.0
    if not np.isfinite(cct):
        return None
    rounded = int(round(cct))
    if rounded < 1000 or rounded > 25000:
        return None
    return rounded


def dominant_colors_from_bgr(
    frame_bgr: np.ndarray,
    *,
    max_colors: int = DOMINANT_COLOR_COUNT,
) -> list[dict]:
    """Deterministic 5-bit RGB quantization with fixed merge distance."""
    sampled = downsample_for_measurement(frame_bgr, max_pixels=MAX_LUMA_SAMPLE_PIXELS_PER_FRAME)
    sampled = sampled[:: max(1, sampled.shape[0] // 64), :: max(1, sampled.shape[1] // 64), :]
    pixels = sampled.reshape(-1, 3)
    if pixels.size == 0:
        return []
    quant = (pixels // 8).astype(np.int32)
    bins: dict[tuple[int, int, int], int] = {}
    for b, g, r in quant:
        key = (int(r), int(g), int(b))
        bins[key] = bins.get(key, 0) + 1
    total = sum(bins.values())
    ranked = sorted(bins.items(), key=lambda item: (-item[1], item[0]))
    colors: list[dict] = []
    for (qr, qg, qb), count in ranked:
        if count / total < DOMINANT_MIN_FRACTION:
            continue
        center_r = min(255, qr * 8 + 4)
        center_g = min(255, qg * 8 + 4)
        center_b = min(255, qb * 8 + 4)
        colors.append(
            {
                "hex": f"#{center_r:02x}{center_g:02x}{center_b:02x}",
                "fraction": round(count / total, 4),
            }
        )
        if len(colors) >= max_colors:
            break
    return colors


def aggregate_shot_metrics(
    *,
    shot_index: int,
    luma_values: np.ndarray,
    saturation_values: list[float],
    rgb_frames: list[np.ndarray],
) -> ExposureShotMetrics:
    if luma_values.size == 0:
        raise ValueError(f"shot {shot_index} has no decoded exposure samples")
    p05 = float(np.percentile(luma_values, 5))
    p95 = float(np.percentile(luma_values, 95))
    mean = float(np.mean(luma_values))
    contrast = p95 - p05
    saturation = float(np.mean(saturation_values)) if saturation_values else 0.0
    stacked = np.concatenate(rgb_frames, axis=0)
    temp_k = estimate_color_temperature_k(stacked)
    dominant = dominant_colors_from_bgr(stacked)
    return ExposureShotMetrics(
        shot_index=shot_index,
        mean_luma=round(mean, 4),
        p05_luma=round(p05, 4),
        p95_luma=round(p95, 4),
        contrast=round(contrast, 4),
        saturation=round(saturation, 4),
        estimated_color_temperature_k=temp_k,
        dominant_colors=dominant,
    )


def metrics_to_dict(metrics: ExposureShotMetrics) -> dict[str, Any]:
    return {
        "shotIndex": metrics.shot_index,
        "meanLuma": metrics.mean_luma,
        "p05Luma": metrics.p05_luma,
        "p95Luma": metrics.p95_luma,
        "contrast": metrics.contrast,
        "saturation": metrics.saturation,
        "estimatedColorTemperatureK": metrics.estimated_color_temperature_k,
        "dominantColors": metrics.dominant_colors,
    }


def analyze_exposure(
    path: Path,
    media: ProbedMedia,
    *,
    shots: ShotsInput = None,
    prior_facts: PriorFactsInput = None,
    fill_only: bool = False,
    frame_provider: FrameProvider | None = None,
    cancel_check: Callable[[], None] | None = None,
    samples_per_shot: int = REPRESENTATIVE_SAMPLES_PER_SHOT,
) -> ExposureAnalyzeResult:
    """Measure per-shot exposure/color facts against canonical half-open shots."""
    resolved = resolve_shots(
        prior_facts=prior_facts,
        supplied_shots=shots,
        frame_count=media.frame_count,
        fill_only=fill_only,
    )
    ranges = shot_ranges(resolved, media.frame_count)
    if not ranges:
        return ExposureAnalyzeResult(per_shot=[], status="completed")

    reader = _OpenCvFrameProvider(path) if frame_provider is None else frame_provider
    owns_reader = frame_provider is None
    per_shot: list[dict[str, Any]] = []

    try:
        for shot_index, (start, end) in enumerate(ranges):
            if cancel_check:
                cancel_check()
            sample_frames = representative_sample_frames(
                start,
                end,
                max_samples=samples_per_shot,
            )
            luma_chunks: list[np.ndarray] = []
            saturation_values: list[float] = []
            rgb_frames: list[np.ndarray] = []
            decoded = 0
            for frame_index in sample_frames:
                if cancel_check:
                    cancel_check()
                frame = _read_frame(reader, frame_index)
                if frame is None:
                    continue
                decoded += 1
                luma_chunks.append(sample_luma_values(frame))
                saturation_values.append(mean_saturation_from_bgr(frame))
                rgb_frames.append(
                    downsample_for_measurement(
                        frame,
                        max_pixels=MAX_LUMA_SAMPLE_PIXELS_PER_FRAME,
                    )
                )
            if decoded == 0:
                raise RuntimeError(f"could not decode exposure samples for shot {shot_index}")
            luma_values = np.concatenate(luma_chunks)
            metrics = aggregate_shot_metrics(
                shot_index=shot_index,
                luma_values=luma_values,
                saturation_values=saturation_values,
                rgb_frames=rgb_frames,
            )
            per_shot.append(metrics_to_dict(metrics))
    finally:
        if owns_reader and isinstance(reader, _OpenCvFrameProvider):
            reader.close()

    return ExposureAnalyzeResult(per_shot=per_shot, status="completed")


@dataclass(slots=True)
class _OpenCvFrameProvider:
    path: Path
    _capture: cv2.VideoCapture | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Could not open video for exposure analysis")
        self._capture = capture

    def __call__(self, frame_index: int) -> np.ndarray | None:
        if self._capture is None:
            return None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def read(self, frame_index: int) -> np.ndarray | None:
        return self(frame_index)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def _read_frame(
    reader: FrameProvider | _OpenCvFrameProvider,
    frame_index: int,
) -> np.ndarray | None:
    if callable(reader):
        return reader(frame_index)
    return reader.read(frame_index)


def exposure_policy_metadata() -> dict[str, str | float | int | tuple[float, ...]]:
    """Versioned formulas for cache identity / provenance wiring by parent."""
    return {
        "analyzerVersion": EXPOSURE_ANALYZER_VERSION,
        "samplingPolicyVersion": EXPOSURE_SAMPLING_POLICY_VERSION,
        "lumaColorSpace": LUMA_COLOR_SPACE,
        "saturationColorSpace": SATURATION_COLOR_SPACE,
        "contrastFormula": CONTRAST_FORMULA,
        "dominantColorMethod": DOMINANT_COLOR_METHOD,
        "colorTemperatureMethod": COLOR_TEMPERATURE_METHOD,
        "bt709Coefficients": BT709_LUMA_COEFFICIENTS,
        "colorTempMinSaturation": COLOR_TEMP_MIN_SATURATION,
        "maxLumaSamplePixelsPerFrame": MAX_LUMA_SAMPLE_PIXELS_PER_FRAME,
        "provisional": True,
    }
