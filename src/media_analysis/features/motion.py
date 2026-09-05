"""Global motion estimation with pyramidal Lucas-Kanade and affine RANSAC.

Provisional (benchmark-gated) policy constants are encoded in
``MOTION_ANALYZER_VERSION`` and ``MotionAnalysisConfig``. Thresholds are
deterministic heuristics, not calibrated probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from media_analysis.decode import ProbedMedia
from media_analysis.errors import DECODE_FAILED, AnalyzeError
from media_analysis.features.measure_common import FrameProvider, ReadableFrameProvider

# Half-open aggregation window size (canonical frames).
WINDOW_FRAMES = 15
# Sample every N frames for pairwise optical-flow estimation.
SAMPLE_STRIDE = 2
# Max forward/backward LK displacement (px) for track consistency.
FB_CONSISTENCY_PX = 3.0
# Minimum inlier ratio for a pairwise estimate to contribute to coherence.
MIN_INLIER_RATIO = 0.55
# goodFeaturesToTrack corner budget and quality.
MAX_CORNERS = 120
CORNER_QUALITY = 0.01
CORNER_MIN_DISTANCE_PX = 12
# RANSAC reprojection threshold (px) for partial affine fit.
RANSAC_REPROJ_PX = 3.0
# Camera-move heuristic thresholds (normalized translation / scale).
STATIC_MAG = 0.0018
PAN_TILT_MAG = 0.0025
HANDHELD_MAG = 0.0020
WHIP_MAG = 0.028
SCALE_PUSH = 1.004
SCALE_PULL = 0.996
COHERENCE_DIRECTION = 0.52
COHERENCE_MOTION = 0.48
WHIP_COHERENCE = 0.42
HANDHELD_RESIDUAL = 0.0035
HANDHELD_COHERENCE = 0.54
# Safe border margin as a fraction of frame width/height.
BORDER_MARGIN_FRAC = 1 / 12
SUBJECT_BOX_PAD_FRAC = 1 / 40


def build_motion_analyzer_version() -> str:
    return (
        "opencv-lk-ransac-aff-provisional-v1.2.0"
        f"-w{WINDOW_FRAMES}-s{SAMPLE_STRIDE}-fb{FB_CONSISTENCY_PX}-ir{MIN_INLIER_RATIO}"
        f"-sm{STATIC_MAG}-pt{PAN_TILT_MAG}-hm{HANDHELD_MAG}-wm{WHIP_MAG}"
        f"-sp{SCALE_PUSH}-sl{SCALE_PULL}-cd{COHERENCE_DIRECTION}-cm{COHERENCE_MOTION}"
        f"-wc{WHIP_COHERENCE}-hr{HANDHELD_RESIDUAL}-hc{HANDHELD_COHERENCE}"
    )


MOTION_ANALYZER_VERSION = build_motion_analyzer_version()

NormalizedBox = dict[str, float]
SubjectBoxesByFrame = dict[int, list[NormalizedBox]]


@dataclass(frozen=True, slots=True)
class MotionAnalysisConfig:
    window_frames: int = WINDOW_FRAMES
    sample_stride: int = SAMPLE_STRIDE
    fb_consistency_px: float = FB_CONSISTENCY_PX
    min_inlier_ratio: float = MIN_INLIER_RATIO
    analyzer_version: str = MOTION_ANALYZER_VERSION

    def __post_init__(self) -> None:
        if self.window_frames < 1:
            raise ValueError("window_frames must be >= 1")
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be >= 1")
        if self.fb_consistency_px <= 0:
            raise ValueError("fb_consistency_px must be > 0")
        if not 0.0 < self.min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class PairwiseMotion:
    """Per-pair global motion estimate; ``start_frame`` is the first frame index."""

    start_frame: int
    translation_x: float
    translation_y: float
    scale: float
    rotation_rad: float
    magnitude: float
    coherence: float
    residual_rms: float
    dominant_direction_rad: float | None


@dataclass(frozen=True, slots=True)
class MotionAnalyzeResult:
    analysis: dict[str, Any]
    pairwise: tuple[PairwiseMotion, ...]

    def as_motion_analysis(self) -> dict[str, Any]:
        return self.analysis


def analyze_motion(
    path: Path,
    media: ProbedMedia,
    *,
    config: MotionAnalysisConfig | None = None,
    frame_provider: FrameProvider | ReadableFrameProvider | None = None,
    gray_frame_provider: FrameProvider | ReadableFrameProvider | None = None,
    subject_boxes_by_frame: SubjectBoxesByFrame | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> MotionAnalyzeResult:
    """Estimate global motion windows and versioned camera-move heuristics."""
    cfg = config or MotionAnalysisConfig()
    if media.frame_count <= 0:
        return _empty_result(cfg)

    selected_provider = gray_frame_provider or frame_provider
    read_frame, cleanup = _open_frame_reader(path, selected_provider)
    try:
        return estimate_motion(
            media,
            read_frame,
            config=cfg,
            subject_boxes_by_frame=subject_boxes_by_frame,
            cancel_check=cancel_check,
        )
    finally:
        if cleanup is not None:
            cleanup()


def estimate_motion(
    media: ProbedMedia,
    read_frame: FrameProvider,
    *,
    config: MotionAnalysisConfig | None = None,
    subject_boxes_by_frame: SubjectBoxesByFrame | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> MotionAnalyzeResult:
    """Core motion primitive shared with quality shake estimation."""
    cfg = config or MotionAnalysisConfig()
    if media.frame_count <= 0:
        return _empty_result(cfg)

    pairwise, _decoded = _estimate_pairwise_motion(
        media,
        read_frame=read_frame,
        cfg=cfg,
        subject_boxes_by_frame=subject_boxes_by_frame,
        cancel_check=cancel_check,
    )
    windows, window_stats = _aggregate_windows(media.frame_count, pairwise, cfg)
    camera_moves = _classify_camera_moves(windows, window_stats)
    analysis = {
        "windowFrames": cfg.window_frames,
        "samples": windows,
        "cameraMoves": camera_moves,
    }
    return MotionAnalyzeResult(analysis=analysis, pairwise=tuple(pairwise))


def _empty_result(cfg: MotionAnalysisConfig) -> MotionAnalyzeResult:
    analysis = {"windowFrames": cfg.window_frames, "samples": [], "cameraMoves": []}
    return MotionAnalyzeResult(analysis=analysis, pairwise=())


def _open_frame_reader(
    path: Path,
    frame_provider: FrameProvider | ReadableFrameProvider | None,
) -> tuple[FrameProvider, Callable[[], None] | None]:
    if frame_provider is not None:
        if hasattr(frame_provider, "read"):
            provider = frame_provider  # ReadableFrameProvider

            def read(index: int) -> np.ndarray | None:
                return provider.read(index)  # type: ignore[union-attr]

            return read, None
        return frame_provider, None

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise AnalyzeError(DECODE_FAILED, "Could not decode source")

    def read(index: int) -> np.ndarray | None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        return frame

    def cleanup() -> None:
        capture.release()

    return read, cleanup


def _estimate_pairwise_motion(
    media: ProbedMedia,
    *,
    read_frame: FrameProvider,
    cfg: MotionAnalysisConfig,
    subject_boxes_by_frame: SubjectBoxesByFrame | None,
    cancel_check: Callable[[], None] | None,
) -> tuple[list[PairwiseMotion], bool]:
    max_frame = media.frame_count - 1
    if max_frame <= 0:
        return [], False

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    results: list[PairwiseMotion] = []
    decode_attempts = 0
    decode_successes = 0

    for start in range(0, max_frame, cfg.sample_stride):
        if cancel_check:
            cancel_check()
        end = min(start + cfg.sample_stride, max_frame)
        decode_attempts += 2
        prev = read_frame(start)
        nxt = read_frame(end)
        if prev is None or nxt is None:
            continue
        decode_successes += 2
        estimate = _motion_between_frames(
            prev,
            nxt,
            media.width,
            media.height,
            cfg,
            lk_params,
            frame_index=start,
            subject_boxes_by_frame=subject_boxes_by_frame,
        )
        if estimate is None:
            continue
        results.append(
            PairwiseMotion(
                start_frame=start,
                translation_x=estimate["tx"],
                translation_y=estimate["ty"],
                scale=estimate["scale"],
                rotation_rad=estimate["rotation"],
                magnitude=estimate["magnitude"],
                coherence=estimate["coherence"],
                residual_rms=estimate["residual_rms"],
                dominant_direction_rad=estimate["direction"],
            )
        )

    if decode_attempts > 0 and decode_successes == 0:
        raise AnalyzeError(DECODE_FAILED, "Could not decode source")
    return results, decode_successes > 0


def _motion_between_frames(
    prev_bgr: np.ndarray,
    next_bgr: np.ndarray,
    width: int,
    height: int,
    cfg: MotionAnalysisConfig,
    lk_params: dict[str, Any],
    *,
    frame_index: int,
    subject_boxes_by_frame: SubjectBoxesByFrame | None,
) -> dict[str, float | None] | None:
    prev_gray = (
        prev_bgr
        if prev_bgr.ndim == 2
        else cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    )
    next_gray = (
        next_bgr
        if next_bgr.ndim == 2
        else cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    )
    height, width = prev_gray.shape[:2]
    mask = _feature_mask(width, height, frame_index, subject_boxes_by_frame)
    corners = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=CORNER_QUALITY,
        minDistance=CORNER_MIN_DISTANCE_PX,
        mask=mask,
    )
    if corners is None or len(corners) < 8:
        return None

    forward, st_f, _err_f = cv2.calcOpticalFlowPyrLK(
        prev_gray, next_gray, corners, None, **lk_params
    )
    if forward is None or st_f is None:
        return None
    backward, st_b, _err_b = cv2.calcOpticalFlowPyrLK(
        next_gray, prev_gray, forward, None, **lk_params
    )
    if backward is None or st_b is None:
        return None

    consistent = _consistent_tracks(corners, forward, backward, st_f, st_b, cfg.fb_consistency_px)
    if len(consistent) < 6:
        return None

    src = corners[consistent].reshape(-1, 2)
    dst = forward[consistent].reshape(-1, 2)
    affine, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_PX,
    )
    if affine is None or inliers is None:
        return None

    inlier_mask = inliers.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    coherence = inlier_count / len(consistent)
    if coherence < cfg.min_inlier_ratio:
        return None

    tx_px = float(affine[0, 2])
    ty_px = float(affine[1, 2])
    tx = tx_px / max(width, 1)
    ty = ty_px / max(height, 1)
    scale = math.sqrt(float(affine[0, 0] ** 2 + affine[0, 1] ** 2))
    rotation = math.atan2(float(affine[1, 0]), float(affine[0, 0]))
    magnitude = math.sqrt(tx * tx + ty * ty) + abs(scale - 1.0)
    direction: float | None = None
    if magnitude >= STATIC_MAG:
        direction = math.atan2(ty, tx)

    residuals = dst - _apply_affine(src, affine)
    residual_rms = (
        float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1)))) / max(width, height)
    )

    return {
        "tx": tx,
        "ty": ty,
        "scale": scale,
        "rotation": rotation,
        "magnitude": magnitude,
        "coherence": coherence,
        "residual_rms": residual_rms,
        "direction": direction,
    }


def _feature_mask(
    width: int,
    height: int,
    frame_index: int,
    subject_boxes_by_frame: SubjectBoxesByFrame | None,
) -> np.ndarray:
    """Full valid image minus safe borders; optionally exclude supplied subject boxes."""
    margin_x = max(4, int(width * BORDER_MARGIN_FRAC))
    margin_y = max(4, int(height * BORDER_MARGIN_FRAC))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[margin_y : height - margin_y, margin_x : width - margin_x] = 255

    if not subject_boxes_by_frame:
        return mask

    pad_x = max(2, int(width * SUBJECT_BOX_PAD_FRAC))
    pad_y = max(2, int(height * SUBJECT_BOX_PAD_FRAC))
    for box in subject_boxes_by_frame.get(frame_index, []):
        x0 = int(box["x"] * width)
        y0 = int(box["y"] * height)
        x1 = int((box["x"] + box["width"]) * width)
        y1 = int((box["y"] + box["height"]) * height)
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(width, x1 + pad_x)
        y1 = min(height, y1 + pad_y)
        mask[y0:y1, x0:x1] = 0
    return mask


def _consistent_tracks(
    prev: np.ndarray,
    forward: np.ndarray,
    backward: np.ndarray,
    st_f: np.ndarray,
    st_b: np.ndarray,
    fb_threshold_px: float,
) -> np.ndarray:
    good = (st_f.reshape(-1) == 1) & (st_b.reshape(-1) == 1)
    if not np.any(good):
        return np.array([], dtype=np.int64)
    idx = np.flatnonzero(good)
    delta = prev[idx].reshape(-1, 2) - backward[idx].reshape(-1, 2)
    dist = np.linalg.norm(delta, axis=1)
    return idx[dist <= fb_threshold_px]


def _apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homo = np.hstack([points.astype(np.float64), ones])
    transformed = homo @ affine.T
    return transformed


@dataclass(frozen=True, slots=True)
class _WindowStats:
    mean_scale: float
    mean_residual: float


def _aggregate_windows(
    frame_count: int,
    pairwise: Sequence[PairwiseMotion],
    cfg: MotionAnalysisConfig,
) -> tuple[list[dict[str, Any]], list[_WindowStats]]:
    if frame_count <= 0:
        return [], []

    windows: list[dict[str, Any]] = []
    stats: list[_WindowStats] = []
    start = 0
    while start < frame_count:
        end = min(start + cfg.window_frames, frame_count)
        if end <= start:
            break
        members = [p for p in pairwise if start <= p.start_frame < end]
        if members:
            magnitude = float(np.mean([p.magnitude for p in members]))
            coherence = float(np.mean([p.coherence for p in members]))
            mean_scale = float(np.mean([p.scale for p in members]))
            mean_residual = float(np.mean([p.residual_rms for p in members]))
            directions = [
                p.dominant_direction_rad for p in members if p.dominant_direction_rad is not None
            ]
            if directions and magnitude >= STATIC_MAG:
                sin_sum = sum(math.sin(d) for d in directions)
                cos_sum = sum(math.cos(d) for d in directions)
                dominant = math.atan2(sin_sum, cos_sum)
            else:
                dominant = None
        else:
            magnitude = 0.0
            coherence = 0.0
            mean_scale = 1.0
            mean_residual = 0.0
            dominant = None
        windows.append(
            {
                "startFrame": start,
                "endFrameExclusive": end,
                "magnitude": round(magnitude, 6),
                "dominantDirectionRad": None if dominant is None else round(dominant, 6),
                "coherence": round(coherence, 6),
            }
        )
        stats.append(_WindowStats(mean_scale=mean_scale, mean_residual=mean_residual))
        start = end
    return windows, stats


def _classify_camera_moves(
    windows: Sequence[dict[str, Any]],
    window_stats: Sequence[_WindowStats],
) -> list[dict[str, Any]]:
    if not windows:
        return []

    labeled: list[tuple[str, dict[str, Any], float]] = []
    for window, stats in zip(windows, window_stats, strict=True):
        kind, score = _classify_window(window, stats)
        labeled.append((kind, window, score))

    merged: list[dict[str, Any]] = []
    current_kind: str | None = None
    current_start = 0
    current_end = 0
    mag_sum = 0.0
    score_sum = 0.0
    count = 0

    for kind, window, score in labeled:
        start = int(window["startFrame"])
        end = int(window["endFrameExclusive"])
        magnitude = float(window["magnitude"])
        if current_kind is None:
            current_kind = kind
            current_start = start
            current_end = end
            mag_sum = magnitude
            score_sum = score
            count = 1
            continue
        if kind == current_kind and start == current_end:
            current_end = end
            mag_sum += magnitude
            score_sum += score
            count += 1
            continue
        merged.append(
            _camera_move_segment(
                current_kind, current_start, current_end, mag_sum / count, score_sum / count
            )
        )
        current_kind = kind
        current_start = start
        current_end = end
        mag_sum = magnitude
        score_sum = score
        count = 1

    if current_kind is not None:
        merged.append(
            _camera_move_segment(
                current_kind, current_start, current_end, mag_sum / count, score_sum / count
            )
        )
    return merged


def _camera_move_segment(
    kind: str,
    start: int,
    end: int,
    magnitude: float,
    score: float,
) -> dict[str, Any]:
    return {
        "startFrame": start,
        "endFrameExclusive": end,
        "kind": kind,
        "magnitude": round(magnitude, 6),
        "score": round(min(max(score, 0.0), 1.0), 6),
        "scoreType": "heuristic",
    }


def _classify_window(window: dict[str, Any], stats: _WindowStats) -> tuple[str, float]:
    magnitude = float(window["magnitude"])
    coherence = float(window["coherence"])
    direction = window.get("dominantDirectionRad")
    mean_scale = stats.mean_scale
    mean_residual = stats.mean_residual

    if magnitude >= WHIP_MAG and coherence <= WHIP_COHERENCE:
        return "whip", min(1.0, magnitude / WHIP_MAG * 0.7)

    if coherence >= COHERENCE_MOTION and magnitude >= PAN_TILT_MAG:
        if mean_scale >= SCALE_PUSH:
            return "push_in", min(1.0, (mean_scale - 1.0) * 120.0 + 0.35)
        if mean_scale <= SCALE_PULL:
            return "pull_out", min(1.0, (1.0 - mean_scale) * 120.0 + 0.35)

    if magnitude < STATIC_MAG:
        return "static", min(1.0, 1.0 - magnitude / STATIC_MAG)

    if direction is not None and coherence >= COHERENCE_DIRECTION:
        if abs(math.cos(direction)) >= abs(math.sin(direction)):
            if magnitude >= PAN_TILT_MAG:
                return "pan", min(1.0, magnitude / WHIP_MAG + coherence * 0.2)
        elif magnitude >= PAN_TILT_MAG:
            return "tilt", min(1.0, magnitude / WHIP_MAG + coherence * 0.2)

    if (
        magnitude >= HANDHELD_MAG
        and coherence <= HANDHELD_COHERENCE
        and mean_residual >= HANDHELD_RESIDUAL
    ):
        return "handheld", min(1.0, (1.0 - coherence) * 0.8 + mean_residual * 40.0)

    if magnitude >= PAN_TILT_MAG and coherence >= COHERENCE_MOTION:
        if direction is not None and abs(math.cos(direction)) >= abs(math.sin(direction)):
            return "pan", min(1.0, magnitude / WHIP_MAG + 0.15)
        if direction is not None:
            return "tilt", min(1.0, magnitude / WHIP_MAG + 0.15)

    if magnitude >= HANDHELD_MAG:
        return "handheld", min(1.0, magnitude / WHIP_MAG)

    return "static", min(1.0, 0.5)
