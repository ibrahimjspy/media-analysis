"""Per-shot quality measurements and versioned policy flags.

Provisional (benchmark-gated) thresholds are encoded in ``QUALITY_POLICY_VERSION``
and ``QualityAnalysisConfig``. Returns measurements and flags only — never edit
decisions such as keep/trim/reject.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from media_analysis.decode import ProbedMedia
from media_analysis.errors import CANCELLED, DECODE_FAILED, INVALID_REQUEST, AnalyzeError
from media_analysis.features.measure_common import FrameProvider, ReadableFrameProvider
from media_analysis.features.motion import (
    MotionAnalysisConfig,
    MotionAnalyzeResult,
    PairwiseMotion,
    SubjectBoxesByFrame,
    estimate_motion,
)

# Aspect-preserving working size for sharpness measurement.
SHARPNESS_WORKING_WIDTH = 540
SHARPNESS_WORKING_HEIGHT = 960
# Laplacian variance threshold measured at the working size.
SOFT_FOCUS_THRESHOLD = 120.0
# Residual RMS from global-motion fit (normalized).
SEVERE_SHAKE_THRESHOLD = 0.0065
# BT.709 luma clipping fractions.
BLOWN_HIGHLIGHT_FRACTION = 0.018
CRUSHED_SHADOW_FRACTION = 0.022
BLOWN_LUMA = 0.985
CRUSHED_LUMA = 0.015
# Duplicate detection: structure hash + appearance distance.
DUPLICATE_HASH_SIZE = 8
DUPLICATE_MAX_HASH_DISTANCE = 6
DUPLICATE_MAX_LUMA_DISTANCE = 0.08
DUPLICATE_MAX_COLOR_DISTANCE = 0.12


def build_quality_policy_version() -> str:
    return (
        "quality-provisional-v1.1.0"
        f"-ws{SHARPNESS_WORKING_WIDTH}x{SHARPNESS_WORKING_HEIGHT}"
        f"-sf{SOFT_FOCUS_THRESHOLD}-sh{SEVERE_SHAKE_THRESHOLD}"
        f"-bh{BLOWN_HIGHLIGHT_FRACTION}-cs{CRUSHED_SHADOW_FRACTION}"
        f"-duph{DUPLICATE_MAX_HASH_DISTANCE}-dupl{DUPLICATE_MAX_LUMA_DISTANCE}"
        f"-dupc{DUPLICATE_MAX_COLOR_DISTANCE}"
    )


QUALITY_POLICY_VERSION = build_quality_policy_version()

MotionEstimator = Callable[[], MotionAnalyzeResult]


@dataclass(frozen=True, slots=True)
class QualityAnalysisConfig:
    sharpness_working_width: int = SHARPNESS_WORKING_WIDTH
    sharpness_working_height: int = SHARPNESS_WORKING_HEIGHT
    soft_focus_threshold: float = SOFT_FOCUS_THRESHOLD
    severe_shake_threshold: float = SEVERE_SHAKE_THRESHOLD
    blown_highlight_fraction: float = BLOWN_HIGHLIGHT_FRACTION
    crushed_shadow_fraction: float = CRUSHED_SHADOW_FRACTION
    blown_luma: float = BLOWN_LUMA
    crushed_luma: float = CRUSHED_LUMA
    duplicate_max_hash_distance: int = DUPLICATE_MAX_HASH_DISTANCE
    duplicate_max_luma_distance: float = DUPLICATE_MAX_LUMA_DISTANCE
    duplicate_max_color_distance: float = DUPLICATE_MAX_COLOR_DISTANCE
    policy_version: str = QUALITY_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.sharpness_working_width < 1 or self.sharpness_working_height < 1:
            raise ValueError("sharpness working size must be positive")
        if self.soft_focus_threshold <= 0:
            raise ValueError("soft_focus_threshold must be > 0")
        if self.severe_shake_threshold <= 0:
            raise ValueError("severe_shake_threshold must be > 0")
        if not 0.0 <= self.blown_highlight_fraction <= 1.0:
            raise ValueError("blown_highlight_fraction must be in [0, 1]")
        if not 0.0 <= self.crushed_shadow_fraction <= 1.0:
            raise ValueError("crushed_shadow_fraction must be in [0, 1]")
        if self.duplicate_max_hash_distance < 0:
            raise ValueError("duplicate_max_hash_distance must be >= 0")
        if self.duplicate_max_luma_distance <= 0:
            raise ValueError("duplicate_max_luma_distance must be > 0")
        if self.duplicate_max_color_distance <= 0:
            raise ValueError("duplicate_max_color_distance must be > 0")


@dataclass(frozen=True, slots=True)
class ShotSignature:
    ahash: np.ndarray
    mean_luma: float
    mean_b: float
    mean_g: float
    mean_r: float


@dataclass(frozen=True, slots=True)
class QualityAnalyzeResult:
    analysis: dict[str, Any]

    def as_quality_analysis(self) -> dict[str, Any]:
        return self.analysis


def analyze_quality(
    path: Path,
    media: ProbedMedia,
    *,
    shots: list[dict[str, Any]] | None = None,
    motion: MotionAnalyzeResult | None = None,
    compute_motion: MotionEstimator | None = None,
    subject_boxes_by_frame: SubjectBoxesByFrame | None = None,
    config: QualityAnalysisConfig | None = None,
    frame_provider: FrameProvider | ReadableFrameProvider | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> QualityAnalyzeResult:
    """Measure per-shot quality using shot-local deterministic representatives."""
    cfg = config or QualityAnalysisConfig()
    validated_shots = _validate_shots(shots, media.frame_count)
    if not validated_shots:
        return QualityAnalyzeResult(analysis={"perShot": []})

    read_frame, cleanup = _open_frame_reader(path, frame_provider)
    motion_result = motion
    try:
        representatives = _load_representatives(
            validated_shots,
            read_frame=read_frame,
            cancel_check=cancel_check,
        )
        if motion_result is None and compute_motion is not None:
            motion_result = compute_motion()
        elif motion_result is None:
            motion_result = estimate_motion(
                media,
                read_frame,
                subject_boxes_by_frame=subject_boxes_by_frame,
                cancel_check=cancel_check,
            )

        signatures: list[tuple[int, ShotSignature | None]] = []
        per_shot: list[dict[str, Any]] = []

        for shot_index, start, end, frame in representatives:
            if cancel_check:
                cancel_check()
            sharpness = _sharpness_at_working_size(
                frame,
                media.width,
                media.height,
                cfg.sharpness_working_width,
                cfg.sharpness_working_height,
            )
            shake = _shake_for_shot(start, end, motion_result)
            blown_fraction, crushed_fraction = _luma_clip_fractions(frame, cfg)
            signature = _shot_signature(frame)
            signatures.append((shot_index, signature))

            duplicate_of: int | None = None
            duplicate_score: float | None = None
            duplicate_of, duplicate_score = _find_duplicate(
                shot_index,
                signature,
                signatures[:-1],
                cfg,
            )

            flags = _derive_flags(
                sharpness=sharpness,
                shake=shake,
                blown_fraction=blown_fraction,
                crushed_fraction=crushed_fraction,
                duplicate_of=duplicate_of,
                cfg=cfg,
            )
            entry: dict[str, Any] = {
                "shotIndex": shot_index,
                "sharpness": round(sharpness, 6),
                "shake": round(shake, 6),
                "blownHighlightFraction": round(blown_fraction, 6),
                "crushedShadowFraction": round(crushed_fraction, 6),
                "duplicateOfShotIndex": duplicate_of,
                "flags": flags,
                "policyVersion": cfg.policy_version,
            }
            if duplicate_score is not None:
                entry["duplicateScore"] = round(duplicate_score, 6)
            per_shot.append(entry)

        if len(per_shot) != len(validated_shots):
            raise AnalyzeError(DECODE_FAILED, "Could not decode source")
        return QualityAnalyzeResult(analysis={"perShot": per_shot})
    finally:
        if cleanup is not None:
            cleanup()


def _validate_shots(
    shots: list[dict[str, Any]] | None,
    frame_count: int,
) -> list[dict[str, Any]]:
    if shots is None:
        if frame_count <= 0:
            return []
        return [{"startFrame": 0, "endFrameExclusive": frame_count}]
    if not shots:
        return []
    validated: list[dict[str, Any]] = []
    for shot in shots:
        start = int(shot["startFrame"])
        end = int(shot["endFrameExclusive"])
        if start < 0 or end > frame_count or end <= start:
            raise AnalyzeError(
                INVALID_REQUEST,
                "shot ranges must be half-open and lie inside canonical frameCount",
            )
        validated.append({"startFrame": start, "endFrameExclusive": end})
    return validated


def _representative_frame(start: int, end_exclusive: int) -> int:
    span = end_exclusive - start
    if span <= 0:
        return start
    return start + (span // 2)


def _open_frame_reader(
    path: Path,
    frame_provider: FrameProvider | ReadableFrameProvider | None,
) -> tuple[FrameProvider, Callable[[], None] | None]:
    if frame_provider is not None:
        if hasattr(frame_provider, "read"):
            provider = frame_provider

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


def _load_representatives(
    shots: list[dict[str, Any]],
    *,
    read_frame: FrameProvider,
    cancel_check: Callable[[], None] | None,
) -> list[tuple[int, int, int, np.ndarray]]:
    representatives: list[tuple[int, int, int, np.ndarray]] = []
    decode_attempts = 0
    decode_successes = 0

    for shot_index, shot in enumerate(shots):
        if cancel_check:
            cancel_check()
        start = int(shot["startFrame"])
        end = int(shot["endFrameExclusive"])
        rep_frame = _representative_frame(start, end)
        decode_attempts += 1
        frame = read_frame(rep_frame)
        if frame is None:
            continue
        decode_successes += 1
        representatives.append((shot_index, start, end, frame))

    if decode_attempts > 0 and decode_successes == 0:
        raise AnalyzeError(DECODE_FAILED, "Could not decode source")
    if decode_successes != len(shots):
        raise AnalyzeError(DECODE_FAILED, "Could not decode source")
    return representatives


def _sharpness_at_working_size(
    frame_bgr: np.ndarray,
    src_width: int,
    src_height: int,
    working_width: int,
    working_height: int,
) -> float:
    scale = min(working_width / max(src_width, 1), working_height / max(src_height, 1))
    target_w = max(1, int(round(src_width * scale)))
    target_h = max(1, int(round(src_height * scale)))
    resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _shake_for_shot(
    start: int,
    end: int,
    motion: MotionAnalyzeResult | None,
) -> float:
    if motion is None or not motion.pairwise:
        return 0.0
    residuals = [p.residual_rms for p in motion.pairwise if start <= p.start_frame < end]
    if not residuals:
        return 0.0
    return float(np.mean(residuals))


def _luma_clip_fractions(frame_bgr: np.ndarray, cfg: QualityAnalysisConfig) -> tuple[float, float]:
    rgb = frame_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(rgb)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    total = y.size
    if total == 0:
        return 0.0, 0.0
    blown = float(np.count_nonzero(y >= cfg.blown_luma)) / total
    crushed = float(np.count_nonzero(y <= cfg.crushed_luma)) / total
    return blown, crushed


def _shot_signature(frame_bgr: np.ndarray) -> ShotSignature:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(
        gray, (DUPLICATE_HASH_SIZE, DUPLICATE_HASH_SIZE), interpolation=cv2.INTER_AREA
    )
    mean_luma = float(resized.mean()) / 255.0
    mean_b, mean_g, mean_r = (frame_bgr.astype(np.float32).mean(axis=(0, 1)) / 255.0).tolist()
    ahash = (resized >= float(resized.mean())).astype(np.uint8)
    return ShotSignature(
        ahash=ahash,
        mean_luma=mean_luma,
        mean_b=float(mean_b),
        mean_g=float(mean_g),
        mean_r=float(mean_r),
    )


def _hamming_distance(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def _appearance_distance(left: ShotSignature, right: ShotSignature) -> tuple[float, float]:
    luma_distance = abs(left.mean_luma - right.mean_luma)
    color_distance = math_dist(
        (left.mean_b, left.mean_g, left.mean_r),
        (right.mean_b, right.mean_g, right.mean_r),
    )
    return luma_distance, color_distance


def math_dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)))


def _find_duplicate(
    shot_index: int,
    signature: ShotSignature,
    prior: Sequence[tuple[int, ShotSignature | None]],
    cfg: QualityAnalysisConfig,
) -> tuple[int | None, float | None]:
    best_index: int | None = None
    best_score: float | None = None
    max_bits = DUPLICATE_HASH_SIZE * DUPLICATE_HASH_SIZE

    for earlier_index, earlier in prior:
        if earlier is None or earlier_index >= shot_index:
            continue
        hash_distance = _hamming_distance(signature.ahash, earlier.ahash)
        luma_distance, color_distance = _appearance_distance(signature, earlier)
        if hash_distance > cfg.duplicate_max_hash_distance:
            continue
        if luma_distance > cfg.duplicate_max_luma_distance:
            continue
        if color_distance > cfg.duplicate_max_color_distance:
            continue
        hash_score = 1.0 - (hash_distance / max_bits)
        appearance_score = 1.0 - min(
            1.0,
            (luma_distance / cfg.duplicate_max_luma_distance)
            + (color_distance / cfg.duplicate_max_color_distance) / 2.0,
        )
        score = (hash_score + appearance_score) / 2.0
        if best_score is None or score > best_score:
            best_score = score
            best_index = earlier_index

    return best_index, best_score


def _derive_flags(
    *,
    sharpness: float,
    shake: float,
    blown_fraction: float,
    crushed_fraction: float,
    duplicate_of: int | None,
    cfg: QualityAnalysisConfig,
) -> list[str]:
    flags: list[str] = []
    if sharpness < cfg.soft_focus_threshold:
        flags.append("soft_focus")
    if shake >= cfg.severe_shake_threshold:
        flags.append("severe_shake")
    if blown_fraction >= cfg.blown_highlight_fraction:
        flags.append("blown_highlights")
    if crushed_fraction >= cfg.crushed_shadow_fraction:
        flags.append("crushed_shadows")
    if duplicate_of is not None:
        flags.append("near_duplicate")
    return flags


def shake_from_motion_pairwise(
    pairwise: Sequence[PairwiseMotion],
    start: int,
    end: int,
) -> float | None:
    """Expose motion-derived shake for tests and parent reuse."""
    residuals = [p.residual_rms for p in pairwise if start <= p.start_frame < end]
    if not residuals:
        return None
    return float(np.mean(residuals))


def make_internal_motion_estimator(
    media: ProbedMedia,
    read_frame: FrameProvider,
    *,
    subject_boxes_by_frame: SubjectBoxesByFrame | None = None,
    motion_config: MotionAnalysisConfig | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> MotionEstimator:
    """Build a lazy motion estimator for quality-only requests sharing one frame reader."""

    cached: dict[str, MotionAnalyzeResult | None] = {"value": None}

    def compute() -> MotionAnalyzeResult:
        if cached["value"] is None:
            cached["value"] = estimate_motion(
                media,
                read_frame,
                config=motion_config,
                subject_boxes_by_frame=subject_boxes_by_frame,
                cancel_check=cancel_check,
            )
        return cached["value"]

    return compute


def raise_if_cancelled(flag: bool) -> None:
    if flag:
        raise AnalyzeError(CANCELLED, "Analysis cancelled")
