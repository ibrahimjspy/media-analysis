"""Stage 1 full-duration person matte CPU reference pipeline."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from media_analysis.errors import INVALID_REQUEST, TIMEOUT, UPLOAD_FAILED, AnalyzeError
from media_analysis.person_matte.constants import MATTE_TEMPORAL_POLICY_VERSION
from media_analysis.person_matte.encoding import encoding_contract_dict, stream_matte_mp4
from media_analysis.person_matte.modnet import OnnxInferenceSession, infer_modnet_alpha
from media_analysis.person_matte.refinement import refine_alpha_rgb_guided
from media_analysis.person_matte.temporal import TemporalMatteState
from media_analysis.person_matte.timing import MatteTimings
from media_analysis.person_matte.types import (
    CanonicalMediaSpec,
    MatteOutputGrant,
    PersonMatteCandidate,
)
from media_analysis.person_matte.validation import (
    coerce_mapping,
    validate_matte_stage1_request,
    validate_shot_timeline,
)


def _check_cancel(cancel_check: Callable[[], None] | None) -> None:
    if cancel_check is not None:
        cancel_check()


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise AnalyzeError(TIMEOUT, "Matte inference exceeded deadline")


def half_matte_resolution(width: int, height: int) -> tuple[int, int]:
    matte_w = max(2, width // 2)
    matte_h = max(2, height // 2)
    matte_w -= matte_w % 2
    matte_h -= matte_h % 2
    return matte_w, matte_h


def _resize_bgr(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame_bgr.shape[1] == width and frame_bgr.shape[0] == height:
        return frame_bgr
    return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)


def propagate_alpha_with_flow(
    previous_bgr: np.ndarray,
    current_bgr: np.ndarray,
    previous_alpha: np.ndarray,
) -> np.ndarray:
    """Map current pixels back to the previous alpha using backward optical flow."""
    previous_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    height, width = previous_gray.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return cv2.remap(
        previous_alpha,
        grid_x + flow[:, :, 0],
        grid_y + flow[:, :, 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


UploadCallback = Callable[[bytes, str, MatteOutputGrant], None]
"""(body, content_type, grant) → None; invoked exactly once for Stage 1."""


def _iter_refined_gray_frames(
    frames: Iterable[tuple[int, np.ndarray]],
    *,
    canonical: CanonicalMediaSpec,
    matte_w: int,
    matte_h: int,
    session: OnnxInferenceSession,
    temporal: TemporalMatteState,
    keyframe_interval: int,
    cancel_check: Callable[[], None] | None,
    deadline: float | None,
    timings: MatteTimings | None = None,
) -> Iterator[bytes]:
    timings = timings or MatteTimings()
    if keyframe_interval < 1:
        raise ValueError("keyframe_interval must be positive")
    expected = 0
    previous_frame: np.ndarray | None = None
    previous_alpha: np.ndarray | None = None
    for source_frame, frame_bgr in frames:
        if source_frame != expected:
            raise AnalyzeError(
                INVALID_REQUEST,
                "sourceFrame indices must be contiguous from 0",
            )
        expected += 1
        _check_cancel(cancel_check)
        _check_deadline(deadline)
        with timings.measure("matte_resize"):
            resized = _resize_bgr(frame_bgr, matte_w, matte_h)
        is_keyframe = (
            source_frame == 0
            or source_frame % keyframe_interval == 0
            or temporal.should_reset(source_frame)
            or previous_frame is None
            or previous_alpha is None
        )
        aligned_previous = None
        if previous_frame is not None and previous_alpha is not None and not (
            temporal.should_reset(source_frame)
        ):
            with timings.measure("matte_flow"):
                aligned_previous = propagate_alpha_with_flow(
                    previous_frame, resized, previous_alpha,
                )
        if is_keyframe:
            with timings.measure("matte_inference"):
                raw_alpha = infer_modnet_alpha(session, resized)
        else:
            assert aligned_previous is not None
            raw_alpha = aligned_previous
        with timings.measure("matte_temporal"):
            stabilized = temporal.apply(
                raw_alpha, source_frame=source_frame, aligned_previous=aligned_previous,
            )
        with timings.measure("matte_refinement"):
            refined = refine_alpha_rgb_guided(stabilized, resized)
        previous_frame = resized
        previous_alpha = refined
        yield refined.tobytes()
    if expected != canonical.frame_count:
        raise AnalyzeError(
            INVALID_REQUEST,
            "full_duration matte frame count must match canonical frameCount",
        )


def run_person_matte_stage1(
    *,
    frames: Iterable[tuple[int, np.ndarray]],
    canonical: CanonicalMediaSpec,
    matte_target: dict[str, Any] | None,
    output_grants: dict[str, Any] | None,
    prior_facts: dict[str, Any] | None,
    session: OnnxInferenceSession,
    upload: UploadCallback,
    work_dir: Path,
    cancel_check: Callable[[], None] | None = None,
    deadline: float | None = None,
    keyframe_interval: int = 3,
    record_stage: Callable[[str, int], None] | None = None,
) -> PersonMatteCandidate:
    """Keyframe MODNet inference with temporal propagation and one final PUT."""
    facts = coerce_mapping(prior_facts, field_name="priorFacts")
    target, grant, shots = validate_matte_stage1_request(
        matte_target=matte_target,
        output_grants=output_grants,
        prior_facts=facts,
        frame_count=canonical.frame_count,
    )
    shots = validate_shot_timeline(shots, frame_count=canonical.frame_count)

    matte_w, matte_h = half_matte_resolution(canonical.width, canonical.height)
    temporal = TemporalMatteState(shots)
    dest = work_dir / "matte.mp4"

    with MatteTimings(record_stage) as timings:
        encoded_count = stream_matte_mp4(
            _iter_refined_gray_frames(
                frames,
                canonical=canonical,
                matte_w=matte_w,
                matte_h=matte_h,
                session=session,
                temporal=temporal,
                keyframe_interval=keyframe_interval,
                cancel_check=cancel_check,
                deadline=deadline,
                timings=timings,
            ),
            width=matte_w,
            height=matte_h,
            fps=canonical.fps,
            dest=dest,
            deadline=deadline,
            cancel_check=cancel_check,
            timings=timings,
        )
    if encoded_count != canonical.frame_count:
        raise AnalyzeError(
            INVALID_REQUEST,
            "full_duration matte frame count must match canonical frameCount",
        )

    body = dest.read_bytes()
    digest = hashlib.sha256(body).hexdigest()

    _check_cancel(cancel_check)
    _check_deadline(deadline)
    try:
        upload(body, "video/mp4", grant)
    except AnalyzeError:
        raise
    except Exception as exc:
        raise AnalyzeError(UPLOAD_FAILED, "Matte upload failed") from exc

    return build_person_matte_candidate(
        matte_target=target,
        grant=grant,
        canonical=canonical,
        matte_width=matte_w,
        matte_height=matte_h,
        sha256=digest,
        byte_count=len(body),
    )


def build_person_matte_candidate(
    *,
    matte_target: dict[str, Any],
    grant: MatteOutputGrant,
    canonical: CanonicalMediaSpec,
    matte_width: int,
    matte_height: int,
    sha256: str,
    byte_count: int,
) -> PersonMatteCandidate:
    return PersonMatteCandidate(
        type="person_matte_v1",
        matte_target=matte_target,
        temporal_policy_version=MATTE_TEMPORAL_POLICY_VERSION,
        reset_policy="shot_boundaries",
        matte_resolution={"width": matte_width, "height": matte_height},
        source_fps=canonical.fps.as_dict(),
        source_time_base=canonical.time_base.as_dict(),
        source_frame_count=canonical.frame_count,
        coverage_ranges=[{"startFrame": 0, "endFrameExclusive": canonical.frame_count}],
        coverage_fraction=1.0,
        encoding=encoding_contract_dict(),
        delivery={
            "mode": "full_duration",
            "grantLabel": grant.label,
            "frameCount": canonical.frame_count,
            "sha256": sha256,
            "byteCount": byte_count,
        },
    )


def synthetic_constant_matte_frames(
    *,
    frame_count: int,
    width: int,
    height: int,
    value: int,
) -> list[tuple[int, np.ndarray]]:
    """Test helper: solid BGR frames (MODNet stub can ignore content)."""
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    return [(index, frame.copy()) for index in range(frame_count)]
