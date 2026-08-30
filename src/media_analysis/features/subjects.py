"""Subject (person) detection + ByteTrack association with shot-boundary resets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from media_analysis.decode import ProbedMedia
from media_analysis.features.bytetrack import ByteTracker, STrack, TrackSampleRecord
from media_analysis.features.visual import model_provenance
from media_analysis.features.yolox import (
    YOLOX_PREPROCESSING,
    YoloxDetector,
    create_yolox_session,
    map_boxes_to_normalized,
)
from media_analysis.frames import presentation_time_sec
from media_analysis.models_manifest import ModelEntry

TRACKER_VERSION = "bytetrack-shot-reset-1.0.0"
SAMPLING_POLICY_VERSION = "max-gap-200ms-1.0.0"
MAX_GAP_MS = 200.0


@dataclass(frozen=True, slots=True)
class SubjectAnalysisConfig:
    max_gap_ms: float = MAX_GAP_MS
    tracker_version: str = TRACKER_VERSION
    sampling_policy_version: str = SAMPLING_POLICY_VERSION


def subject_model_provenance(entry: ModelEntry | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    runtime = "stub" if entry.stub else "onnxruntime-cpu"
    return model_provenance(entry, runtime=runtime, preprocessing=YOLOX_PREPROCESSING)


def build_yolox_detector(model_path: Path) -> YoloxDetector:
    session = create_yolox_session(model_path)
    return YoloxDetector(session=session)


def analyze_subjects(
    path: Path,
    media: ProbedMedia,
    *,
    shots: list[dict[str, Any]] | None = None,
    detector: YoloxDetector | None = None,
    model_path: Path | None = None,
    config: SubjectAnalysisConfig | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Run person detection + tracking; returns TRD `SubjectTrack` dicts."""
    cfg = config or SubjectAnalysisConfig()
    if detector is None:
        if model_path is None:
            raise ValueError("detector or model_path is required")
        detector = build_yolox_detector(model_path)

    shot_ranges = _shot_ranges(shots, media.frame_count)
    max_gap_frames = max(1, _ms_to_frames(cfg.max_gap_ms, media.fps))
    all_tracks: list[dict[str, Any]] = []
    decode_attempts = 0
    decode_successes = 0

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Could not open video for subject analysis")

    try:
        for shot_index, (start_frame, end_frame) in enumerate(shot_ranges):
            if cancel_check:
                cancel_check()
            tracker = ByteTracker(frame_width=media.width, frame_height=media.height)
            shot_tracks: list[STrack] = []
            sample_frames = _sample_frames(
                start_frame,
                end_frame,
                max_gap_frames=max_gap_frames,
            )
            mandatory_frames = set(sample_frames) | {start_frame, max(start_frame, end_frame - 1)}

            for frame_index in sample_frames:
                if cancel_check:
                    cancel_check()
                decode_attempts += 1
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    continue
                decode_successes += 1
                detections, meta = detector.detect(frame_bgr)
                boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
                norm_boxes = map_boxes_to_normalized(boxes, meta)
                active = tracker.update(
                    detections,
                    source_frame=frame_index,
                    norm_boxes=norm_boxes,
                )
                shot_tracks.extend(active)

            unique: dict[int, STrack] = {}
            for track in shot_tracks:
                unique[track.track_id] = track
            merged = list(unique.values())
            for track in merged:
                tracker.fill_gaps(
                    [track],
                    mandatory_frames=mandatory_frames,
                    max_gap_frames=max_gap_frames,
                )
            for track in merged:
                if not track.samples:
                    continue
                all_tracks.append(
                    _serialize_track(
                        track,
                        shot_index=shot_index,
                        fps=media.fps,
                    )
                )
    finally:
        capture.release()

    if decode_attempts and decode_successes == 0:
        raise RuntimeError("Could not decode sampled subject frames")
    return all_tracks


def _serialize_track(track: STrack, *, shot_index: int, fps: Any) -> dict[str, Any]:
    scores = [
        sample.detector_score for sample in track.samples if sample.detector_score is not None
    ]
    aggregate = float(sum(scores) / len(scores)) if scores else None
    ordered = sorted(track.samples, key=lambda sample: sample.source_frame)
    return {
        "trackId": f"{shot_index}-{track.track_id}",
        "type": "person",
        "aggregateScore": aggregate,
        "scoreType": "raw_model" if scores else "heuristic",
        "samples": [_serialize_sample(sample, fps=fps) for sample in ordered],
    }


def _serialize_sample(sample: TrackSampleRecord, *, fps: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sourceFrame": sample.source_frame,
        "presentationTimeSecApprox": presentation_time_sec(sample.source_frame, fps),
        "box": sample.box,
        "sampleKind": sample.sample_kind,
        "scoreType": sample.score_type,
        "occluded": sample.occluded,
    }
    if sample.detector_score is not None:
        payload["detectorScore"] = sample.detector_score
    if sample.association_score is not None:
        payload["associationScore"] = sample.association_score
    return payload


def _shot_ranges(shots: list[dict[str, Any]] | None, frame_count: int) -> list[tuple[int, int]]:
    if not shots:
        return [(0, frame_count)]
    ranges: list[tuple[int, int]] = []
    for shot in shots:
        start = int(shot["startFrame"])
        end = int(shot["endFrameExclusive"])
        if end <= start:
            continue
        ranges.append((start, min(end, frame_count)))
    return ranges or [(0, frame_count)]


def _ms_to_frames(ms: float, fps: Any) -> int:
    if fps.numerator == 0:
        return 1
    sec = ms / 1000.0
    return max(1, int(round(sec * fps.numerator / fps.denominator)))


def _sample_frames(start_frame: int, end_frame_exclusive: int, *, max_gap_frames: int) -> list[int]:
    if end_frame_exclusive <= start_frame:
        return []
    frames = list(range(start_frame, end_frame_exclusive, max_gap_frames))
    last = end_frame_exclusive - 1
    if last not in frames:
        frames.append(last)
    return frames


def empty_subjects() -> list[dict[str, Any]]:
    return []


def count_distinct_persons(subjects: list[dict[str, Any]]) -> int:
    return len(subjects)
