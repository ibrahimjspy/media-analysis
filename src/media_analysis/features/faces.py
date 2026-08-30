"""Face detection, shot-local tracking, and optional subject association."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from media_analysis.decode import ProbedMedia
from media_analysis.features.yunet import (
    YUNET_CAPABILITY_VERSION,
    YUNET_MODEL_FILENAME,
    YuNetDetector,
    clip_normalized_box,
    load_yunet_detector,
    normalize_face_box,
)
from media_analysis.frames import Rational, presentation_time_sec

FACE_TRACKER_VERSION = "iou-shot-reset-1.0.0"
FACE_ASSOCIATION_VERSION = "containment-iou-time-1.0.0"
MAX_SAMPLE_GAP_MS = 200.0
MAX_TRACK_MISSED_SAMPLES = 1

IOU_MATCH_THRESHOLD = 0.3
CONTAINMENT_THRESHOLD = 0.85
SUBJECT_IOU_THRESHOLD = 0.25
MIN_ASSOCIATION_FRAME_RATIO = 0.5
MIN_ASSOCIATION_FRAMES = 2
UNIFORM_SCALE_TOLERANCE = 1e-6


class FrameReader(Protocol):
    def read(self, frame_index: int) -> np.ndarray | None: ...


@dataclass(slots=True)
class VideoFrameReader:
    path: Path
    width: int
    height: int
    _capture: cv2.VideoCapture | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video: {self.path}")
        self._capture = capture

    def read(self, frame_index: int) -> np.ndarray | None:
        if self._capture is None:
            return None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> VideoFrameReader:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(slots=True)
class DictFrameReader:
    """In-memory BGR frames keyed by canonical frame index."""

    frames: dict[int, np.ndarray]

    def read(self, frame_index: int) -> np.ndarray | None:
        return self.frames.get(frame_index)


@dataclass(slots=True)
class _TrackSampleDraft:
    source_frame: int
    box: dict[str, float]
    sample_kind: str
    detector_score: float | None = None
    association_score: float | None = None


@dataclass(slots=True)
class _ActiveTrack:
    track_id: str
    samples: list[_TrackSampleDraft] = field(default_factory=list)
    last_box: dict[str, float] | None = None
    missed_sample_frames: int = 0


def analyze_faces(
    path: Path,
    media: ProbedMedia,
    *,
    shots: list[dict[str, Any]],
    subjects: list[dict[str, Any]] | None = None,
    analysis_width: int | None = None,
    analysis_height: int | None = None,
    model_path: Path | None = None,
    detector: YuNetDetector | None = None,
    frame_reader: FrameReader | None = None,
    max_sample_gap_ms: float = MAX_SAMPLE_GAP_MS,
    max_track_missed_samples: int = MAX_TRACK_MISSED_SAMPLES,
    cancel_check: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Run YuNet + shot-local IoU tracking; return TrackSample-compatible face tracks."""
    measure_w, measure_h = _measure_size(media, analysis_width, analysis_height)
    if detector is None:
        if model_path is None:
            raise ValueError("model_path is required when detector is not injected")
        detector = load_yunet_detector(model_path)

    owns_reader = frame_reader is None
    reader: FrameReader = frame_reader or VideoFrameReader(path, measure_w, measure_h)
    try:
        tracks: list[dict[str, Any]] = []
        for shot_index, shot in enumerate(shots):
            if cancel_check:
                cancel_check()
            start = int(shot["startFrame"])
            end = int(shot["endFrameExclusive"])
            if end <= start:
                continue
            shot_tracks = _analyze_shot(
                reader,
                fps=media.fps,
                shot_index=shot_index,
                start_frame=start,
                end_frame_exclusive=end,
                measure_width=measure_w,
                measure_height=measure_h,
                detector=detector,
                max_sample_gap_ms=max_sample_gap_ms,
                max_track_missed_samples=max_track_missed_samples,
                cancel_check=cancel_check,
            )
            tracks.extend(shot_tracks)

        if subjects:
            _associate_subjects(tracks, subjects)
        return tracks
    finally:
        if owns_reader and isinstance(reader, VideoFrameReader):
            reader.close()


def empty_face_analysis() -> list[dict[str, Any]]:
    return []


def _measure_size(
    media: ProbedMedia,
    analysis_width: int | None,
    analysis_height: int | None,
) -> tuple[int, int]:
    if analysis_width is None and analysis_height is None:
        return media.width, media.height
    if analysis_width is None or analysis_height is None:
        raise ValueError("analysis width and height must both be set or both omitted")
    if analysis_width <= 0 or analysis_height <= 0:
        raise ValueError("analysis resolution must be positive")
    scale_x = analysis_width / media.width
    scale_y = analysis_height / media.height
    if abs(scale_x - scale_y) > UNIFORM_SCALE_TOLERANCE:
        raise ValueError("analysisResolution is not a uniform scale of canonical size")
    return analysis_width, analysis_height


def _analyze_shot(
    reader: FrameReader,
    *,
    fps: Rational,
    shot_index: int,
    start_frame: int,
    end_frame_exclusive: int,
    measure_width: int,
    measure_height: int,
    detector: YuNetDetector,
    max_sample_gap_ms: float,
    max_track_missed_samples: int,
    cancel_check: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    sample_frames = _sample_frames(start_frame, end_frame_exclusive, fps, max_sample_gap_ms)
    max_interp_gap = _max_interp_gap_frames(fps, max_sample_gap_ms)
    active: dict[str, _ActiveTrack] = {}
    finished: list[_ActiveTrack] = []
    next_track = 1
    decode_successes = 0

    for frame_index in sample_frames:
        if cancel_check:
            cancel_check()
        frame = reader.read(frame_index)
        if frame is None:
            continue
        decode_successes += 1
        detections = detector.detect(frame)
        normalized = [
            (
                normalize_face_box(
                    item,
                    frame_width=measure_width,
                    frame_height=measure_height,
                ),
                item.score,
            )
            for item in detections
        ]
        matches, unmatched_tracks, unmatched_detections = _match_detections(active, normalized)

        for track_id, (det_index, iou) in matches.items():
            box, score = normalized[det_index]
            track = active[track_id]
            track.missed_sample_frames = 0
            track.samples.append(
                _TrackSampleDraft(
                    source_frame=frame_index,
                    box=box,
                    sample_kind="detected",
                    detector_score=score,
                    association_score=iou,
                )
            )
            track.last_box = box

        for track_id in unmatched_tracks:
            track = active[track_id]
            track.missed_sample_frames += 1
            if track.missed_sample_frames >= max_track_missed_samples:
                finished.append(track)
                del active[track_id]

        for det_index in unmatched_detections:
            box, score = normalized[det_index]
            track_id = f"face-{shot_index:03d}-{next_track:03d}"
            next_track += 1
            track = _ActiveTrack(track_id=track_id, last_box=box)
            track.samples.append(
                _TrackSampleDraft(
                    source_frame=frame_index,
                    box=box,
                    sample_kind="detected",
                    detector_score=score,
                )
            )
            active[track_id] = track

    if sample_frames and decode_successes == 0:
        raise RuntimeError("Could not decode sampled face frames")

    finished.extend(active.values())

    finalized: list[dict[str, Any]] = []
    for track in finished:
        if not track.samples:
            continue
        samples = _densify_samples(track.samples, fps, max_interp_gap=max_interp_gap)
        if not samples:
            continue
        scores = [
            item["detectorScore"]
            for item in samples
            if item.get("detectorScore") is not None
        ]
        aggregate = sum(scores) / len(scores) if scores else None
        finalized.append(
            {
                "trackId": track.track_id,
                "aggregateScore": aggregate,
                "scoreType": "raw_model",
                "samples": samples,
            }
        )
    return finalized


def _max_interp_gap_frames(fps: Rational, max_sample_gap_ms: float) -> int:
    return max(
        1,
        round(max_sample_gap_ms * fps.numerator / (1000.0 * fps.denominator)),
    )


def _sample_frames(
    start_frame: int,
    end_frame_exclusive: int,
    fps: Rational,
    max_sample_gap_ms: float,
) -> list[int]:
    if end_frame_exclusive <= start_frame:
        return []
    gap_frames = _max_interp_gap_frames(fps, max_sample_gap_ms)
    frames = [start_frame]
    cursor = start_frame + gap_frames
    last_frame = end_frame_exclusive - 1
    while cursor < last_frame:
        frames.append(cursor)
        cursor += gap_frames
    if last_frame > start_frame:
        frames.append(last_frame)
    return sorted(set(frames))


def _match_detections(
    active: dict[str, _ActiveTrack],
    detections: list[tuple[dict[str, float], float]],
) -> tuple[dict[str, tuple[int, float]], list[str], list[int]]:
    candidates: list[tuple[float, str, int]] = []
    for track_id, track in active.items():
        if track.last_box is None:
            continue
        for det_index, (box, _score) in enumerate(detections):
            iou = _box_iou(track.last_box, box)
            if iou >= IOU_MATCH_THRESHOLD:
                candidates.append((iou, track_id, det_index))
    candidates.sort(reverse=True)

    matches: dict[str, tuple[int, float]] = {}
    used_tracks: set[str] = set()
    used_detections: set[int] = set()
    for iou, track_id, det_index in candidates:
        if track_id in used_tracks or det_index in used_detections:
            continue
        matches[track_id] = (det_index, iou)
        used_tracks.add(track_id)
        used_detections.add(det_index)

    unmatched_tracks = [track_id for track_id in active if track_id not in used_tracks]
    unmatched_detections = [
        index for index in range(len(detections)) if index not in used_detections
    ]
    return matches, unmatched_tracks, unmatched_detections


def _densify_samples(
    drafts: list[_TrackSampleDraft],
    fps: Rational,
    *,
    max_interp_gap: int,
) -> list[dict[str, Any]]:
    ordered = sorted(drafts, key=lambda item: item.source_frame)
    dense: list[dict[str, Any]] = []
    for index, draft in enumerate(ordered):
        dense.append(_finalize_sample(draft, fps))
        if index + 1 >= len(ordered):
            continue
        next_draft = ordered[index + 1]
        gap = next_draft.source_frame - draft.source_frame
        if gap <= 1 or gap > max_interp_gap:
            continue
        for step in range(1, gap):
            alpha = step / gap
            interpolated_box = _lerp_box(draft.box, next_draft.box, alpha)
            dense.append(
                _finalize_sample(
                    _TrackSampleDraft(
                        source_frame=draft.source_frame + step,
                        box=interpolated_box,
                        sample_kind="interpolated",
                    ),
                    fps,
                )
            )
    dense.sort(key=lambda item: item["sourceFrame"])
    return dense


def _finalize_sample(draft: _TrackSampleDraft, fps: Rational) -> dict[str, Any]:
    if draft.sample_kind == "detected":
        score_type = "raw_model"
        occluded = False
    else:
        score_type = "heuristic"
        occluded = draft.sample_kind == "tracked"

    sample: dict[str, Any] = {
        "sourceFrame": draft.source_frame,
        "presentationTimeSecApprox": presentation_time_sec(draft.source_frame, fps),
        "box": clip_normalized_box(draft.box),
        "sampleKind": draft.sample_kind,
        "scoreType": score_type,
        "occluded": occluded,
    }
    if draft.detector_score is not None:
        sample["detectorScore"] = draft.detector_score
    if draft.association_score is not None:
        sample["associationScore"] = draft.association_score
    return sample


def _lerp_box(
    left: dict[str, float],
    right: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    return {
        "x": left["x"] + (right["x"] - left["x"]) * alpha,
        "y": left["y"] + (right["y"] - left["y"]) * alpha,
        "width": left["width"] + (right["width"] - left["width"]) * alpha,
        "height": left["height"] + (right["height"] - left["height"]) * alpha,
    }


def _box_area(box: dict[str, float]) -> float:
    return max(0.0, box["width"]) * max(0.0, box["height"])


def _box_intersection(a: dict[str, float], b: dict[str, float]) -> float:
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["width"], b["x"] + b["width"])
    y2 = min(a["y"] + a["height"], b["y"] + b["height"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    intersection = _box_intersection(a, b)
    if intersection <= 0.0:
        return 0.0
    union = _box_area(a) + _box_area(b) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _containment(inner: dict[str, float], outer: dict[str, float]) -> float:
    area = _box_area(inner)
    if area <= 0.0:
        return 0.0
    return _box_intersection(inner, outer) / area


def _subject_sample_at_frame(subject: dict[str, Any], frame: int) -> dict[str, float] | None:
    for sample in subject.get("samples", []):
        if sample.get("sourceFrame") == frame:
            return sample.get("box")
    return None


def _associate_subjects(faces: list[dict[str, Any]], subjects: list[dict[str, Any]]) -> None:
    for face in faces:
        subject_id = _pick_subject_track(face, subjects)
        if subject_id is not None:
            face["subjectTrackId"] = subject_id


def _pick_subject_track(face: dict[str, Any], subjects: list[dict[str, Any]]) -> str | None:
    samples = face.get("samples", [])
    if not samples:
        return None

    per_subject_score: dict[str, float] = {}
    per_subject_frames: dict[str, int] = {}

    for sample in samples:
        if sample.get("sampleKind") != "detected":
            continue
        face_box = sample["box"]
        frame = sample["sourceFrame"]
        for subject in subjects:
            subject_id = subject["trackId"]
            subject_box = _subject_sample_at_frame(subject, frame)
            if subject_box is None:
                continue
            containment = _containment(face_box, subject_box)
            if containment >= CONTAINMENT_THRESHOLD:
                score = containment
            else:
                iou = _box_iou(face_box, subject_box)
                if iou < SUBJECT_IOU_THRESHOLD:
                    continue
                score = iou
            per_subject_score[subject_id] = per_subject_score.get(subject_id, 0.0) + score
            per_subject_frames[subject_id] = per_subject_frames.get(subject_id, 0) + 1

    if not per_subject_score:
        return None

    key_samples = [sample for sample in samples if sample.get("sampleKind") == "detected"]
    required_frames = max(
        MIN_ASSOCIATION_FRAMES,
        int(len(key_samples) * MIN_ASSOCIATION_FRAME_RATIO + 0.999999),
    )
    eligible = [
        subject_id
        for subject_id in per_subject_score
        if per_subject_frames.get(subject_id, 0) >= required_frames
    ]
    if not eligible:
        return None

    best_score = max(per_subject_score[subject_id] for subject_id in eligible)
    winners = [subject_id for subject_id in eligible if per_subject_score[subject_id] == best_score]
    return min(winners)


__all__ = [
    "FACE_ASSOCIATION_VERSION",
    "FACE_TRACKER_VERSION",
    "MAX_SAMPLE_GAP_MS",
    "MAX_TRACK_MISSED_SAMPLES",
    "YUNET_CAPABILITY_VERSION",
    "YUNET_MODEL_FILENAME",
    "DictFrameReader",
    "FrameReader",
    "VideoFrameReader",
    "analyze_faces",
    "empty_face_analysis",
]
