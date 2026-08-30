"""OCR region detection: sampling, PP-OCR det, temporal merge into ReservedRegion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from media_analysis.decode import ProbedMedia
from media_analysis.features.ppocr import (
    PREPROCESSING_VERSION,
    DetPolygon,
    OrtSessionLike,
    create_det_session,
    polygon_to_box,
    run_det_inference,
)
from media_analysis.frames import Rational

OCR_DETECTOR_VERSION = "PP-OCRv5_mobile_det"
OCR_SAMPLER_VERSION = "top20-bottom15-shot-burst-1000ms-1.0.0"
OCR_MERGE_VERSION = "temporal-iou-merge-per-shot-1.0.0"

OCR_ERROR_INFERENCE = "OCR_INFERENCE_FAILED"
OCR_ERROR_DECODE = "OCR_DECODE_FAILED"
OCR_ERROR_SESSION = "OCR_SESSION_FAILED"
OCR_WARNING_PARTIAL_DECODE = "OCR_PARTIAL_DECODE"

TOP_ROI_FRACTION = 0.20
BOTTOM_ROI_START = 0.85
MAX_SAMPLE_GAP_MS = 1000
SHOT_BOUNDARY_BURST = 2
IOU_MATCH_THRESHOLD = 0.30
TRACK_GAP_FRAMES = 15
PERSISTENT_MIN_SPAN_RATIO = 0.40
PERSISTENT_MAX_CENTER_VAR = 0.004

FrameProvider = Callable[[Path, int], np.ndarray | None]
SessionFactory = Callable[[Path], OrtSessionLike]


@dataclass(frozen=True, slots=True)
class FrameDetection:
    source_frame: int
    polygon: tuple[tuple[float, float], ...]
    box: dict[str, float]
    detector_score: float
    shot_index: int = 0


@dataclass
class _ActiveTrack:
    region_id: str
    start_frame: int
    end_frame_exclusive: int
    shot_index: int
    samples: list[dict[str, Any]] = field(default_factory=list)
    last_box: dict[str, float] | None = None
    sampled_frames: set[int] = field(default_factory=set)

    def to_region(self, *, frame_count: int) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "startFrame": self.start_frame,
            "endFrameExclusive": self.end_frame_exclusive,
            "kind": classify_region_kind(self.samples, frame_count),
            "samples": list(self.samples),
        }


@dataclass(frozen=True, slots=True)
class OcrAnalyzeResult:
    regions: list[dict[str, Any]]
    status: Literal["completed", "failed"]
    version: str = OCR_DETECTOR_VERSION
    error: str | None = None
    warning_codes: tuple[str, ...] = ()


class OcrInferenceError(Exception):
    """Internal inference failure; mapped to OCR_ERROR_INFERENCE."""


class VideoFrameReader:
    """Single VideoCapture session reused across sampled frames."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self._capture = cv2.VideoCapture(str(self._path))
        return self._capture.isOpened()

    def read(self, frame_index: int) -> np.ndarray | None:
        if self._capture is None or not self._capture.isOpened():
            return None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def detection_in_roi(box: dict[str, float]) -> bool:
    """Keep detections whose vertical center lies in top ~20% or bottom ~15%."""
    center_y = box["y"] + box["height"] / 2.0
    return center_y <= TOP_ROI_FRACTION or center_y >= BOTTOM_ROI_START


def sample_ocr_frames(
    frame_count: int,
    fps: Rational,
    shots: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Deterministic temporal sampler: 1000ms max-gap cadence plus shot-boundary bursts."""
    if frame_count <= 0:
        return []

    gap_frames = max(
        1,
        int(round(MAX_SAMPLE_GAP_MS * fps.numerator / (fps.denominator * 1000))),
    )
    frames: set[int] = {0}
    if frame_count > 1:
        frames.add(frame_count - 1)

    cursor = 0
    while cursor < frame_count:
        frames.add(cursor)
        cursor += gap_frames

    if shots:
        for shot in shots:
            for boundary in (shot["startFrame"], shot["endFrameExclusive"] - 1):
                for offset in range(-SHOT_BOUNDARY_BURST, SHOT_BOUNDARY_BURST + 1):
                    idx = boundary + offset
                    if 0 <= idx < frame_count:
                        frames.add(idx)

    return sorted(frames)


def _shot_ranges(
    shots: list[dict[str, Any]] | None,
    frame_count: int,
) -> list[tuple[int, int]]:
    if not shots:
        return [(0, frame_count)]
    return [
        (shot["startFrame"], shot["endFrameExclusive"])
        for shot in shots
        if shot["endFrameExclusive"] > shot["startFrame"]
    ]


def _shot_index_for_frame(
    frame: int,
    ranges: list[tuple[int, int]],
) -> int | None:
    for index, (start, end) in enumerate(ranges):
        if start <= frame < end:
            return index
    return None


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax2 = a["x"] + a["width"]
    ay2 = a["y"] + a["height"]
    bx2 = b["x"] + b["width"]
    by2 = b["y"] + b["height"]
    ix0 = max(a["x"], b["x"])
    iy0 = max(a["y"], b["y"])
    ix1 = min(ax2, bx2)
    iy1 = min(ay2, by2)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    if union <= 0:
        return 0.0
    return inter / union


def polygon_to_sample(det: FrameDetection) -> dict[str, Any]:
    return {
        "sourceFrame": det.source_frame,
        "polygon": [{"x": x, "y": y} for x, y in det.polygon],
        "box": det.box,
        "detectorScore": det.detector_score,
        "scoreType": "raw_model",
    }


def classify_region_kind(samples: list[dict[str, Any]], frame_count: int) -> str:
    """Location/persistence heuristic; not a watermark claim."""
    if not samples or frame_count <= 0:
        return "ocr_text"
    start = min(s["sourceFrame"] for s in samples)
    end = max(s["sourceFrame"] for s in samples) + 1
    span_ratio = (end - start) / frame_count
    if span_ratio < PERSISTENT_MIN_SPAN_RATIO:
        return "ocr_text"
    centers = np.array(
        [
            (
                s["box"]["x"] + s["box"]["width"] / 2.0,
                s["box"]["y"] + s["box"]["height"] / 2.0,
            )
            for s in samples
        ],
        dtype=np.float64,
    )
    if float(centers.var(axis=0).sum()) <= PERSISTENT_MAX_CENTER_VAR:
        return "persistent_overlay"
    return "ocr_text"


def _merge_detections_within_shot(
    detections: list[FrameDetection],
    *,
    frame_count: int,
    shot_index: int,
    id_counter: list[int],
) -> list[dict[str, Any]]:
    if not detections:
        return []

    ordered = sorted(detections, key=lambda item: item.source_frame)
    active: list[_ActiveTrack] = []
    finished: list[_ActiveTrack] = []

    by_frame: dict[int, list[FrameDetection]] = {}
    for det in ordered:
        by_frame.setdefault(det.source_frame, []).append(det)

    for frame in sorted(by_frame):
        still_active: list[_ActiveTrack] = []
        for track in active:
            if frame - (track.end_frame_exclusive - 1) > TRACK_GAP_FRAMES:
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active

        matched_tracks: set[int] = set()
        for det in by_frame[frame]:
            best_track: _ActiveTrack | None = None
            best_iou = 0.0
            for track in active:
                track_key = id(track)
                if track_key in matched_tracks:
                    continue
                if frame in track.sampled_frames:
                    continue
                if track.last_box is None:
                    continue
                iou = box_iou(track.last_box, det.box)
                if iou >= IOU_MATCH_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    best_track = track

            sample = polygon_to_sample(det)
            if best_track is not None:
                best_track.samples.append(sample)
                best_track.end_frame_exclusive = frame + 1
                best_track.last_box = det.box
                best_track.sampled_frames.add(frame)
                matched_tracks.add(id(best_track))
            else:
                region_id = f"ocr-{id_counter[0]}"
                id_counter[0] += 1
                track = _ActiveTrack(
                    region_id=region_id,
                    start_frame=frame,
                    end_frame_exclusive=frame + 1,
                    shot_index=shot_index,
                    samples=[sample],
                    last_box=det.box,
                    sampled_frames={frame},
                )
                active.append(track)

    finished.extend(active)
    return [track.to_region(frame_count=frame_count) for track in finished if track.samples]


def merge_temporal_detections(
    detections: list[FrameDetection],
    *,
    frame_count: int,
    shots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge per-frame detections into ReservedRegion tracks; never crosses shot cuts."""
    if not detections:
        return []

    ranges = _shot_ranges(shots, frame_count)
    by_shot: dict[int, list[FrameDetection]] = {}
    for det in detections:
        shot_index = det.shot_index
        if shot_index is None:
            shot_index = _shot_index_for_frame(det.source_frame, ranges)
        if shot_index is None:
            continue
        by_shot.setdefault(shot_index, []).append(det)

    id_counter = [0]
    regions: list[dict[str, Any]] = []
    for shot_index in sorted(by_shot):
        regions.extend(
            _merge_detections_within_shot(
                by_shot[shot_index],
                frame_count=frame_count,
                shot_index=shot_index,
                id_counter=id_counter,
            )
        )
    return regions


def assign_shot_indices(
    detections: list[FrameDetection],
    shots: list[dict[str, Any]] | None,
    frame_count: int,
) -> list[FrameDetection]:
    ranges = _shot_ranges(shots, frame_count)
    assigned: list[FrameDetection] = []
    for det in detections:
        shot_index = _shot_index_for_frame(det.source_frame, ranges)
        if shot_index is None:
            continue
        assigned.append(
            FrameDetection(
                source_frame=det.source_frame,
                polygon=det.polygon,
                box=det.box,
                detector_score=det.detector_score,
                shot_index=shot_index,
            )
        )
    return assigned


def polygons_to_detections(
    polygons: list[DetPolygon],
    *,
    source_frame: int,
    shot_index: int = 0,
) -> list[FrameDetection]:
    detections: list[FrameDetection] = []
    for poly in polygons:
        box = polygon_to_box(poly.points)
        if box["width"] <= 0 or box["height"] <= 0:
            continue
        if not detection_in_roi(box):
            continue
        detections.append(
            FrameDetection(
                source_frame=source_frame,
                polygon=poly.points,
                box=box,
                detector_score=poly.score,
                shot_index=shot_index,
            )
        )
    return detections


def analyze_ocr(
    path: Path,
    media: ProbedMedia,
    *,
    shots: list[dict[str, Any]] | None = None,
    model_path: Path,
    session: OrtSessionLike | None = None,
    session_factory: SessionFactory | None = None,
    frame_provider: FrameProvider | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> OcrAnalyzeResult:
    """Run detection-only OCR on sampled frames and merge into reserved regions."""
    sample_frames = sample_ocr_frames(media.frame_count, media.fps, shots)
    if not sample_frames:
        return OcrAnalyzeResult(regions=[], status="completed")

    owned_session = session
    created_session = False
    if owned_session is None:
        factory = session_factory or create_det_session
        try:
            owned_session = factory(model_path)
            created_session = True
        except Exception:
            return OcrAnalyzeResult(
                regions=[],
                status="failed",
                error=OCR_ERROR_SESSION,
            )

    reader: VideoFrameReader | None = None
    read_frame: FrameProvider
    if frame_provider is None:
        reader = VideoFrameReader(path)
        if not reader.open():
            return OcrAnalyzeResult(
                regions=[],
                status="failed",
                error=OCR_ERROR_DECODE,
            )
        read_frame = reader.read
    else:
        read_frame = frame_provider

    shot_ranges = _shot_ranges(shots, media.frame_count)
    all_detections: list[FrameDetection] = []
    decode_attempts = 0
    decode_successes = 0
    warning_codes: list[str] = []

    try:
        for frame_index in sample_frames:
            if cancel_check:
                cancel_check()
            decode_attempts += 1
            if frame_provider is not None:
                image = read_frame(path, frame_index)
            else:
                image = read_frame(frame_index)
            if image is None:
                continue
            decode_successes += 1
            shot_index = _shot_index_for_frame(frame_index, shot_ranges)
            if shot_index is None:
                continue
            try:
                polygons = run_det_inference(owned_session, image)
            except Exception as exc:
                raise OcrInferenceError from exc
            all_detections.extend(
                polygons_to_detections(
                    polygons,
                    source_frame=frame_index,
                    shot_index=shot_index,
                )
            )
    except OcrInferenceError:
        return OcrAnalyzeResult(
            regions=[],
            status="failed",
            error=OCR_ERROR_INFERENCE,
        )
    finally:
        if reader is not None:
            reader.close()
        if created_session and hasattr(owned_session, "close"):
            owned_session.close()

    if decode_attempts > 0 and decode_successes == 0:
        return OcrAnalyzeResult(
            regions=[],
            status="failed",
            error=OCR_ERROR_DECODE,
        )

    if decode_successes < decode_attempts:
        warning_codes.append(OCR_WARNING_PARTIAL_DECODE)

    regions = merge_temporal_detections(
        all_detections,
        frame_count=media.frame_count,
        shots=shots,
    )
    return OcrAnalyzeResult(
        regions=regions,
        status="completed",
        warning_codes=tuple(warning_codes),
    )


def preprocessing_version() -> str:
    return PREPROCESSING_VERSION
