"""ByteTrack-style multi-object association (person boxes, no ReID)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from media_analysis.features.yolox import DEFAULT_LOW_DET_THRESH, DEFAULT_TRACK_THRESH, Detection

ASSOCIATION_MEASURE = "iou-xyxy-1.0.0"


class TrackState(StrEnum):
    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass(slots=True)
class STrack:
    track_id: int
    tlbr: np.ndarray
    score: float
    state: TrackState = TrackState.NEW
    frame_id: int = 0
    start_frame: int = 0
    end_frame: int = 0
    tracklet_len: int = 0
    samples: list[TrackSampleRecord] = field(default_factory=list)

    @property
    def box_xywh_norm(self) -> dict[str, float] | None:
        if not self.samples:
            return None
        return self.samples[-1].box


@dataclass(frozen=True, slots=True)
class TrackSampleRecord:
    source_frame: int
    box: dict[str, float]
    sample_kind: str
    detector_score: float | None
    association_score: float | None
    score_type: str
    occluded: bool = False


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def linear_assignment(
    cost: np.ndarray,
    thresh: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    matched: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    flat = [(cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    for value, row, col in sorted(flat, key=lambda item: item[0]):
        if value > thresh:
            continue
        if row in used_rows or col in used_cols:
            continue
        matched.append((row, col))
        used_rows.add(row)
        used_cols.add(col)
    unmatched_rows = [i for i in range(cost.shape[0]) if i not in used_rows]
    unmatched_cols = [j for j in range(cost.shape[1]) if j not in used_cols]
    return matched, unmatched_rows, unmatched_cols


def _det_to_tlbr(det: Detection) -> np.ndarray:
    return np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)


def _norm_box_from_tlbr(tlbr: np.ndarray, frame_width: int, frame_height: int) -> dict[str, float]:
    x1, y1, x2, y2 = tlbr
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    return {
        "x": _clamp01(float(x1) / frame_width),
        "y": _clamp01(float(y1) / frame_height),
        "width": _clamp01(w / frame_width),
        "height": _clamp01(h / frame_height),
    }


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _interpolate_box(
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


@dataclass(slots=True)
class ByteTracker:
    """Two-threshold association without appearance embeddings."""

    frame_width: int
    frame_height: int
    track_thresh: float = DEFAULT_TRACK_THRESH
    det_thresh: float = DEFAULT_LOW_DET_THRESH
    match_iou: float = 0.5
    max_time_lost: int = 30
    _tracks: list[STrack] = field(default_factory=list)
    _lost: list[STrack] = field(default_factory=list)
    _removed: list[STrack] = field(default_factory=list)
    _next_id: int = 1
    _frame_id: int = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._lost.clear()
        self._removed.clear()
        self._next_id = 1
        self._frame_id = 0

    def update(
        self,
        detections: list[Detection],
        *,
        source_frame: int,
        norm_boxes: list[dict[str, float]] | None = None,
    ) -> list[STrack]:
        self._frame_id += 1
        norm_boxes = norm_boxes or [
            _norm_box_from_tlbr(_det_to_tlbr(det), self.frame_width, self.frame_height)
            for det in detections
        ]
        pairs = list(zip(detections, norm_boxes, strict=True))

        pool = self._tracks + self._lost
        active = [track for track in pool if track.state in {TrackState.TRACKED, TrackState.LOST}]

        if not pairs:
            for track in self._tracks:
                if track.state == TrackState.TRACKED:
                    track.state = TrackState.LOST
                    if track not in self._lost:
                        self._lost.append(track)
            self._tracks.clear()
            for track in list(self._lost):
                if source_frame - track.end_frame > self.max_time_lost:
                    track.state = TrackState.REMOVED
                    self._removed.append(track)
            self._lost = [track for track in self._lost if track.state != TrackState.REMOVED]
            return []

        high_pairs = [(det, box) for det, box in pairs if det.score >= self.track_thresh]
        low_pairs = [
            (det, box)
            for det, box in pairs
            if self.det_thresh <= det.score < self.track_thresh
        ]

        high_dets = [det for det, _ in high_pairs]
        matched, unmatched_tracks, unmatched_high = self._associate(active, high_dets)
        for track_idx, det_idx in matched:
            track = active[track_idx]
            det, norm = high_pairs[det_idx]
            match_iou = iou_xyxy(track.tlbr, _det_to_tlbr(det))
            self._update_track(
                track,
                det,
                norm,
                source_frame=source_frame,
                sample_kind="detected",
                association_score=match_iou,
            )

        remaining_tracks = [active[i] for i in unmatched_tracks]
        low_dets = [det for det, _ in low_pairs]
        matched_low, unmatched_tracks, _unmatched_low = self._associate(
            remaining_tracks,
            low_dets,
        )
        for track_idx, det_idx in matched_low:
            track = remaining_tracks[track_idx]
            det, norm = low_pairs[det_idx]
            match_iou = iou_xyxy(track.tlbr, _det_to_tlbr(det))
            self._update_track(
                track,
                det,
                norm,
                source_frame=source_frame,
                sample_kind="detected",
                association_score=match_iou,
            )

        for track_idx in unmatched_tracks:
            track = remaining_tracks[track_idx]
            if track.state == TrackState.TRACKED:
                track.state = TrackState.LOST
                if track in self._tracks:
                    self._tracks.remove(track)
                if track not in self._lost:
                    self._lost.append(track)

        for det_idx in unmatched_high:
            det, norm = high_pairs[det_idx]
            self._activate(det, norm, source_frame=source_frame)

        self._tracks = [track for track in self._tracks if track.state == TrackState.TRACKED]
        for track in list(self._lost):
            if source_frame - track.end_frame > self.max_time_lost:
                track.state = TrackState.REMOVED
                self._removed.append(track)
        self._lost = [track for track in self._lost if track.state != TrackState.REMOVED]
        return [track for track in self._tracks if track.samples]

    def _associate(
        self,
        tracks: list[STrack],
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                cost[i, j] = 1.0 - iou_xyxy(track.tlbr, _det_to_tlbr(det))
        return linear_assignment(cost, 1.0 - self.match_iou)

    def _activate(self, det: Detection, norm_box: dict[str, float], *, source_frame: int) -> STrack:
        track = STrack(
            track_id=self._next_id,
            tlbr=_det_to_tlbr(det),
            score=det.score,
            state=TrackState.TRACKED,
            frame_id=self._frame_id,
            start_frame=source_frame,
            end_frame=source_frame,
            tracklet_len=1,
        )
        self._next_id += 1
        track.samples.append(
            TrackSampleRecord(
                source_frame=source_frame,
                box=norm_box,
                sample_kind="detected",
                detector_score=det.score,
                association_score=None,
                score_type="raw_model",
            )
        )
        self._tracks.append(track)
        return track

    def _update_track(
        self,
        track: STrack,
        det: Detection,
        norm_box: dict[str, float],
        *,
        source_frame: int,
        sample_kind: str,
        association_score: float | None,
    ) -> None:
        track.tlbr = _det_to_tlbr(det)
        track.score = det.score
        track.state = TrackState.TRACKED
        track.frame_id = self._frame_id
        track.end_frame = source_frame
        track.tracklet_len += 1
        if track in self._lost:
            self._lost.remove(track)
        if track not in self._tracks:
            self._tracks.append(track)
        track.samples.append(
            TrackSampleRecord(
                source_frame=source_frame,
                box=norm_box,
                sample_kind=sample_kind,
                detector_score=det.score,
                association_score=association_score,
                score_type="raw_model",
            )
        )

    def fill_gaps(
        self,
        tracks: list[STrack],
        *,
        mandatory_frames: set[int],
        max_gap_frames: int,
    ) -> None:
        """Insert tracked/interpolated samples so gaps respect max_gap_frames."""
        for track in tracks:
            if not track.samples:
                continue
            expanded: list[TrackSampleRecord] = []
            samples = sorted(track.samples, key=lambda item: item.source_frame)
            for left, right in zip(samples, samples[1:], strict=False):
                expanded.append(left)
                gap = right.source_frame - left.source_frame
                if gap <= 1:
                    continue
                step = max_gap_frames
                frame = left.source_frame + step
                while frame < right.source_frame:
                    alpha = (frame - left.source_frame) / gap
                    box = _interpolate_box(left.box, right.box, alpha)
                    kind = "tracked" if frame in mandatory_frames else "interpolated"
                    expanded.append(
                        TrackSampleRecord(
                            source_frame=frame,
                            box=box,
                            sample_kind=kind,
                            detector_score=None,
                            association_score=None,
                            score_type="heuristic",
                            occluded=False,
                        )
                    )
                    frame += step
            expanded.append(samples[-1])
            track.samples = expanded
