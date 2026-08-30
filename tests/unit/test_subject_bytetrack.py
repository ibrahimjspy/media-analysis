"""Unit tests for ByteTrack-style association and shot-boundary behavior."""

from __future__ import annotations

import pytest

from media_analysis.features.bytetrack import (
    ASSOCIATION_MEASURE,
    ByteTracker,
    TrackState,
    _det_to_tlbr,
    iou_xyxy,
)
from media_analysis.features.yolox import Detection


def _det(x1: float, y1: float, x2: float, y2: float, score: float) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score)


@pytest.mark.unit
def test_tracker_creates_track_on_high_confidence_enter() -> None:
    tracker = ByteTracker(frame_width=320, frame_height=240)
    active = tracker.update([_det(10, 10, 60, 120, 0.9)], source_frame=0)
    assert len(active) == 1
    assert active[0].track_id == 1
    assert active[0].samples[-1].sample_kind == "detected"
    assert active[0].samples[-1].detector_score == pytest.approx(0.9)
    assert active[0].samples[-1].association_score is None


@pytest.mark.unit
def test_tracker_association_score_is_match_iou() -> None:
    tracker = ByteTracker(frame_width=320, frame_height=240, match_iou=0.3)
    first_box = _det(10, 10, 60, 120, 0.9)
    second_box = _det(20, 12, 70, 122, 0.88)
    tracker.update([first_box], source_frame=0)
    second = tracker.update([second_box], source_frame=6)[0]
    expected_iou = iou_xyxy(_det_to_tlbr(first_box), _det_to_tlbr(second_box))
    assert second.samples[-1].association_score == pytest.approx(expected_iou)
    assert ASSOCIATION_MEASURE == "iou-xyxy-1.0.0"


@pytest.mark.unit
def test_low_confidence_never_starts_new_track() -> None:
    tracker = ByteTracker(frame_width=320, frame_height=240, match_iou=0.2)
    active = tracker.update([_det(10, 10, 60, 120, 0.15)], source_frame=0)
    assert active == []
    assert tracker._tracks == []
    assert tracker._next_id == 1


@pytest.mark.unit
def test_low_confidence_can_match_existing_track_with_iou_score() -> None:
    tracker = ByteTracker(frame_width=320, frame_height=240, match_iou=0.2)
    first_box = _det(10, 10, 60, 120, 0.9)
    low_box = _det(12, 11, 62, 121, 0.15)
    tracker.update([first_box], source_frame=0)
    active = tracker.update([low_box], source_frame=6)
    assert len(active) == 1
    assert active[0].samples[-1].detector_score == pytest.approx(0.15)
    expected_iou = iou_xyxy(_det_to_tlbr(first_box), _det_to_tlbr(low_box))
    assert active[0].samples[-1].association_score == pytest.approx(expected_iou)


@pytest.mark.unit
def test_tracker_marks_lost_when_detection_disappears() -> None:
    tracker = ByteTracker(frame_width=320, frame_height=240)
    tracker.update([_det(10, 10, 60, 120, 0.9)], source_frame=0)
    tracker.update([], source_frame=6)
    assert len(tracker._lost) == 1
    assert tracker._lost[0].state == TrackState.LOST


@pytest.mark.unit
def test_lost_track_removed_by_source_frame_gap_under_sparse_sampling() -> None:
    tracker = ByteTracker(frame_width=320, frame_height=240, max_time_lost=12)
    tracker.update([_det(10, 10, 60, 120, 0.9)], source_frame=0)
    tracker.update([], source_frame=6)
    assert len(tracker._lost) == 1
    tracker.update([], source_frame=18)
    assert tracker._lost == []
    assert len(tracker._removed) == 1
    assert tracker._removed[0].state == TrackState.REMOVED


@pytest.mark.unit
def test_tracker_reset_clears_ids_for_next_shot() -> None:
    tracker = ByteTracker(frame_width=320, frame_height=240)
    tracker.update([_det(10, 10, 60, 120, 0.9)], source_frame=0)
    tracker.reset()
    active = tracker.update([_det(15, 15, 65, 125, 0.91)], source_frame=100)
    assert active[0].track_id == 1
    assert active[0].start_frame == 100


@pytest.mark.unit
def test_fill_gaps_emits_interpolated_samples() -> None:
    tracker = ByteTracker(frame_width=320, frame_height=240, match_iou=0.3)
    track = tracker.update([_det(10, 10, 60, 120, 0.9)], source_frame=0)[0]
    tracker.update([_det(20, 12, 70, 122, 0.88)], source_frame=12)
    tracker.fill_gaps([track], mandatory_frames={0, 12}, max_gap_frames=6)
    kinds = {sample.sample_kind for sample in track.samples}
    assert "detected" in kinds
    assert "interpolated" in kinds
    frames = [sample.source_frame for sample in track.samples]
    assert 6 in frames
