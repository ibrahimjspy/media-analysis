import pytest

from media_analysis.features.ocr import (
    MAX_SAMPLE_GAP_MS,
    OCR_SAMPLER_VERSION,
    FrameDetection,
    box_iou,
    classify_region_kind,
    detection_in_roi,
    merge_temporal_detections,
    sample_ocr_frames,
)
from media_analysis.frames import Rational


@pytest.mark.unit
def test_detection_in_top_and_bottom_rois() -> None:
    assert detection_in_roi({"x": 0.1, "y": 0.05, "width": 0.2, "height": 0.05})
    assert detection_in_roi({"x": 0.1, "y": 0.88, "width": 0.2, "height": 0.05})
    assert not detection_in_roi({"x": 0.1, "y": 0.4, "width": 0.2, "height": 0.1})


@pytest.mark.unit
def test_sample_frames_uses_1000ms_cadence_version() -> None:
    assert OCR_SAMPLER_VERSION.endswith("1000ms-1.0.0")
    fps = Rational(30, 1)
    frames = sample_ocr_frames(120, fps, None)
    gap_frames = max(1, int(round(MAX_SAMPLE_GAP_MS * fps.numerator / (fps.denominator * 1000))))
    assert gap_frames == 30
    assert frames == sorted(set(range(0, 120, gap_frames)) | {119})


@pytest.mark.unit
def test_sample_frames_is_deterministic_with_gap_and_shot_burst() -> None:
    fps = Rational(30, 1)
    shots = [
        {"startFrame": 0, "endFrameExclusive": 60},
        {"startFrame": 60, "endFrameExclusive": 120},
    ]
    first = sample_ocr_frames(120, fps, shots)
    second = sample_ocr_frames(120, fps, shots)
    assert first == second
    assert 0 in first
    assert 119 in first
    assert 60 in first
    assert 59 in first
    assert all(0 <= frame < 120 for frame in first)


@pytest.mark.unit
def test_sample_frames_empty_for_zero_frame_count() -> None:
    assert sample_ocr_frames(0, Rational(30, 1), None) == []


@pytest.mark.unit
def test_box_iou_for_overlapping_boxes() -> None:
    a = {"x": 0.0, "y": 0.0, "width": 0.5, "height": 0.5}
    b = {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}
    assert box_iou(a, b) == pytest.approx(0.0625 / 0.4375)


@pytest.mark.unit
def test_merge_links_same_location_across_frames() -> None:
    det_a = FrameDetection(
        source_frame=0,
        polygon=((0.1, 0.05), (0.3, 0.05), (0.3, 0.1), (0.1, 0.1)),
        box={"x": 0.1, "y": 0.05, "width": 0.2, "height": 0.05},
        detector_score=0.9,
        shot_index=0,
    )
    det_b = FrameDetection(
        source_frame=6,
        polygon=((0.11, 0.05), (0.31, 0.05), (0.31, 0.1), (0.11, 0.1)),
        box={"x": 0.11, "y": 0.05, "width": 0.2, "height": 0.05},
        detector_score=0.88,
        shot_index=0,
    )
    regions = merge_temporal_detections([det_a, det_b], frame_count=100, shots=None)
    assert len(regions) == 1
    assert regions[0]["startFrame"] == 0
    assert regions[0]["endFrameExclusive"] == 7
    assert len(regions[0]["samples"]) == 2


@pytest.mark.unit
def test_merge_splits_distant_boxes_into_separate_regions() -> None:
    left = FrameDetection(
        source_frame=0,
        polygon=((0.05, 0.05), (0.15, 0.05), (0.15, 0.1), (0.05, 0.1)),
        box={"x": 0.05, "y": 0.05, "width": 0.1, "height": 0.05},
        detector_score=0.9,
        shot_index=0,
    )
    right = FrameDetection(
        source_frame=0,
        polygon=((0.75, 0.88), (0.95, 0.88), (0.95, 0.95), (0.75, 0.95)),
        box={"x": 0.75, "y": 0.88, "width": 0.2, "height": 0.07},
        detector_score=0.85,
        shot_index=0,
    )
    regions = merge_temporal_detections([left, right], frame_count=100, shots=None)
    assert len(regions) == 2


@pytest.mark.unit
def test_merge_is_one_to_one_per_frame_for_overlapping_boxes() -> None:
    first = FrameDetection(
        source_frame=0,
        polygon=((0.1, 0.05), (0.3, 0.05), (0.3, 0.1), (0.1, 0.1)),
        box={"x": 0.1, "y": 0.05, "width": 0.2, "height": 0.05},
        detector_score=0.9,
        shot_index=0,
    )
    second = FrameDetection(
        source_frame=0,
        polygon=((0.12, 0.05), (0.32, 0.05), (0.32, 0.1), (0.12, 0.1)),
        box={"x": 0.12, "y": 0.05, "width": 0.2, "height": 0.05},
        detector_score=0.88,
        shot_index=0,
    )
    regions = merge_temporal_detections([first, second], frame_count=100, shots=None)
    assert len(regions) == 2
    assert all(len(region["samples"]) == 1 for region in regions)


@pytest.mark.unit
def test_merge_does_not_cross_shot_cut_with_same_box() -> None:
    box = {"x": 0.1, "y": 0.05, "width": 0.2, "height": 0.05}
    polygon = ((0.1, 0.05), (0.3, 0.05), (0.3, 0.1), (0.1, 0.1))
    before_cut = FrameDetection(
        source_frame=59,
        polygon=polygon,
        box=box,
        detector_score=0.9,
        shot_index=0,
    )
    after_cut = FrameDetection(
        source_frame=60,
        polygon=polygon,
        box=box,
        detector_score=0.9,
        shot_index=1,
    )
    shots = [
        {"startFrame": 0, "endFrameExclusive": 60},
        {"startFrame": 60, "endFrameExclusive": 120},
    ]
    regions = merge_temporal_detections(
        [before_cut, after_cut],
        frame_count=120,
        shots=shots,
    )
    assert len(regions) == 2
    assert {region["startFrame"] for region in regions} == {59, 60}
    assert {region["id"] for region in regions} == {"ocr-0", "ocr-1"}


@pytest.mark.unit
def test_persistent_overlay_heuristic_for_long_stable_region() -> None:
    samples = [
        {
            "sourceFrame": i,
            "box": {"x": 0.1, "y": 0.02, "width": 0.2, "height": 0.04},
        }
        for i in range(0, 80, 4)
    ]
    assert classify_region_kind(samples, frame_count=100) == "persistent_overlay"


@pytest.mark.unit
def test_short_region_stays_ocr_text() -> None:
    samples = [
        {
            "sourceFrame": 0,
            "box": {"x": 0.1, "y": 0.02, "width": 0.2, "height": 0.04},
        }
    ]
    assert classify_region_kind(samples, frame_count=100) == "ocr_text"
