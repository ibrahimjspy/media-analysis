from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.synthetic_frames import (
    dict_frame_provider,
    pan_sequence,
    scale_sequence,
    solid_sequence,
    static_sequence,
    tilt_sequence,
)

from media_analysis.decode import ProbedMedia
from media_analysis.errors import CANCELLED, DECODE_FAILED, AnalyzeError
from media_analysis.features.motion import (
    MOTION_ANALYZER_VERSION,
    WINDOW_FRAMES,
    MotionAnalysisConfig,
    _feature_mask,
    analyze_motion,
    estimate_motion,
)
from media_analysis.frames import Rational

PATH = Path("unused.mp4")


def _media(frame_count: int, width: int = 320, height: int = 240) -> ProbedMedia:
    return ProbedMedia(
        width=width,
        height=height,
        fps=Rational(30, 1),
        frame_count=frame_count,
        duration=frame_count / 30.0,
        codec="h264",
        has_audio=False,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channel_count=None,
    )


def _required_motion_keys(sample: dict) -> set[str]:
    return {"startFrame", "endFrameExclusive", "magnitude", "dominantDirectionRad", "coherence"}


def _required_camera_keys(segment: dict) -> set[str]:
    return {"startFrame", "endFrameExclusive", "kind", "magnitude", "score", "scoreType"}


@pytest.mark.unit
def test_empty_video_returns_empty_motion() -> None:
    result = analyze_motion(PATH, _media(0), frame_provider=dict_frame_provider({}))
    analysis = result.as_motion_analysis()
    assert analysis["windowFrames"] == WINDOW_FRAMES
    assert analysis["samples"] == []
    assert analysis["cameraMoves"] == []


@pytest.mark.unit
def test_decode_failure_raises() -> None:
    media = _media(30)
    with pytest.raises(AnalyzeError) as exc:
        analyze_motion(
            PATH,
            media,
            frame_provider=dict_frame_provider({index: None for index in range(30)}),
        )
    assert exc.value.code == DECODE_FAILED


@pytest.mark.unit
def test_flat_solid_video_returns_static_zero_windows_not_decode_failed() -> None:
    frames = solid_sequence(45, color=(90, 90, 90))
    result = analyze_motion(PATH, _media(45), frame_provider=dict_frame_provider(frames))
    assert result.pairwise == ()
    assert all(sample["magnitude"] == 0.0 for sample in result.analysis["samples"])
    assert all(sample["coherence"] == 0.0 for sample in result.analysis["samples"])
    assert {move["kind"] for move in result.analysis["cameraMoves"]} == {"static"}


@pytest.mark.unit
def test_half_open_windows_cover_full_timeline() -> None:
    frame_count = 47
    frames = static_sequence(frame_count)
    result = analyze_motion(PATH, _media(frame_count), frame_provider=dict_frame_provider(frames))
    samples = result.analysis["samples"]
    assert samples[0]["startFrame"] == 0
    assert samples[-1]["endFrameExclusive"] == frame_count
    for left, right in zip(samples, samples[1:], strict=False):
        assert left["endFrameExclusive"] == right["startFrame"]
    assert all(
        sample["endFrameExclusive"] - sample["startFrame"] <= WINDOW_FRAMES for sample in samples
    )


@pytest.mark.unit
def test_static_sequence_classifies_static() -> None:
    frames = static_sequence(45)
    result = analyze_motion(PATH, _media(45), frame_provider=dict_frame_provider(frames))
    kinds = {move["kind"] for move in result.analysis["cameraMoves"]}
    assert "static" in kinds
    assert all(move["scoreType"] == "heuristic" for move in result.analysis["cameraMoves"])


@pytest.mark.unit
def test_pan_sequence_detects_pan() -> None:
    frames = pan_sequence(45, shift_px=6)
    result = analyze_motion(PATH, _media(45), frame_provider=dict_frame_provider(frames))
    kinds = {move["kind"] for move in result.analysis["cameraMoves"]}
    assert "pan" in kinds


@pytest.mark.unit
def test_tilt_sequence_detects_tilt() -> None:
    frames = tilt_sequence(45, shift_px=6)
    result = analyze_motion(PATH, _media(45), frame_provider=dict_frame_provider(frames))
    kinds = {move["kind"] for move in result.analysis["cameraMoves"]}
    assert "tilt" in kinds


@pytest.mark.unit
def test_scale_sequence_detects_push_in() -> None:
    frames = scale_sequence(45, scale_step=1.018)
    result = analyze_motion(PATH, _media(45), frame_provider=dict_frame_provider(frames))
    kinds = {move["kind"] for move in result.analysis["cameraMoves"]}
    assert "push_in" in kinds


@pytest.mark.unit
def test_default_mask_keeps_center_features_for_centered_scene() -> None:
    width, height = 320, 240
    default = _feature_mask(width, height, 0, None)
    assert default[height // 2, width // 2] == 255


@pytest.mark.unit
def test_subject_boxes_exclude_supplied_regions_only() -> None:
    width, height = 320, 240
    margin_x = max(4, width // 12)
    margin_y = max(4, height // 12)
    boxes = {0: [{"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}]}
    masked = _feature_mask(width, height, 0, boxes)
    assert masked[height // 2, width // 2] == 0
    assert masked[margin_y + 2, margin_x + 2] == 255


@pytest.mark.unit
def test_centered_pan_works_without_subject_boxes() -> None:
    frames = pan_sequence(45, shift_px=6)
    result = analyze_motion(
        PATH,
        _media(45),
        frame_provider=dict_frame_provider(frames),
        subject_boxes_by_frame=None,
    )
    assert "pan" in {move["kind"] for move in result.analysis["cameraMoves"]}


@pytest.mark.unit
def test_motion_config_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="window_frames"):
        MotionAnalysisConfig(window_frames=0)
    with pytest.raises(ValueError, match="sample_stride"):
        MotionAnalysisConfig(sample_stride=0)
    with pytest.raises(ValueError, match="min_inlier_ratio"):
        MotionAnalysisConfig(min_inlier_ratio=1.5)


@pytest.mark.unit
def test_motion_config_invalid_window_does_not_loop_forever() -> None:
    cfg = MotionAnalysisConfig(window_frames=1)
    media = _media(10)
    frames = static_sequence(10)
    result = estimate_motion(media, dict_frame_provider(frames), config=cfg)
    assert len(result.analysis["samples"]) == 10


@pytest.mark.unit
def test_motion_output_shape_and_no_edit_fields() -> None:
    frames = pan_sequence(30)
    result = analyze_motion(PATH, _media(30), frame_provider=dict_frame_provider(frames))
    analysis = result.analysis
    assert analysis["windowFrames"] == WINDOW_FRAMES
    assert _required_motion_keys(analysis["samples"][0]) <= set(analysis["samples"][0].keys())
    camera = analysis["cameraMoves"][0]
    assert _required_camera_keys(camera) <= set(camera.keys())
    forbidden = {"keep", "trim", "reject", "editDecision", "editPlan"}
    assert forbidden.isdisjoint(analysis.keys())
    for sample in analysis["samples"]:
        assert forbidden.isdisjoint(sample.keys())
    for move in analysis["cameraMoves"]:
        assert forbidden.isdisjoint(move.keys())
        assert move["scoreType"] in {"heuristic", "calibrated_probability"}


@pytest.mark.unit
def test_deterministic_motion_output() -> None:
    frames = pan_sequence(30, shift_px=5)
    first = analyze_motion(PATH, _media(30), frame_provider=dict_frame_provider(frames))
    second = analyze_motion(PATH, _media(30), frame_provider=dict_frame_provider(frames))
    assert first.analysis == second.analysis


@pytest.mark.unit
def test_cancellation_raises() -> None:
    frames = pan_sequence(30)
    cancelled = {"value": False}

    def cancel_check() -> None:
        cancelled["value"] = True
        raise AnalyzeError(CANCELLED, "Analysis cancelled")

    with pytest.raises(AnalyzeError) as exc:
        analyze_motion(
            PATH,
            _media(30),
            frame_provider=dict_frame_provider(frames),
            cancel_check=cancel_check,
        )
    assert exc.value.code == CANCELLED
    assert cancelled["value"] is True


@pytest.mark.unit
def test_analyzer_version_marks_provisional_and_encodes_thresholds() -> None:
    assert "provisional" in MOTION_ANALYZER_VERSION
    assert "w15" in MOTION_ANALYZER_VERSION
    assert "-sm0.0018" in MOTION_ANALYZER_VERSION
    assert "-wm0.028" in MOTION_ANALYZER_VERSION
    assert MotionAnalysisConfig().analyzer_version == MOTION_ANALYZER_VERSION
