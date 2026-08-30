from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.unit.synthetic_frames import (
    RecordingFrameProvider,
    blurred_frame,
    checkerboard,
    clipped_highlights,
    clipped_shadows,
    dict_frame_provider,
    pan_sequence,
    solid_color,
    solid_sequence,
    static_sequence,
)

from media_analysis.decode import ProbedMedia
from media_analysis.errors import CANCELLED, DECODE_FAILED, INVALID_REQUEST, AnalyzeError
from media_analysis.features.motion import MotionAnalyzeResult, analyze_motion
from media_analysis.features.quality import (
    QUALITY_POLICY_VERSION,
    QualityAnalysisConfig,
    _sharpness_at_working_size,
    analyze_quality,
    make_internal_motion_estimator,
    shake_from_motion_pairwise,
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


def _shots(*ranges: tuple[int, int]) -> list[dict]:
    return [{"startFrame": start, "endFrameExclusive": end} for start, end in ranges]


@pytest.mark.unit
def test_empty_shots_returns_empty_per_shot() -> None:
    result = analyze_quality(PATH, _media(0), shots=[], frame_provider=dict_frame_provider({}))
    assert result.analysis["perShot"] == []


@pytest.mark.unit
def test_decode_failure_raises_when_no_representatives_decode() -> None:
    shots = _shots((0, 30))
    with pytest.raises(AnalyzeError) as exc:
        analyze_quality(
            PATH,
            _media(30),
            shots=shots,
            frame_provider=dict_frame_provider({index: None for index in range(30)}),
        )
    assert exc.value.code == DECODE_FAILED


@pytest.mark.unit
def test_partial_representative_decode_fails_feature() -> None:
    frame = checkerboard()
    with pytest.raises(AnalyzeError) as exc:
        analyze_quality(
            PATH,
            _media(50),
            shots=_shots((0, 30), (30, 50)),
            frame_provider=dict_frame_provider({15: frame}),
        )
    assert exc.value.code == DECODE_FAILED


@pytest.mark.unit
def test_invalid_shot_range_rejected() -> None:
    with pytest.raises(AnalyzeError) as exc:
        analyze_quality(
            PATH,
            _media(30),
            shots=[{"startFrame": 0, "endFrameExclusive": 40}],
            frame_provider=dict_frame_provider(static_sequence(30)),
        )
    assert exc.value.code == INVALID_REQUEST


def _static_motion() -> MotionAnalyzeResult:
    return MotionAnalyzeResult(
        analysis={"windowFrames": 15, "samples": [], "cameraMoves": []},
        pairwise=(),
    )


@pytest.mark.unit
def test_blurred_frame_flags_soft_focus() -> None:
    sharp = checkerboard()
    blurred = blurred_frame(sharp, ksize=51)
    frames = static_sequence(30)
    frames[15] = sharp
    sharpness_sharp = analyze_quality(
        PATH,
        _media(30),
        shots=_shots((0, 30)),
        motion=_static_motion(),
        frame_provider=dict_frame_provider(frames),
    ).analysis["perShot"][0]["sharpness"]
    blurred_frames = static_sequence(30)
    blurred_frames[15] = blurred
    cfg = QualityAnalysisConfig(soft_focus_threshold=sharpness_sharp * 0.95)
    entry = analyze_quality(
        PATH,
        _media(30),
        shots=_shots((0, 30)),
        config=cfg,
        motion=_static_motion(),
        frame_provider=dict_frame_provider(blurred_frames),
    ).analysis["perShot"][0]
    assert entry["sharpness"] < sharpness_sharp
    assert "soft_focus" in entry["flags"]


@pytest.mark.unit
def test_sharpness_uses_working_size_not_inverse_area() -> None:
    frame = checkerboard(width=640, height=480)
    at_native = _sharpness_at_working_size(frame, 640, 480, 540, 960)
    upscaled = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)
    at_upscaled = _sharpness_at_working_size(upscaled, 1280, 960, 540, 960)
    assert at_native == pytest.approx(at_upscaled, rel=0.05)


@pytest.mark.unit
def test_clipped_frames_flag_exposure_issues() -> None:
    base = checkerboard()
    blown = clipped_highlights(base, fraction=0.40)
    crushed = clipped_shadows(base, fraction=0.40)
    blown_frames = static_sequence(30)
    blown_frames[15] = blown
    crushed_frames = static_sequence(30)
    crushed_frames[15] = crushed
    result_blown = analyze_quality(
        PATH,
        _media(30),
        shots=_shots((0, 30)),
        motion=_static_motion(),
        frame_provider=dict_frame_provider(blown_frames),
    )
    result_crushed = analyze_quality(
        PATH,
        _media(30),
        shots=_shots((0, 30)),
        motion=_static_motion(),
        frame_provider=dict_frame_provider(crushed_frames),
    )
    assert "blown_highlights" in result_blown.analysis["perShot"][0]["flags"]
    assert "crushed_shadows" in result_crushed.analysis["perShot"][0]["flags"]


@pytest.mark.unit
def test_near_duplicate_points_to_earlier_shot_only() -> None:
    frame = checkerboard()
    frames = static_sequence(60)
    frames[15] = frame
    frames[45] = frame.copy()
    result = analyze_quality(
        PATH,
        _media(60),
        shots=_shots((0, 30), (30, 60)),
        motion=_static_motion(),
        frame_provider=dict_frame_provider(frames),
    )
    first, second = result.analysis["perShot"]
    assert first["duplicateOfShotIndex"] is None
    assert second["duplicateOfShotIndex"] == 0
    assert "near_duplicate" in second["flags"]
    assert second["duplicateScore"] is not None


@pytest.mark.unit
def test_black_and_white_uniform_shots_are_not_near_duplicates() -> None:
    black = solid_color(color=(0, 0, 0))
    white = solid_color(color=(255, 255, 255))
    frames = solid_sequence(60, color=(64, 64, 64))
    frames[15] = black
    frames[45] = white
    result = analyze_quality(
        PATH,
        _media(60),
        shots=_shots((0, 30), (30, 60)),
        motion=_static_motion(),
        frame_provider=dict_frame_provider(frames),
    )
    first, second = result.analysis["perShot"]
    assert first["duplicateOfShotIndex"] is None
    assert second["duplicateOfShotIndex"] is None
    assert "near_duplicate" not in second["flags"]


@pytest.mark.unit
def test_detailed_static_checkerboard_is_not_severe_shake() -> None:
    frames = static_sequence(45)
    motion = analyze_motion(PATH, _media(45), frame_provider=dict_frame_provider(frames))
    result = analyze_quality(
        PATH,
        _media(45),
        shots=_shots((0, 45)),
        motion=motion,
        frame_provider=dict_frame_provider({22: frames[22]}),
    )
    entry = result.analysis["perShot"][0]
    assert entry["shake"] == 0.0
    assert "severe_shake" not in entry["flags"]


@pytest.mark.unit
def test_shake_uses_motion_dependency_without_recompute() -> None:
    pan_frames = pan_sequence(45, shift_px=8)
    provider = RecordingFrameProvider(pan_frames)
    motion = analyze_motion(PATH, _media(45), frame_provider=provider)
    motion_reads = len(provider.read_indices)

    quality_provider = RecordingFrameProvider(pan_frames)
    with_dependency = analyze_quality(
        PATH,
        _media(45),
        shots=_shots((0, 45)),
        motion=motion,
        frame_provider=quality_provider,
    )
    dep_shake = with_dependency.analysis["perShot"][0]["shake"]
    expected = shake_from_motion_pairwise(motion.pairwise, 0, 45)
    assert expected is not None
    assert dep_shake == round(expected, 6)
    assert len(quality_provider.read_indices) == 1
    assert motion_reads >= 2


@pytest.mark.unit
def test_compute_motion_callback_runs_shared_primitive_once() -> None:
    pan_frames = pan_sequence(30, shift_px=6)
    provider = RecordingFrameProvider(pan_frames)
    media = _media(30)
    estimator = make_internal_motion_estimator(media, provider)
    first_motion = estimator()
    second_motion = estimator()
    assert first_motion is second_motion

    first = analyze_quality(
        PATH,
        media,
        shots=_shots((0, 30)),
        compute_motion=estimator,
        frame_provider=provider,
    )
    second = analyze_quality(
        PATH,
        media,
        shots=_shots((0, 30)),
        compute_motion=estimator,
        frame_provider=provider,
    )
    assert first.analysis["perShot"][0]["shake"] == second.analysis["perShot"][0]["shake"]


@pytest.mark.unit
def test_no_forbidden_edit_fields() -> None:
    frame = checkerboard()
    frames = static_sequence(30)
    frames[15] = frame
    result = analyze_quality(
        PATH,
        _media(30),
        shots=_shots((0, 30)),
        motion=_static_motion(),
        frame_provider=dict_frame_provider(frames),
    )
    forbidden = {"keep", "trim", "reject", "editDecision", "editPlan"}
    assert forbidden.isdisjoint(result.analysis.keys())
    for entry in result.analysis["perShot"]:
        assert forbidden.isdisjoint(entry.keys())
        assert entry["policyVersion"] == QUALITY_POLICY_VERSION


@pytest.mark.unit
def test_representative_frame_is_shot_middle_and_emits_all_shots() -> None:
    frame = checkerboard()
    frames = static_sequence(50)
    frames[15] = frame
    frames[40] = frame.copy()
    result = analyze_quality(
        PATH,
        _media(50),
        shots=_shots((0, 30), (30, 50)),
        motion=_static_motion(),
        frame_provider=dict_frame_provider(frames),
    )
    assert len(result.analysis["perShot"]) == 2
    assert [entry["shotIndex"] for entry in result.analysis["perShot"]] == [0, 1]


@pytest.mark.unit
def test_deterministic_quality_output() -> None:
    frame = checkerboard()
    frames = static_sequence(60)
    frames[15] = frame
    frames[45] = frame.copy()
    kwargs = dict(
        path=PATH,
        media=_media(60),
        shots=_shots((0, 30), (30, 60)),
        motion=_static_motion(),
        frame_provider=dict_frame_provider(frames),
    )
    first = analyze_quality(**kwargs)
    second = analyze_quality(**kwargs)
    assert first.analysis == second.analysis


@pytest.mark.unit
def test_cancellation_raises() -> None:
    frames = static_sequence(30)

    def cancel_check() -> None:
        raise AnalyzeError(CANCELLED, "Analysis cancelled")

    with pytest.raises(AnalyzeError) as exc:
        analyze_quality(
            PATH,
            _media(30),
            shots=_shots((0, 30)),
            frame_provider=dict_frame_provider(frames),
            cancel_check=cancel_check,
        )
    assert exc.value.code == CANCELLED


@pytest.mark.unit
def test_allowed_quality_flags_only() -> None:
    allowed = {
        "soft_focus",
        "severe_shake",
        "blown_highlights",
        "crushed_shadows",
        "near_duplicate",
    }
    frame = checkerboard()
    frames = static_sequence(30)
    frames[15] = frame
    result = analyze_quality(
        PATH,
        _media(30),
        shots=_shots((0, 30)),
        motion=_static_motion(),
        frame_provider=dict_frame_provider(frames),
    )
    for entry in result.analysis["perShot"]:
        assert set(entry["flags"]).issubset(allowed)


@pytest.mark.unit
def test_policy_version_marks_provisional_and_encodes_working_size() -> None:
    assert "provisional" in QUALITY_POLICY_VERSION
    assert "ws540x960" in QUALITY_POLICY_VERSION
    assert QualityAnalysisConfig().policy_version == QUALITY_POLICY_VERSION


@pytest.mark.unit
def test_quality_config_validation() -> None:
    with pytest.raises(ValueError, match="soft_focus_threshold"):
        QualityAnalysisConfig(soft_focus_threshold=0)
    with pytest.raises(ValueError, match="duplicate_max_hash_distance"):
        QualityAnalysisConfig(duplicate_max_hash_distance=-1)
