import pytest

from media_analysis.frames import (
    Rational,
    analysis_transform,
    duration_sec,
    frame_count,
    map_box_from_analysis,
    ranges_meet,
)


@pytest.mark.unit
def test_duration_is_frame_count_times_rational_inverse() -> None:
    fps = Rational(30, 1)
    assert duration_sec(1800, fps) == 60
    assert duration_sec(1, Rational(24000, 1001)) == pytest.approx(1001 / 24000)


@pytest.mark.unit
def test_half_open_range_math() -> None:
    assert frame_count(0, 30) == 30
    assert ranges_meet(30, 30)
    assert not ranges_meet(30, 31)


@pytest.mark.unit
def test_uniform_analysis_boxes_stay_normalized() -> None:
    box = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}
    mapped = map_box_from_analysis(
        box,
        analysis_width=540,
        analysis_height=960,
        canonical_width=1080,
        canonical_height=1920,
    )
    assert mapped == box
    transform = analysis_transform(
        analysis_width=540,
        analysis_height=960,
        canonical_width=1080,
        canonical_height=1920,
    )
    assert transform["uniformScale"] == 0.5


@pytest.mark.unit
def test_non_uniform_analysis_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="uniform scale"):
        map_box_from_analysis(
            {"x": 0, "y": 0, "width": 1, "height": 1},
            analysis_width=100,
            analysis_height=100,
            canonical_width=1080,
            canonical_height=1920,
        )
