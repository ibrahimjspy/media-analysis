import pytest

from media_analysis.tools.ppocr_recall import (
    assert_recall_floors,
    dedupe_temporal_boxes,
    render_latin_banner,
    score_detections,
)


@pytest.mark.unit
def test_latin_banner_stays_in_top_sampler_roi() -> None:
    image, box = render_latin_banner("HELLO")
    assert image.shape == (360, 640, 3)
    assert image.max() == 255
    assert box["y"] + box["height"] <= 0.20
    assert 0.0 < box["width"] < 1.0


@pytest.mark.unit
def test_recall_floors_pass_on_overlapping_prediction() -> None:
    expected = [{"x": 0.1, "y": 0.05, "width": 0.4, "height": 0.1}]
    predicted = [{"x": 0.12, "y": 0.05, "width": 0.4, "height": 0.1}]
    scores = assert_recall_floors(predicted, expected)
    assert scores["recall"] == 1.0
    assert scores["precision"] == 1.0


@pytest.mark.unit
def test_recall_floors_fail_when_text_is_missed() -> None:
    expected = [{"x": 0.1, "y": 0.05, "width": 0.4, "height": 0.1}]
    with pytest.raises(AssertionError, match="recall"):
        assert_recall_floors([], expected)


@pytest.mark.unit
def test_dedupe_temporal_boxes_collapses_repeated_samples() -> None:
    box = {"x": 0.1, "y": 0.05, "width": 0.4, "height": 0.1}
    jitter = {"x": 0.11, "y": 0.05, "width": 0.4, "height": 0.1}
    other = {"x": 0.7, "y": 0.05, "width": 0.2, "height": 0.1}
    assert len(dedupe_temporal_boxes([box, box, jitter, other])) == 2


@pytest.mark.unit
def test_recall_precision_recovers_after_temporal_dedupe() -> None:
    expected = [{"x": 0.1, "y": 0.05, "width": 0.4, "height": 0.1}]
    repeated = expected * 8
    with pytest.raises(AssertionError, match="precision"):
        assert_recall_floors(repeated, expected)
    scores = assert_recall_floors(dedupe_temporal_boxes(repeated), expected)
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0


@pytest.mark.unit
def test_empty_expected_and_predicted_is_perfect() -> None:
    scores = score_detections([], [])
    assert scores["recall"] == 1.0
    assert scores["precision"] == 1.0
