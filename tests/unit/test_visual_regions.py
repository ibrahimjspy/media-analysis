"""Verify detector boundary geometry and opt-in behavior without model downloads."""

from media_analysis.features.visual_regions import available, normalize


def test_disabled_does_not_require_optional_runtime():
    assert available(False) is False


def test_geometry_preserves_normalized_canonical_bounds():
    assert normalize([10, 20, 40, 60], 100, 100) == dict(x=0.1, y=0.2, width=0.3, height=0.4)


def test_invalid_boxes_are_not_repaired():
    for box in ([0, 0, 101, 90], [0, 0, 0, 90], [float("nan"), 0, 20, 30]):
        assert normalize(box, 100, 100) is None
