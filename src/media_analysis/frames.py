"""Canonical half-open frame math. Seconds are display-only for visual features."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rational:
    numerator: int
    denominator: int

    def as_float(self) -> float:
        if self.denominator == 0:
            raise ValueError("rational denominator is 0")
        return self.numerator / self.denominator

    def as_dict(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}


def duration_sec(frame_count: int, fps: Rational) -> float:
    """durationSec = frameCount × fps.denominator / fps.numerator"""
    if fps.numerator == 0:
        raise ValueError("fps numerator is 0")
    return frame_count * fps.denominator / fps.numerator


def frame_count(start_frame: int, end_frame_exclusive: int) -> int:
    if end_frame_exclusive < start_frame:
        raise ValueError("endFrameExclusive is before startFrame")
    return end_frame_exclusive - start_frame


def ranges_meet(left_end: int, right_start: int) -> bool:
    """Adjacent half-open ranges meet without overlap: [0, 30) + [30, 60)."""
    return left_end == right_start


def presentation_time_sec(frame: int, fps: Rational) -> float:
    return duration_sec(frame, fps)


def map_box_from_analysis(
    box: dict[str, float],
    *,
    analysis_width: int,
    analysis_height: int,
    canonical_width: int,
    canonical_height: int,
) -> dict[str, float]:
    """Boxes are already 0..1 of the analysis frame; they map 1:1 if letterbox-free.

    analysisResolution is a measure size. When it is a uniform scale of the
    canonical frame, normalized boxes are identical. This function exists so
    the transform is explicit, not implied.
    """
    if analysis_width <= 0 or analysis_height <= 0:
        raise ValueError("analysis resolution must be positive")
    if canonical_width <= 0 or canonical_height <= 0:
        raise ValueError("canonical size must be positive")
    scale_x = analysis_width / canonical_width
    scale_y = analysis_height / canonical_height
    if abs(scale_x - scale_y) > 1e-6:
        # Non-uniform scale would distort boxes. Callers must letterbox first.
        raise ValueError("analysisResolution is not a uniform scale of canonical size")
    return {
        "x": box["x"],
        "y": box["y"],
        "width": box["width"],
        "height": box["height"],
    }


def analysis_transform(
    *,
    analysis_width: int,
    analysis_height: int,
    canonical_width: int,
    canonical_height: int,
) -> dict[str, object]:
    return {
        "analysisResolution": {"width": analysis_width, "height": analysis_height},
        "canonicalResolution": {"width": canonical_width, "height": canonical_height},
        "uniformScale": analysis_width / canonical_width,
    }
