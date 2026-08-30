from __future__ import annotations

import numpy as np
import pytest

from media_analysis.decode import ProbedMedia
from media_analysis.errors import CANCELLED, INVALID_REQUEST, AnalyzeError
from media_analysis.features.exposure import (
    BT709_LUMA_COEFFICIENTS,
    CONTRAST_FORMULA,
    EXPOSURE_ANALYZER_VERSION,
    LUMA_COLOR_SPACE,
    MAX_LUMA_SAMPLE_PIXELS_PER_FRAME,
    aggregate_shot_metrics,
    analyze_exposure,
    bt709_luma_from_bgr,
    dominant_colors_from_bgr,
    estimate_color_temperature_k,
    exposure_policy_metadata,
    mean_saturation_from_bgr,
    metrics_to_dict,
    sample_luma_values,
    srgb_uint8_to_linear,
)
from media_analysis.features.measure_common import representative_sample_frames
from media_analysis.frames import Rational


def _media(frame_count: int = 90) -> ProbedMedia:
    return ProbedMedia(
        width=64,
        height=48,
        fps=Rational(30, 1),
        frame_count=frame_count,
        duration=frame_count / 30.0,
        codec="h264",
        has_audio=False,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channel_count=None,
    )


def _solid_bgr(b: int, g: int, r: int, *, width: int = 64, height: int = 48) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (b, g, r)
    return frame


class DictFrameProvider:
    def __init__(self, frames: dict[int, np.ndarray]) -> None:
        self.frames = frames

    def __call__(self, frame_index: int) -> np.ndarray | None:
        return self.frames.get(frame_index)


@pytest.mark.unit
def test_luma_uses_linear_rgb_after_srgb_eotf() -> None:
    frame = _solid_bgr(0, 0, 255)
    luma = bt709_luma_from_bgr(frame)
    linear_r = float(srgb_uint8_to_linear(np.array(255, dtype=np.uint8)))
    expected = BT709_LUMA_COEFFICIENTS[0] * linear_r
    assert LUMA_COLOR_SPACE == "bt709_linear_rgb_after_srgb_eotf"
    assert pytest.approx(float(luma[0, 0]), rel=1e-4) == expected


@pytest.mark.unit
def test_sample_luma_values_are_bounded_per_frame() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    values = sample_luma_values(frame)
    assert values.size <= MAX_LUMA_SAMPLE_PIXELS_PER_FRAME
    meta = exposure_policy_metadata()
    assert meta["maxLumaSamplePixelsPerFrame"] == MAX_LUMA_SAMPLE_PIXELS_PER_FRAME
    assert "provisional" in EXPOSURE_ANALYZER_VERSION


@pytest.mark.unit
def test_contrast_is_p95_minus_p05() -> None:
    dark = _solid_bgr(0, 0, 0)
    bright = _solid_bgr(255, 255, 255)
    metrics = aggregate_shot_metrics(
        shot_index=0,
        luma_values=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        saturation_values=[0.0, 0.0],
        rgb_frames=[dark, bright],
    )
    assert metrics.contrast == pytest.approx(1.0, abs=1e-4)
    assert CONTRAST_FORMULA == "p95_luma_minus_p05_luma"


@pytest.mark.unit
def test_gray_and_low_saturation_color_temperature_is_null() -> None:
    gray = _solid_bgr(128, 128, 128)
    assert estimate_color_temperature_k(gray) is None
    low_sat = _solid_bgr(120, 125, 130)
    assert mean_saturation_from_bgr(low_sat) < 0.08
    assert estimate_color_temperature_k(low_sat) is None


@pytest.mark.unit
def test_warm_is_cooler_than_cool_without_fake_precision() -> None:
    warm = _solid_bgr(40, 120, 255)
    cool = _solid_bgr(255, 120, 40)
    warm_k = estimate_color_temperature_k(warm)
    cool_k = estimate_color_temperature_k(cool)
    assert warm_k is not None
    assert cool_k is not None
    assert warm_k < cool_k


@pytest.mark.unit
def test_dominant_colors_are_stable_and_quantized() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, :16] = (10, 20, 200)
    frame[:, 16:] = (10, 20, 40)
    first = dominant_colors_from_bgr(frame)
    second = dominant_colors_from_bgr(frame)
    assert first == second
    assert sum(color["fraction"] for color in first) == pytest.approx(1.0, abs=1e-3)
    assert all(str(color["hex"]).startswith("#") for color in first)


@pytest.mark.unit
def test_representative_sampling_is_deterministic() -> None:
    assert representative_sample_frames(0, 30, max_samples=5) == representative_sample_frames(
        0, 30, max_samples=5
    )
    assert 0 in representative_sample_frames(0, 30, max_samples=5)
    assert 29 in representative_sample_frames(0, 30, max_samples=5)


@pytest.mark.unit
def test_analyze_exposure_per_shot_synthetic() -> None:
    media = _media(30)
    frames = {
        0: _solid_bgr(20, 20, 20),
        7: _solid_bgr(20, 20, 20),
        14: _solid_bgr(20, 20, 20),
        22: _solid_bgr(20, 20, 20),
        29: _solid_bgr(20, 20, 20),
        15: _solid_bgr(240, 240, 240),
        16: _solid_bgr(240, 240, 240),
        17: _solid_bgr(240, 240, 240),
        23: _solid_bgr(240, 240, 240),
        24: _solid_bgr(240, 240, 240),
        25: _solid_bgr(240, 240, 240),
        26: _solid_bgr(240, 240, 240),
        27: _solid_bgr(240, 240, 240),
        28: _solid_bgr(240, 240, 240),
    }
    shots = [
        {"startFrame": 0, "endFrameExclusive": 15},
        {"startFrame": 15, "endFrameExclusive": 30},
    ]
    result = analyze_exposure(
        path=__file__,
        media=media,
        shots=shots,
        frame_provider=DictFrameProvider(frames),
    )
    assert result.status == "completed"
    assert result.version == EXPOSURE_ANALYZER_VERSION
    assert len(result.per_shot) == 2
    assert result.per_shot[0]["shotIndex"] == 0
    assert result.per_shot[1]["meanLuma"] > result.per_shot[0]["meanLuma"]
    assert "dominantColors" in result.per_shot[0]


@pytest.mark.unit
def test_analyze_exposure_fill_only_requires_prior_shots() -> None:
    media = _media(10)
    frames = {index: _solid_bgr(100, 100, 100) for index in range(10)}
    with pytest.raises(AnalyzeError) as exc:
        analyze_exposure(
            path=__file__,
            media=media,
            prior_facts={"subjects": []},
            fill_only=True,
            frame_provider=DictFrameProvider(frames),
        )
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_analyze_exposure_cancellation() -> None:
    media = _media(10)
    frames = {index: _solid_bgr(10, 10, 10) for index in range(10)}

    def cancel_check() -> None:
        raise AnalyzeError(CANCELLED, "Analysis cancelled")

    with pytest.raises(AnalyzeError) as exc:
        analyze_exposure(
            path=__file__,
            media=media,
            shots=[{"startFrame": 0, "endFrameExclusive": 10}],
            frame_provider=DictFrameProvider(frames),
            cancel_check=cancel_check,
        )
    assert exc.value.code == CANCELLED


@pytest.mark.unit
def test_exposure_policy_metadata_declares_formulas() -> None:
    meta = exposure_policy_metadata()
    assert meta["analyzerVersion"] == EXPOSURE_ANALYZER_VERSION
    assert meta["contrastFormula"] == CONTRAST_FORMULA
    assert meta["bt709Coefficients"] == BT709_LUMA_COEFFICIENTS
    assert meta["provisional"] is True


@pytest.mark.unit
def test_metrics_to_dict_matches_contract_keys() -> None:
    metrics = aggregate_shot_metrics(
        shot_index=3,
        luma_values=np.array([0.2, 0.5, 0.8], dtype=np.float64),
        saturation_values=[0.4],
        rgb_frames=[_solid_bgr(10, 20, 30)],
    )
    payload = metrics_to_dict(metrics)
    assert set(payload.keys()) == {
        "shotIndex",
        "meanLuma",
        "p05Luma",
        "p95Luma",
        "contrast",
        "saturation",
        "estimatedColorTemperatureK",
        "dominantColors",
    }
