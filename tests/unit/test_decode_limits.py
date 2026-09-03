import subprocess

import pytest

from media_analysis import decode
from media_analysis.config import Settings
from media_analysis.decode import ProbedMedia, decoded_pixel_count, enforce_limits
from media_analysis.errors import LIMIT_EXCEEDED, TIMEOUT, AnalyzeError
from media_analysis.frames import Rational


def _media(*, width=320, height=240, frames=30, duration=1.0) -> ProbedMedia:
    return ProbedMedia(
        width=width,
        height=height,
        fps=Rational(30, 1),
        frame_count=frames,
        duration=duration,
        codec="h264",
        has_audio=False,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channel_count=None,
    )


@pytest.mark.unit
def test_duration_limit() -> None:
    settings = Settings(media_analysis_max_duration_sec=60)
    with pytest.raises(AnalyzeError) as exc:
        enforce_limits(_media(duration=90), settings)
    assert exc.value.code == LIMIT_EXCEEDED


@pytest.mark.unit
def test_dimension_limit() -> None:
    settings = Settings(media_analysis_max_width=1920, media_analysis_max_height=1920)
    with pytest.raises(AnalyzeError) as exc:
        enforce_limits(_media(width=3840, height=2160), settings)
    assert exc.value.code == LIMIT_EXCEEDED


@pytest.mark.unit
def test_decoded_pixel_limit() -> None:
    settings = Settings(media_analysis_max_decoded_pixels=320 * 240 * 10)
    media = _media(width=320, height=240, frames=30)
    assert decoded_pixel_count(media) == 320 * 240 * 30
    with pytest.raises(AnalyzeError) as exc:
        enforce_limits(media, settings)
    assert exc.value.code == LIMIT_EXCEEDED
    assert "DECODED_PIXELS" in exc.value.message


@pytest.mark.unit
def test_decoded_pixel_limit_allows_current_envelope() -> None:
    settings = Settings()
    enforce_limits(_media(width=1920, height=1920, frames=1800, duration=60), settings)


@pytest.mark.unit
def test_decode_subprocess_timeout_maps_to_stable_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=0.01)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(AnalyzeError) as exc:
        decode._run(["ffprobe"], timeout_sec=0.01)
    assert exc.value.code == TIMEOUT
