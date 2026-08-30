import shutil
import subprocess

import numpy as np
import pytest
from tests.unit.audio_fixtures import pulse_train, silence, sine_wave, write_mono_wav

from media_analysis.errors import CANCELLED, AnalyzeError
from media_analysis.features.audio_pcm import (
    ANALYSIS_SAMPLE_RATE,
    empty_analysis_pcm,
    extract_analysis_pcm,
    pcm_duration_sec,
)


@pytest.mark.unit
def test_empty_analysis_pcm_preserves_source_facts() -> None:
    pcm = empty_analysis_pcm(
        source_duration_sec=12.5,
        source_sample_rate=48_000,
        source_channel_count=2,
    )
    assert pcm.has_audio is False
    assert pcm.duration_sec == 0.0
    assert pcm.source_duration_sec == 12.5
    assert pcm.source_sample_rate == 48_000
    assert pcm.source_channel_count == 2


@pytest.mark.unit
def test_pcm_duration_sec_zero_based() -> None:
    assert pcm_duration_sec(16_000, ANALYSIS_SAMPLE_RATE) == pytest.approx(1.0)
    assert pcm_duration_sec(0, ANALYSIS_SAMPLE_RATE) == 0.0


@pytest.mark.unit
def test_extract_analysis_pcm_skips_ffmpeg_when_has_audio_false(tmp_path) -> None:
    src = tmp_path / "video.mp4"
    src.write_bytes(b"not-real")
    pcm = extract_analysis_pcm(src, has_audio=False, source_duration_sec=3.0)
    assert pcm.has_audio is False
    assert pcm.samples.size == 0
    assert pcm.source_duration_sec == 3.0


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_extract_analysis_pcm_mono16k_from_wav(tmp_path) -> None:
    src = tmp_path / "tone.wav"
    samples = sine_wave(duration_sec=0.5, frequency_hz=440.0)
    write_mono_wav(src, samples, sample_rate=16_000)
    pcm = extract_analysis_pcm(
        src,
        has_audio=True,
        source_duration_sec=0.5,
        source_sample_rate=16_000,
        source_channel_count=1,
    )
    assert pcm.has_audio is True
    assert pcm.sample_rate == ANALYSIS_SAMPLE_RATE
    assert pcm.duration_sec == pytest.approx(0.5, abs=0.02)
    assert pcm.samples.size > 0
    assert float(np.max(np.abs(pcm.samples))) > 0.1


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_extract_analysis_pcm_resamples_non16k_source(tmp_path) -> None:
    src = tmp_path / "tone.wav"
    samples = sine_wave(duration_sec=0.25, frequency_hz=220.0, sample_rate=48_000)
    write_mono_wav(src, samples, sample_rate=48_000)
    pcm = extract_analysis_pcm(src, has_audio=True, source_sample_rate=48_000)
    assert pcm.duration_sec == pytest.approx(0.25, abs=0.03)
    assert pcm.sample_rate == ANALYSIS_SAMPLE_RATE


@pytest.mark.unit
def test_extract_analysis_pcm_cancel_before_start(tmp_path) -> None:
    src = tmp_path / "tone.wav"
    write_mono_wav(src, sine_wave(duration_sec=0.1, frequency_hz=440.0))

    def cancel() -> None:
        raise AnalyzeError(CANCELLED, "Analysis cancelled")

    with pytest.raises(AnalyzeError) as exc:
        extract_analysis_pcm(src, has_audio=True, cancel_check=cancel)
    assert exc.value.code == CANCELLED


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_extract_analysis_pcm_silence_is_present_not_absent(tmp_path) -> None:
    src = tmp_path / "silent.wav"
    write_mono_wav(src, silence(0.4))
    pcm = extract_analysis_pcm(src, has_audio=True)
    assert pcm.has_audio is True
    assert pcm.duration_sec == pytest.approx(0.4, abs=0.03)
    assert float(np.max(np.abs(pcm.samples))) == 0.0


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_extract_analysis_pcm_pulse_duration(tmp_path) -> None:
    src = tmp_path / "pulse.wav"
    write_mono_wav(src, pulse_train(duration_sec=1.0, pulse_sec=0.05, gap_sec=0.15))
    pcm = extract_analysis_pcm(src, has_audio=True)
    assert pcm.duration_sec == pytest.approx(1.0, abs=0.03)


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_extract_analysis_pcm_missing_ffmpeg_raises(monkeypatch, tmp_path) -> None:
    src = tmp_path / "tone.wav"
    write_mono_wav(src, sine_wave(duration_sec=0.1, frequency_hz=440.0))
    monkeypatch.setattr("media_analysis.features.audio_pcm.shutil.which", lambda _: None)
    with pytest.raises(AnalyzeError) as exc:
        extract_analysis_pcm(src, has_audio=True)
    assert exc.value.code == "DECODE_FAILED"


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_extract_analysis_pcm_video_without_audio_returns_empty(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    src = tmp_path / "video_only.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=0.2",
            "-an",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    pcm = extract_analysis_pcm(src, has_audio=True, source_duration_sec=0.2)
    assert pcm.has_audio is False
    assert pcm.samples.size == 0
