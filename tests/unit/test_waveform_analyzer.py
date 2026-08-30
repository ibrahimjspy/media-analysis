import numpy as np
import pytest
from tests.unit.audio_fixtures import pulse_train, silence, sine_wave, write_mono_wav

from media_analysis.features.audio import AUDIO_ABSENT_WARNING
from media_analysis.features.audio_pcm import AnalysisPcm, extract_analysis_pcm
from media_analysis.features.waveform import (
    WAVEFORM_HOP_SEC,
    analyze_waveform,
    compute_waveform_arrays,
)


@pytest.mark.unit
def test_waveform_empty_on_no_audio_pcm() -> None:
    pcm = AnalysisPcm(
        samples=np.zeros(0, dtype=np.float32),
        sample_rate=16_000,
        duration_sec=0.0,
        has_audio=False,
    )
    payload, warnings = analyze_waveform(pcm)
    assert payload == {"hopSec": WAVEFORM_HOP_SEC, "min": [], "max": []}
    assert warnings == [AUDIO_ABSENT_WARNING]


@pytest.mark.unit
def test_waveform_extrema_on_generated_pcm() -> None:
    samples = pulse_train(duration_sec=0.2, pulse_sec=0.05, gap_sec=0.05, amplitude=0.75)
    hop_sec, mins, maxs = compute_waveform_arrays(samples, sample_rate=16_000)
    assert hop_sec == pytest.approx(WAVEFORM_HOP_SEC)
    assert len(mins) == len(maxs) > 0
    assert min(mins) <= 0.0
    assert max(maxs) >= 0.7


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_waveform_matches_shared_pcm_path(tmp_path) -> None:
    src = tmp_path / "tone.wav"
    write_mono_wav(src, sine_wave(duration_sec=0.3, frequency_hz=440.0, amplitude=0.6))
    pcm = extract_analysis_pcm(src, has_audio=True)
    payload, warnings = analyze_waveform(pcm)
    assert not warnings
    assert payload["hopSec"] == pytest.approx(WAVEFORM_HOP_SEC)
    assert len(payload["min"]) == len(payload["max"]) > 0
    assert min(payload["min"]) == pytest.approx(-0.6, abs=0.01)
    assert max(payload["max"]) == pytest.approx(0.6, abs=0.01)


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_waveform_silent_track_is_measured_without_audio_absent(tmp_path) -> None:
    src = tmp_path / "silent.wav"
    write_mono_wav(src, silence(0.25))
    pcm = extract_analysis_pcm(src, has_audio=True)
    payload, warnings = analyze_waveform(pcm)
    assert AUDIO_ABSENT_WARNING not in warnings
    assert payload["min"]
    assert payload["max"]
    assert all(value == pytest.approx(0.0) for value in payload["min"])
    assert all(value == pytest.approx(0.0) for value in payload["max"])


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_waveform_deterministic_output(tmp_path) -> None:
    src = tmp_path / "tone.wav"
    write_mono_wav(src, sine_wave(duration_sec=0.4, frequency_hz=330.0))
    pcm = extract_analysis_pcm(src, has_audio=True)
    first, _ = analyze_waveform(pcm)
    second, _ = analyze_waveform(pcm)
    assert first == second
