import shutil

import numpy as np
import pytest
from tests.unit.audio_fixtures import pulse_train, silence, sine_wave, write_mono_wav

from media_analysis.errors import CANCELLED, FEATURE_UNAVAILABLE, AnalyzeError
from media_analysis.features.audio import (
    AUDIO_ABSENT_WARNING,
    analyze_audio,
    empty_audio_analysis,
)
from media_analysis.features.audio_dsp import RMS_HOP_SEC, compute_onsets, compute_rms_series
from media_analysis.features.audio_pcm import AnalysisPcm, extract_analysis_pcm
from media_analysis.features.silero_vad import FakeSileroVadSession
from media_analysis.frames import Rational


@pytest.mark.unit
def test_empty_audio_analysis_completed_shape() -> None:
    result = empty_audio_analysis()
    payload = result.to_payload()
    assert result.status == "completed"
    assert payload["durationSec"] == 0.0
    assert payload["rms"]["hopSec"] == RMS_HOP_SEC
    assert payload["rms"]["values"] == []
    assert payload["onsets"] == []
    assert payload["speech"] == []
    assert payload["integratedLufs"] is None
    assert payload["truePeakDb"] is None
    assert payload["bpm"] is None
    assert AUDIO_ABSENT_WARNING in result.warning_codes


@pytest.mark.unit
def test_rms_window_boundaries_cover_full_duration() -> None:
    sample_rate = 16_000
    samples = np.ones(sample_rate, dtype=np.float32) * 0.25
    hop_sec, values = compute_rms_series(samples, sample_rate=sample_rate, hop_sec=0.05)
    assert hop_sec == pytest.approx(0.05)
    assert len(values) == 20
    assert values[0] == pytest.approx(0.25)
    assert values[-1] == pytest.approx(0.25)


@pytest.mark.unit
def test_onset_timing_on_pulse_train_is_deterministic() -> None:
    samples = pulse_train(duration_sec=1.0, pulse_sec=0.02, gap_sec=0.18, sample_rate=16_000)
    first = compute_onsets(samples, sample_rate=16_000, fps=Rational(30, 1))
    second = compute_onsets(samples, sample_rate=16_000, fps=Rational(30, 1))
    assert first == second
    assert first
    assert all(0.0 <= float(item["sec"]) <= 1.0 for item in first)
    assert all("strength" in item for item in first)
    assert all("sourceFrameApprox" in item for item in first)


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_analyze_audio_sine_completed_with_metrics(tmp_path) -> None:
    src = tmp_path / "tone.wav"
    write_mono_wav(src, sine_wave(duration_sec=2.0, frequency_hz=120.0))
    pcm = extract_analysis_pcm(src, has_audio=True)
    session = FakeSileroVadSession(constant=0.05)
    result = analyze_audio(
        pcm,
        source_path=src,
        fps=Rational(30, 1),
        vad_session=session,
    )
    payload = result.to_payload()
    assert result.status == "completed"
    assert payload["durationSec"] == pytest.approx(2.0, abs=0.05)
    assert payload["rms"]["values"]
    assert payload["speech"] == []
    assert payload["bpm"] is None
    assert AUDIO_ABSENT_WARNING not in result.warning_codes


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_analyze_audio_pulse_bpm_detected_or_null(tmp_path) -> None:
    src = tmp_path / "pulse.wav"
    write_mono_wav(
        src,
        pulse_train(duration_sec=4.0, pulse_sec=0.02, gap_sec=0.23, sample_rate=16_000),
    )
    pcm = extract_analysis_pcm(src, has_audio=True)
    session = FakeSileroVadSession(constant=0.05)
    result = analyze_audio(pcm, fps=Rational(30, 1), vad_session=session)
    bpm = result.to_payload()["bpm"]
    assert bpm is None or 60.0 <= float(bpm) <= 200.0


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_analyze_audio_silent_track_is_measured_without_audio_absent(tmp_path) -> None:
    src = tmp_path / "silent.wav"
    write_mono_wav(src, silence(0.5))
    pcm = extract_analysis_pcm(src, has_audio=True)
    session = FakeSileroVadSession(constant=0.05)
    result = analyze_audio(
        pcm,
        source_path=src,
        vad_session=session,
        require_vad=False,
    )
    payload = result.to_payload()
    assert result.status == "completed"
    assert AUDIO_ABSENT_WARNING not in result.warning_codes
    assert payload["durationSec"] == pytest.approx(0.5, abs=0.05)
    assert payload["rms"]["values"]
    assert all(value == pytest.approx(0.0) for value in payload["rms"]["values"])
    assert payload["bpm"] is None


@pytest.mark.unit
def test_analyze_audio_missing_vad_fails_by_default() -> None:
    pcm = AnalysisPcm(
        samples=np.ones(16_000, dtype=np.float32) * 0.1,
        sample_rate=16_000,
        duration_sec=1.0,
        has_audio=True,
    )
    with pytest.raises(AnalyzeError) as exc:
        analyze_audio(pcm)
    assert exc.value.code == FEATURE_UNAVAILABLE


@pytest.mark.unit
def test_analyze_audio_no_stream_is_completed_empty() -> None:
    pcm = AnalysisPcm(
        samples=np.zeros(0, dtype=np.float32),
        sample_rate=16_000,
        duration_sec=0.0,
        has_audio=False,
        source_duration_sec=5.0,
    )
    result = analyze_audio(pcm, require_vad=False)
    assert result.status == "completed"
    assert result.duration_sec == 0.0
    assert AUDIO_ABSENT_WARNING in result.warning_codes


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_analyze_audio_vad_speech_segments(tmp_path) -> None:
    src = tmp_path / "tone.wav"
    write_mono_wav(src, sine_wave(duration_sec=0.5, frequency_hz=300.0))
    pcm = extract_analysis_pcm(src, has_audio=True)
    probs = [0.1] * 4 + [0.95] * 6 + [0.05] * 4
    session = FakeSileroVadSession(probabilities=probs)
    result = analyze_audio(pcm, vad_session=session)
    speech = result.to_payload()["speech"]
    assert speech
    assert speech[0]["scoreType"] == "raw_model"
    assert speech[0]["endSecExclusive"] > speech[0]["startSec"]
    assert speech[-1]["endSecExclusive"] <= pcm.duration_sec + 1e-6


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_analyze_audio_loudness_from_ffmpeg_when_source_path_provided(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    src = tmp_path / "tone.wav"
    write_mono_wav(src, sine_wave(duration_sec=0.75, frequency_hz=440.0, amplitude=0.2))
    pcm = extract_analysis_pcm(src, has_audio=True)
    session = FakeSileroVadSession(constant=0.05)
    result = analyze_audio(pcm, source_path=src, vad_session=session)
    payload = result.to_payload()
    assert payload["integratedLufs"] is None or isinstance(payload["integratedLufs"], float)
    assert payload["truePeakDb"] is None or isinstance(payload["truePeakDb"], float)


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_analyze_audio_cancel_during_loudness(tmp_path) -> None:
    src = tmp_path / "tone.wav"
    write_mono_wav(src, sine_wave(duration_sec=0.5, frequency_hz=440.0))
    pcm = extract_analysis_pcm(src, has_audio=True)
    session = FakeSileroVadSession(constant=0.05)

    def cancel() -> None:
        raise AnalyzeError(CANCELLED, "Analysis cancelled")

    with pytest.raises(AnalyzeError) as exc:
        analyze_audio(pcm, source_path=src, vad_session=session, cancel_check=cancel)
    assert exc.value.code == CANCELLED


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_analyze_audio_deterministic_output(tmp_path) -> None:
    src = tmp_path / "tone.wav"
    write_mono_wav(src, sine_wave(duration_sec=1.0, frequency_hz=220.0))
    pcm = extract_analysis_pcm(src, has_audio=True)
    session = FakeSileroVadSession(probabilities=[0.2, 0.8, 0.85, 0.1])
    first = analyze_audio(pcm, fps=Rational(24, 1), vad_session=session).to_payload()
    second = analyze_audio(pcm, fps=Rational(24, 1), vad_session=session).to_payload()
    assert first == second
