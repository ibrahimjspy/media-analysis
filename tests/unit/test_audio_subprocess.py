import subprocess

import pytest
from tests.unit.audio_fixtures import sine_wave, write_mono_wav

from media_analysis.errors import CANCELLED, DECODE_FAILED, AnalyzeError
from media_analysis.features.audio_dsp import parse_ebur128_output
from media_analysis.features.audio_pcm import extract_analysis_pcm
from media_analysis.features.audio_subprocess import run_bounded_subprocess

EBUR128_STDERR = """
[Parsed_ebur128_0 @ 0x600003d14000] Summary:

  Integrated loudness:
    I:         -23.0 LUFS
    Threshold: -33.1 LUFS

  True peak:
    Peak:        -1.2 dBFS
"""

EBUR128_SILENT_STDERR = """
[Parsed_ebur128_0 @ 0x600003d14000] Summary:

  Integrated loudness:
    I:          -inf LUFS
    Threshold: -70.0 LUFS

  True peak:
    Peak:         -inf dBFS
"""


@pytest.mark.unit
def test_parse_ebur128_realistic_multiline_output() -> None:
    result = parse_ebur128_output(EBUR128_STDERR)
    assert result.integrated_lufs == pytest.approx(-23.0)
    assert result.true_peak_db == pytest.approx(-1.2)


@pytest.mark.unit
def test_parse_ebur128_negative_infinity_maps_to_null() -> None:
    result = parse_ebur128_output(EBUR128_SILENT_STDERR)
    assert result.integrated_lufs is None
    assert result.true_peak_db is None


@pytest.mark.unit
def test_run_bounded_subprocess_cancel_terminates_child(monkeypatch) -> None:
    import subprocess

    calls = {"count": 0}

    class _FakeProc:
        def __init__(self) -> None:
            self._returncode: int | None = None

        def poll(self):
            return self._returncode

        def communicate(self, timeout=None):
            self._returncode = 0
            return ("", "")

        def kill(self) -> None:
            self._returncode = -9

        def wait(self) -> None:
            self._returncode = -9

    def fake_popen(*args, **kwargs):
        return _FakeProc()

    def cancel() -> None:
        calls["count"] += 1
        if calls["count"] >= 2:
            raise AnalyzeError(CANCELLED, "Analysis cancelled")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with pytest.raises(AnalyzeError) as exc:
        run_bounded_subprocess(["sleep", "5"], cancel_check=cancel, text_mode=True)
    assert exc.value.code == CANCELLED


@pytest.mark.unit
def test_extract_analysis_pcm_cancel_during_subprocess(tmp_path, monkeypatch) -> None:
    import subprocess

    from media_analysis.features import audio_pcm

    src = tmp_path / "tone.wav"
    write_mono_wav(src, sine_wave(duration_sec=0.1, frequency_hz=440.0))
    calls = {"count": 0}

    def cancel() -> None:
        calls["count"] += 1
        if calls["count"] >= 2:
            raise AnalyzeError(CANCELLED, "Analysis cancelled")

    def fake_run(*args, **kwargs):
        check = kwargs.get("cancel_check")
        if check:
            check()
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(audio_pcm, "run_bounded_subprocess", fake_run)

    with pytest.raises(AnalyzeError) as exc:
        extract_analysis_pcm(src, has_audio=True, cancel_check=cancel)
    assert exc.value.code == CANCELLED


@pytest.mark.unit
def test_measure_loudness_nonzero_exit_raises_decode_failed(monkeypatch, tmp_path) -> None:
    from media_analysis.features import audio_dsp

    tmp_path.joinpath("tone.wav").write_bytes(b"x")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(audio_dsp, "run_bounded_subprocess", fake_run)

    with pytest.raises(AnalyzeError) as exc:
        audio_dsp.measure_loudness_ffmpeg(tmp_path / "tone.wav")
    assert exc.value.code == DECODE_FAILED
