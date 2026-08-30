import subprocess
from pathlib import Path

import pytest

from media_analysis.decode import (
    canonicalize,
    needs_audio_pipeline,
    needs_canonical_audio,
    probe,
)


def make_audio_mp4(path: Path, *, seconds: float = 0.5, fps: int = 30) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s=320x240:d={seconds}:r={fps}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.mark.unit
def test_needs_audio_pipeline_helpers() -> None:
    assert needs_audio_pipeline(["audio"])
    assert needs_audio_pipeline(["waveform"])
    assert not needs_audio_pipeline(["shots"])
    assert needs_canonical_audio(canonicalize=True, features=["shots"])
    assert needs_canonical_audio(canonicalize=False, features=["audio"])
    assert not needs_canonical_audio(canonicalize=False, features=["motion"])


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_canonicalize_retains_audio_from_generated_fixture(tmp_path: Path) -> None:
    source = make_audio_mp4(tmp_path / "source.mp4")
    dest = tmp_path / "canonical.mp4"
    source_probe = probe(source)
    assert source_probe.has_audio

    media = canonicalize(source, dest, preserve_audio=True)
    assert media.has_audio
    assert media.audio_codec == "aac"
    assert media.audio_sample_rate == 48000
    assert media.frame_count >= 10
    assert media.fps.numerator == 30


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_canonicalize_without_audio_strips_audio(tmp_path: Path) -> None:
    source = make_audio_mp4(tmp_path / "source.mp4")
    dest = tmp_path / "silent.mp4"
    media = canonicalize(source, dest, preserve_audio=False)
    assert not media.has_audio


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_canonicalize_timeout_maps_to_stable_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_audio_mp4(tmp_path / "source.mp4")
    dest = tmp_path / "canonical.mp4"

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=0.01)

    monkeypatch.setattr(subprocess, "run", timeout)
    from media_analysis.errors import TIMEOUT, AnalyzeError

    with pytest.raises(AnalyzeError) as exc:
        canonicalize(source, dest, timeout_sec=0.01)
    assert exc.value.code == TIMEOUT
