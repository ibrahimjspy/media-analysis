from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import media_analysis.person_matte.encoding as encoding_module
from media_analysis.errors import CANCELLED, AnalyzeError
from media_analysis.frames import Rational
from media_analysis.person_matte.constants import DECODED_LUMA_TOLERANCE
from media_analysis.person_matte.encoding import (
    encode_matte_mp4,
    encoding_contract_dict,
    probe_matte_mp4,
    probe_video_frame_count,
    rational_framerate,
    stream_matte_mp4,
)


def _decode_first_luma(path: Path) -> int:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return completed.stdout[0]


def _frame_bytes(value: int, width: int = 64, height: int = 64) -> bytes:
    return np.full((height, width), value, dtype=np.uint8).tobytes()


@pytest.mark.unit
@pytest.mark.ffmpeg
@pytest.mark.parametrize("value", [0, 128, 255])
def test_encode_decode_luma_roundtrip_constant_patches(tmp_path: Path, value: int) -> None:
    dest = tmp_path / "matte.mp4"
    encode_matte_mp4(
        [_frame_bytes(value)],
        width=64,
        height=64,
        fps=Rational(30, 1),
        dest=dest,
    )
    decoded = _decode_first_luma(dest)
    assert abs(decoded - value) <= DECODED_LUMA_TOLERANCE


@pytest.mark.unit
def test_encoding_contract_documents_yuvj420p_honesty() -> None:
    contract = encoding_contract_dict()
    assert contract["pixelFormat"] == "yuv420p"
    assert contract["containerNotes"]["ffprobePixelFormatMayReport"] == "yuvj420p"
    assert contract["decodedLumaSemantics"]["decodedTolerance"] == DECODED_LUMA_TOLERANCE


@pytest.mark.unit
def test_rational_framerate_preserves_ntsc() -> None:
    assert rational_framerate(Rational(30000, 1001)) == "30000/1001"


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_streaming_encode_consumes_iterator_incrementally(tmp_path: Path) -> None:
    peak = 0
    current = 0

    def gray_frames():
        nonlocal peak, current
        for value in (0, 128, 255, 64):
            current += 1
            peak = max(peak, current)
            yield _frame_bytes(value)
            current -= 1
            peak = max(peak, current)

    dest = tmp_path / "streamed.mp4"
    count = stream_matte_mp4(
        gray_frames(),
        width=64,
        height=64,
        fps=Rational(30, 1),
        dest=dest,
    )
    assert count == 4
    assert peak == 1


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_ffprobe_contract_rational_fps_idr_gop_no_audio(tmp_path: Path) -> None:
    width, height = 64, 64
    frame_count = 30

    def frames():
        for _ in range(frame_count):
            yield _frame_bytes(128, width, height)

    dest = tmp_path / "matte.mp4"
    stream_matte_mp4(
        frames(),
        width=width,
        height=height,
        fps=Rational(30000, 1001),
        dest=dest,
    )
    payload = probe_matte_mp4(dest)
    assert not any(item.get("codec_type") == "audio" for item in payload["streams"])
    video = next(item for item in payload["streams"] if item.get("codec_type") == "video")
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] in {"yuv420p", "yuvj420p"}
    assert video.get("avg_frame_rate") == "30000/1001"
    assert int(video.get("has_b_frames", 0)) == 0
    assert probe_video_frame_count(dest) == frame_count

    frame_payload = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-read_intervals",
                "%+#1",
                "-show_entries",
                "frame=key_frame,pict_type",
                "-of",
                "json",
                str(dest),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    first = frame_payload["frames"][0]
    assert first["key_frame"] == 1
    assert first["pict_type"] == "I"


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_stream_encode_honours_cancel_check(tmp_path: Path) -> None:
    calls = {"n": 0}

    def frames():
        for _ in range(100):
            calls["n"] += 1
            if calls["n"] > 2:
                raise AnalyzeError(CANCELLED, "Analysis cancelled")
            yield _frame_bytes(128)

    with pytest.raises(AnalyzeError) as exc:
        stream_matte_mp4(
            frames(),
            width=64,
            height=64,
            fps=Rational(30, 1),
            dest=tmp_path / "cancelled.mp4",
        )
    assert exc.value.code == CANCELLED


@pytest.mark.unit
def test_stream_encode_terminates_ffmpeg_when_frame_generator_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStdin:
        def write(self, _data: bytes) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = FakeStdin()
            self.returncode = None
            self.terminated = False

        def poll(self):
            return None if not self.terminated else -15

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode or 0

    process = FakeProcess()
    monkeypatch.setattr(encoding_module.subprocess, "Popen", lambda *_args, **_kwargs: process)

    def broken_frames():
        yield _frame_bytes(128)
        raise RuntimeError("inference crashed")

    with pytest.raises(RuntimeError, match="inference crashed"):
        stream_matte_mp4(
            broken_frames(),
            width=64,
            height=64,
            fps=Rational(30, 1),
            dest=tmp_path / "failed.mp4",
        )
    assert process.terminated is True
