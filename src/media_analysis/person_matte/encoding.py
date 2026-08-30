"""H.264 decoded-luma matte encoding contract."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from media_analysis.errors import DECODE_FAILED, TIMEOUT, AnalyzeError
from media_analysis.frames import Rational
from media_analysis.person_matte.constants import (
    DECODED_LUMA_TOLERANCE,
    MATTE_ENCODING_RECIPE,
    MAX_GOP_FRAMES,
)


def encoding_contract_dict() -> dict[str, Any]:
    return {
        "recipe": MATTE_ENCODING_RECIPE,
        "mimeType": "video/mp4",
        "codec": "h264",
        "pixelFormat": "yuv420p",
        "colorMatrix": "bt709",
        "lumaRange": "full",
        "alphaChannel": "decoded_luma",
        "firstFrame": "idr",
        "closedGop": True,
        "maxGopFrames": MAX_GOP_FRAMES,
        "bFrames": 0,
        "audio": False,
        "decodedLumaSemantics": {
            "opaqueSubject": 255,
            "background": 0,
            "softEdges": "intermediate values preserved within tolerance",
            "decodedTolerance": DECODED_LUMA_TOLERANCE,
        },
        "containerNotes": {
            "ffprobePixelFormatMayReport": "yuvj420p",
            "contractPixelFormat": "yuv420p",
            "reason": (
                "Full-range gray H.264 is often labeled yuvj420p by ffprobe; "
                "correctness is defined at decoded luma, not container string."
            ),
        },
    }


def rational_framerate(fps: Rational) -> str:
    if fps.denominator == 0:
        raise AnalyzeError(DECODE_FAILED, "fps denominator is zero")
    return f"{fps.numerator}/{fps.denominator}"


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AnalyzeError(TIMEOUT, "Matte encode exceeded deadline")
    return remaining


def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)


def stream_matte_mp4(
    gray_frames: Iterable[bytes],
    *,
    width: int,
    height: int,
    fps: Rational,
    dest: Path,
    timeout_sec: float | None = None,
    deadline: float | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> int:
    """Stream grayscale frames to ffmpeg stdin; returns encoded frame count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = width * height
    fps_arg = rational_framerate(fps)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-framerate",
        fps_arg,
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "scale=in_range=full:out_range=full",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-color_range",
        "pc",
        "-g",
        str(MAX_GOP_FRAMES),
        "-keyint_min",
        str(MAX_GOP_FRAMES),
        "-sc_threshold",
        "0",
        "-bf",
        "0",
        "-x264-params",
        "open-gop=0",
        "-bsf:v",
        "h264_metadata=video_full_range_flag=1",
        str(dest),
    ]
    frame_count = 0
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise AnalyzeError(DECODE_FAILED, "ffmpeg is not installed") from exc
    assert proc.stdin is not None
    try:
        for index, frame in enumerate(gray_frames):
            if cancel_check is not None:
                cancel_check()
            if deadline is not None and time.monotonic() >= deadline:
                raise AnalyzeError(TIMEOUT, "Matte encode exceeded deadline")
            if len(frame) != expected_bytes:
                raise AnalyzeError(
                    DECODE_FAILED,
                    f"Matte frame {index} byte length mismatch",
                )
            proc.stdin.write(frame)
            frame_count += 1
        proc.stdin.close()
        proc.stdin = None
        remaining = _remaining(deadline) if deadline is not None else timeout_sec
        _, stderr = proc.communicate(timeout=remaining)
        if proc.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")[:240]
            raise AnalyzeError(DECODE_FAILED, f"Could not encode matte MP4: {detail}")
    except AnalyzeError:
        _terminate_process(proc)
        raise
    except subprocess.TimeoutExpired as exc:
        _terminate_process(proc)
        raise AnalyzeError(TIMEOUT, "Matte encode exceeded deadline") from exc
    except OSError as exc:
        _terminate_process(proc)
        raise AnalyzeError(DECODE_FAILED, "Could not encode matte MP4") from exc
    except BaseException:
        _terminate_process(proc)
        raise
    if frame_count == 0:
        raise AnalyzeError(DECODE_FAILED, "No matte frames to encode")
    return frame_count


def encode_matte_mp4(
    frames: Iterable[bytes],
    *,
    width: int,
    height: int,
    fps: Rational,
    dest: Path,
    timeout_sec: float | None = None,
    deadline: float | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> None:
    """Encode grayscale matte frames to contract H.264 MP4 (decoded luma = alpha)."""
    stream_matte_mp4(
        frames,
        width=width,
        height=height,
        fps=fps,
        dest=dest,
        timeout_sec=timeout_sec,
        deadline=deadline,
        cancel_check=cancel_check,
    )


def probe_matte_mp4(path: Path) -> dict[str, Any]:
    """Return ffprobe stream metadata for contract tests."""
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise AnalyzeError(DECODE_FAILED, "ffprobe is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise AnalyzeError(DECODE_FAILED, "Could not probe matte MP4") from exc
    return json.loads(completed.stdout)


def probe_video_frame_count(path: Path) -> int:
    payload = probe_matte_mp4(path)
    video = next(item for item in payload["streams"] if item.get("codec_type") == "video")
    raw = video.get("nb_frames")
    if raw is not None and str(raw).isdigit():
        return int(raw)
    counted = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(counted.stdout.strip())
