from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from media_analysis.config import Settings
from media_analysis.errors import DECODE_FAILED, LIMIT_EXCEEDED, TIMEOUT, AnalyzeError
from media_analysis.frames import Rational, duration_sec

DECODE_PIPELINE_VERSION = "decode-cfr-1.1.0"
AUDIO_FEATURES = frozenset({"audio", "waveform"})


@dataclass(frozen=True, slots=True)
class ProbedMedia:
    width: int
    height: int
    fps: Rational
    frame_count: int
    duration: float
    codec: str
    has_audio: bool
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channel_count: int | None


def _run(args: list[str], *, timeout_sec: float | None = None) -> str:
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError as exc:
        raise AnalyzeError(DECODE_FAILED, "ffprobe/ffmpeg is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise AnalyzeError(DECODE_FAILED, "Could not decode source") from exc
    except subprocess.TimeoutExpired as exc:
        raise AnalyzeError(TIMEOUT, "Analysis exceeded MEDIA_ANALYSIS_JOB_TIMEOUT_SEC") from exc
    return completed.stdout


def _parse_rate(raw: str) -> Rational:
    if not raw or raw in {"0/0", "N/A"}:
        return Rational(30, 1)
    if "/" in raw:
        num, den = raw.split("/", 1)
        return Rational(int(num), int(den) or 1)
    return Rational(int(round(float(raw) * 1000)), 1000)


def probe(path: Path, *, timeout_sec: float | None = None) -> ProbedMedia:
    deadline = time.monotonic() + timeout_sec if timeout_sec is not None else None
    payload = json.loads(
        _run(
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
            timeout_sec=_remaining(deadline),
        )
    )
    streams = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise AnalyzeError(DECODE_FAILED, "Could not decode source")
    fps = _parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1")
    raw_frames = video.get("nb_frames")
    if raw_frames and raw_frames != "N/A":
        frame_count = int(raw_frames)
    else:
        frame_count = _count_frames(path, timeout_sec=_remaining(deadline))
    duration = float(payload.get("format", {}).get("duration") or duration_sec(frame_count, fps))
    return ProbedMedia(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        codec=video.get("codec_name") or "unknown",
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        audio_sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        audio_channel_count=int(audio["channels"]) if audio and audio.get("channels") else None,
    )


def _count_frames(path: Path, *, timeout_sec: float | None = None) -> int:
    text = _run(
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
            "csv=p=0",
            str(path),
        ],
        timeout_sec=timeout_sec,
    ).strip()
    if not text or text == "N/A":
        raise AnalyzeError(DECODE_FAILED, "Could not decode source")
    return int(text)


def decoded_pixel_count(media: ProbedMedia) -> int:
    return media.width * media.height * media.frame_count


def enforce_limits(media: ProbedMedia, settings: Settings) -> None:
    if media.duration > settings.media_analysis_max_duration_sec + 0.05:
        raise AnalyzeError(
            LIMIT_EXCEEDED,
            "Source duration exceeds MEDIA_ANALYSIS_MAX_DURATION_SEC",
        )
    if (
        media.width > settings.media_analysis_max_width
        or media.height > settings.media_analysis_max_height
    ):
        raise AnalyzeError(LIMIT_EXCEEDED, "Source dimensions exceed configured maximum")
    if decoded_pixel_count(media) > settings.media_analysis_max_decoded_pixels:
        raise AnalyzeError(
            LIMIT_EXCEEDED,
            "Source decoded pixels exceed MEDIA_ANALYSIS_MAX_DECODED_PIXELS",
        )


def needs_audio_pipeline(features: list[str]) -> bool:
    return bool(set(features) & AUDIO_FEATURES)


def needs_canonical_audio(*, canonicalize: bool, features: list[str]) -> bool:
    return canonicalize or needs_audio_pipeline(features)


def canonicalize(
    src: Path,
    dest: Path,
    *,
    timeout_sec: float | None = None,
    preserve_audio: bool = True,
) -> ProbedMedia:
    """CFR H.264 proxy. Used only when canonicalize=true."""
    deadline = time.monotonic() + timeout_sec if timeout_sec is not None else None
    if shutil.which("ffmpeg") is None:
        raise AnalyzeError(DECODE_FAILED, "ffmpeg is not installed")
    source = probe(src, timeout_sec=_remaining(deadline))
    include_audio = preserve_audio and source.has_audio
    args: list[str] = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
    ]
    if include_audio:
        args.extend(["-map", "0:a:0"])
    args.extend(
        [
            "-vf",
            "fps=fps=30:round=near",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
        ]
    )
    if include_audio:
        args.extend(["-c:a", "aac", "-ar", "48000", "-b:a", "192k"])
    else:
        args.append("-an")
    args.append(str(dest))
    try:
        subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=_remaining(deadline),
        )
    except subprocess.CalledProcessError as exc:
        raise AnalyzeError(DECODE_FAILED, "Could not decode source") from exc
    except subprocess.TimeoutExpired as exc:
        raise AnalyzeError(TIMEOUT, "Analysis exceeded MEDIA_ANALYSIS_JOB_TIMEOUT_SEC") from exc
    return probe(dest, timeout_sec=_remaining(deadline))


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AnalyzeError(TIMEOUT, "Analysis exceeded MEDIA_ANALYSIS_JOB_TIMEOUT_SEC")
    return remaining
