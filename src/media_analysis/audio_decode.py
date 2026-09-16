"""Standalone audio probe; source playback and analysis PCM stay distinct."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from media_analysis.config import Settings
from media_analysis.errors import DECODE_FAILED, LIMIT_EXCEEDED, AnalyzeError
from media_analysis.features.audio_subprocess import run_bounded_subprocess

AUDIO_DECODE_VERSION = "audio-ffmpeg-decoder-trim-zero-origin-v1"
AUDIO_CODECS = {
    "pcm_s16le",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_f32le",
    "flac",
    "mp3",
    "aac",
    "opus",
    "vorbis",
    "alac",
}


@dataclass(frozen=True)
class AudioMedia:
    duration: float
    audio_codec: str
    audio_sample_rate: int
    audio_channel_count: int
    start_time: float
    has_audio: bool = True


def probe_audio(path: Path, settings: Settings, stage) -> AudioMedia:
    try:
        completed = run_bounded_subprocess(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            timeout_sec=stage.remaining(),
            cancel_check=stage,
            text_mode=True,
        )
        if completed.returncode:
            raise ValueError("probe failed")
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        # Attached album art is not a video timeline.
        if any(
            s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
            for s in streams
        ):
            raise ValueError("audio kind contains video")
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        if len(audio) != 1:
            raise ValueError("exactly one audio stream required")
        stream = audio[0]
        duration = float(stream.get("duration") or payload.get("format", {}).get("duration"))
        rate, channels = int(stream["sample_rate"]), int(stream["channels"])
        start = float(stream.get("start_time", 0))
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(start):
            raise ValueError("invalid audio duration/clock")
        if rate <= 0 or channels <= 0 or stream["codec_name"] not in AUDIO_CODECS:
            raise ValueError("unsupported audio stream")
        if (
            duration > settings.media_analysis_max_audio_duration_sec
            or channels > settings.media_analysis_max_audio_channels
            or rate > settings.media_analysis_max_audio_sample_rate
        ):
            raise AnalyzeError(LIMIT_EXCEEDED, "Audio exceeds duration/channel/sample-rate bounds")
        return AudioMedia(duration, stream["codec_name"], rate, channels, start)
    except AnalyzeError:
        raise
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise AnalyzeError(DECODE_FAILED, "Could not probe supported standalone audio") from exc


def canonical_audio(path: Path, dest: Path, media: AudioMedia, settings: Settings, stage) -> None:
    estimated_bytes = (
        media.duration * media.audio_sample_rate * media.audio_channel_count * 2 + 4096
    )
    if estimated_bytes > settings.media_analysis_max_bytes:
        raise AnalyzeError(LIMIT_EXCEEDED, "Canonical audio exceeds artifact byte limit")
    completed = run_bounded_subprocess(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-t",
            str(settings.media_analysis_max_audio_duration_sec + 0.1),
            "-fs",
            str(settings.media_analysis_max_bytes + 1),
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(media.audio_sample_rate),
            "-ac",
            str(media.audio_channel_count),
            "-map_metadata",
            "-1",
            str(dest),
        ],
        timeout_sec=stage.remaining(),
        cancel_check=stage,
    )
    if completed.returncode:
        raise AnalyzeError(DECODE_FAILED, "Could not canonicalize audio")
    if dest.stat().st_size > settings.media_analysis_max_bytes:
        raise AnalyzeError(LIMIT_EXCEEDED, "Canonical audio exceeds artifact byte limit")
