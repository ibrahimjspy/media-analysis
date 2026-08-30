"""Canonical mono 16 kHz analysis PCM extraction (zero-based timeline)."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from media_analysis.errors import DECODE_FAILED, TIMEOUT, AnalyzeError
from media_analysis.features.audio_subprocess import run_bounded_subprocess

PCM_EXTRACTION_VERSION = "pcm-mono16k-1.0.0"
ANALYSIS_SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class AnalysisPcm:
    """Mono float32 PCM for v1.1/v1.2 audio features."""

    samples: np.ndarray
    sample_rate: int
    duration_sec: float
    has_audio: bool
    source_duration_sec: float | None = None
    source_sample_rate: int | None = None
    source_channel_count: int | None = None


def pcm_duration_sec(sample_count: int, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if sample_count <= 0:
        return 0.0
    return sample_count / sample_rate


def empty_analysis_pcm(
    *,
    source_duration_sec: float | None = None,
    source_sample_rate: int | None = None,
    source_channel_count: int | None = None,
) -> AnalysisPcm:
    return AnalysisPcm(
        samples=np.zeros(0, dtype=np.float32),
        sample_rate=ANALYSIS_SAMPLE_RATE,
        duration_sec=0.0,
        has_audio=False,
        source_duration_sec=source_duration_sec,
        source_sample_rate=source_sample_rate,
        source_channel_count=source_channel_count,
    )


def extract_analysis_pcm(
    path: Path,
    *,
    has_audio: bool,
    source_duration_sec: float | None = None,
    source_sample_rate: int | None = None,
    source_channel_count: int | None = None,
    timeout_sec: float | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> AnalysisPcm:
    """Extract mono 16 kHz float32 PCM with a zero-based authoritative timeline."""
    if cancel_check:
        cancel_check()
    if not has_audio:
        return empty_analysis_pcm(
            source_duration_sec=source_duration_sec,
            source_sample_rate=source_sample_rate,
            source_channel_count=source_channel_count,
        )
    if shutil.which("ffmpeg") is None:
        raise AnalyzeError(DECODE_FAILED, "ffmpeg is not installed")
    if cancel_check:
        cancel_check()

    deadline = time.monotonic() + timeout_sec if timeout_sec is not None else None
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(ANALYSIS_SAMPLE_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    try:
        completed = run_bounded_subprocess(
            args,
            timeout_sec=_remaining(deadline),
            cancel_check=cancel_check,
            text_mode=False,
        )
    except AnalyzeError:
        raise
    except Exception as exc:
        raise AnalyzeError(DECODE_FAILED, "Could not decode source") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
        if _looks_like_no_audio(stderr):
            return empty_analysis_pcm(
                source_duration_sec=source_duration_sec,
                source_sample_rate=source_sample_rate,
                source_channel_count=source_channel_count,
            )
        raise AnalyzeError(DECODE_FAILED, "Could not decode source")

    if cancel_check:
        cancel_check()

    raw = completed.stdout
    if not raw:
        return empty_analysis_pcm(
            source_duration_sec=source_duration_sec,
            source_sample_rate=source_sample_rate,
            source_channel_count=source_channel_count,
        )

    samples = np.frombuffer(raw, dtype=np.float32)
    if samples.size == 0:
        return empty_analysis_pcm(
            source_duration_sec=source_duration_sec,
            source_sample_rate=source_sample_rate,
            source_channel_count=source_channel_count,
        )

    samples = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    duration = pcm_duration_sec(int(samples.size), ANALYSIS_SAMPLE_RATE)
    return AnalysisPcm(
        samples=samples,
        sample_rate=ANALYSIS_SAMPLE_RATE,
        duration_sec=duration,
        has_audio=True,
        source_duration_sec=source_duration_sec,
        source_sample_rate=source_sample_rate,
        source_channel_count=source_channel_count,
    )


def _looks_like_no_audio(stderr: str) -> bool:
    lowered = stderr.lower()
    markers = (
        "does not contain any stream",
        "output file #0 does not contain any stream",
        "no audio stream",
        "invalid argument",
    )
    return any(marker in lowered for marker in markers)


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AnalyzeError(TIMEOUT, "Analysis exceeded MEDIA_ANALYSIS_JOB_TIMEOUT_SEC")
    return remaining
