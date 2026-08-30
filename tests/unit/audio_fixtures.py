"""Shared PCM fixtures for audio/waveform unit tests."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np


def write_mono_wav(path: Path, samples: np.ndarray, *, sample_rate: int = 16_000) -> None:
    clipped = np.clip(samples.astype(np.float64), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


def write_silent_video_container(path: Path) -> None:
    """Minimal container without audio stream (video-only stub bytes)."""
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")


def sine_wave(
    *,
    duration_sec: float,
    frequency_hz: float,
    sample_rate: int = 16_000,
    amplitude: float = 0.5,
) -> np.ndarray:
    count = int(round(duration_sec * sample_rate))
    times = np.arange(count, dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2.0 * math.pi * frequency_hz * times)).astype(np.float32)


def pulse_train(
    *,
    duration_sec: float,
    pulse_sec: float,
    gap_sec: float,
    sample_rate: int = 16_000,
    amplitude: float = 0.8,
) -> np.ndarray:
    count = int(round(duration_sec * sample_rate))
    out = np.zeros(count, dtype=np.float32)
    cursor = 0
    pulse_samples = int(round(pulse_sec * sample_rate))
    gap_samples = int(round(gap_sec * sample_rate))
    while cursor < count:
        end = min(cursor + pulse_samples, count)
        out[cursor:end] = amplitude
        cursor += pulse_samples + gap_samples
    return out


def silence(duration_sec: float, *, sample_rate: int = 16_000) -> np.ndarray:
    return np.zeros(int(round(duration_sec * sample_rate)), dtype=np.float32)
