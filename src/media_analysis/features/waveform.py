"""v1.2 waveform analyzer — min/max arrays from shared analysis PCM."""

from __future__ import annotations

import numpy as np

from media_analysis.features.audio import AUDIO_ABSENT_WARNING
from media_analysis.features.audio_pcm import AnalysisPcm

WAVEFORM_ANALYZER_VERSION = "waveform-v1.2.0-provisional"
WAVEFORM_HOP_SEC = 0.01


def compute_waveform_arrays(
    samples: np.ndarray,
    *,
    sample_rate: int,
    hop_sec: float = WAVEFORM_HOP_SEC,
) -> tuple[float, list[float], list[float]]:
    if samples.size == 0:
        return hop_sec, [], []
    hop = max(int(round(hop_sec * sample_rate)), 1)
    mins: list[float] = []
    maxs: list[float] = []
    for start in range(0, samples.size, hop):
        window = samples[start : start + hop]
        if window.size == 0:
            continue
        mins.append(float(np.min(window)))
        maxs.append(float(np.max(window)))
    return hop_sec, mins, maxs


def analyze_waveform(pcm: AnalysisPcm) -> tuple[dict[str, object], list[str]]:
    """Return waveform payload and warning codes."""
    warnings: list[str] = []
    if not pcm.has_audio or pcm.samples.size == 0:
        warnings.append(AUDIO_ABSENT_WARNING)
        return {"hopSec": WAVEFORM_HOP_SEC, "min": [], "max": []}, warnings

    hop_sec, mins, maxs = compute_waveform_arrays(
        pcm.samples,
        sample_rate=pcm.sample_rate,
        hop_sec=WAVEFORM_HOP_SEC,
    )
    return {"hopSec": hop_sec, "min": mins, "max": maxs}, warnings


def waveform_capability(
    *, status: str = "completed", warning_codes: list[str] | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "version": WAVEFORM_ANALYZER_VERSION,
    }
    if warning_codes:
        payload["warningCodes"] = warning_codes
    return payload
