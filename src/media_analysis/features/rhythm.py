"""Timestamped acoustic attacks and conservative periodic-attack beat evidence.

No missing events are interpolated. No downbeats, bars or semantic impacts inferred.
"""

from __future__ import annotations

import numpy as np

from media_analysis.features.audio_pcm import AnalysisPcm

RHYTHM_VERSION = "rms-attack-periodic-runs-v1"
RHYTHM_HOP_SEC = 0.005


def analyze_rhythm(pcm: AnalysisPcm, *, min_bpm=50.0, max_bpm=200.0, cancel_check=None) -> dict:
    if cancel_check:
        cancel_check()
    result = {
        "beats": [],
        "onsetCandidates": [],
        "bpm": None,
        "segments": [],
        "algorithmVersion": RHYTHM_VERSION,
        "sampleRate": pcm.sample_rate,
        "hopSec": RHYTHM_HOP_SEC,
        "timestampOrigin": "decoded-playback-start",
        "evidenceTier": "none",
        "reasons": [],
        "productionQualified": False,
    }
    if not pcm.has_audio:
        result["reasons"] = ["AUDIO_ABSENT"]
        return result
    samples = pcm.samples
    if samples.size == 0 or float(np.max(np.abs(samples))) < 1e-4:
        result["reasons"] = ["SILENCE"]
        return result
    hop = max(1, round(pcm.sample_rate * RHYTHM_HOP_SEC))
    padded = np.pad(samples, (0, (-samples.size) % hop)).reshape(-1, hop)
    energy = np.sqrt(np.mean(padded * padded, axis=1))
    novelty = np.maximum(0, energy - np.r_[0.0, energy[:-1]])
    threshold = max(
        0.001,
        float(np.max(novelty)) * 0.15,
        float(np.median(novelty) + 4 * np.median(np.abs(novelty - np.median(novelty)))),
    )
    events = []
    refractory = round(0.08 / RHYTHM_HOP_SEC)
    for index, value in enumerate(novelty):
        if index % 1024 == 0 and cancel_check:
            cancel_check()
        if value < threshold or (index and value < novelty[index - 1]):
            continue
        if index + 1 < len(novelty) and value < novelty[index + 1]:
            continue
        if events and index - events[-1] < refractory:
            continue
        events.append(index)
    candidates = []
    for index in events:
        window = np.abs(samples[index * hop : min(samples.size, (index + 1) * hop)])
        active = np.flatnonzero(window >= max(1e-4, float(window.max()) * 0.1))
        sample = index * hop + (int(active[0]) if active.size else 0)
        candidates.append(
            {
                "timeSec": sample / pcm.sample_rate,
                "sampleIndex": sample,
                "strength": float(novelty[index]),
                "scoreType": "measurement",
            }
        )
    result["onsetCandidates"] = candidates
    times = np.array([event["timeSec"] for event in candidates])
    if len(times) < 4:
        result["reasons"] = ["INSUFFICIENT_PERIODIC_EVIDENCE"]
        return result
    intervals = np.diff(times)
    accepted = set()
    start = 0
    while start < len(intervals):
        period = intervals[start]
        if not 60 / max_bpm <= period <= 60 / min_bpm:
            start += 1
            continue
        end = start + 1
        while end < len(intervals) and abs(intervals[end] / period - 1) <= 0.12:
            end += 1
        if end - start >= 3:
            accepted.update(range(start, end + 1))
            result["segments"].append(
                {
                    "startSec": float(times[start]),
                    "endSec": float(times[end]),
                    "bpm": float(60 / np.median(intervals[start:end])),
                }
            )
        start = end
    result["beats"] = [candidates[index] for index in sorted(accepted)]
    if accepted:
        result["evidenceTier"] = "periodic-acoustic-attacks"
        result["reasons"] = ["HEURISTIC_NOT_MUSICALLY_VERIFIED"]
        bpms = [segment["bpm"] for segment in result["segments"]]
        if max(bpms) / min(bpms) < 1.12:
            result["bpm"] = float(np.median(bpms))
    else:
        result["reasons"] = ["IRREGULAR_OR_AMBIENT"]
    return result
