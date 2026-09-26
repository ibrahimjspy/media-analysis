"""Optional Beat This Small pilot on canonical playback audio.

Weights are checksum-verified and warmed at worker startup, never downloaded or
loaded per request. Legacy 16 kHz evidence is independent of this 22.05 kHz path.
"""

from __future__ import annotations

import hashlib
import math
import threading
from pathlib import Path

import numpy as np

from media_analysis.errors import DECODE_FAILED, LIMIT_EXCEEDED, AnalyzeError
from media_analysis.features.audio_subprocess import run_bounded_subprocess

ALGORITHM = "beat-this-small0-v1"
CHECKPOINT_SHA256 = "6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b"
CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/small0.ckpt"
SAMPLE_RATE = 22_050
RECIPE = f"{ALGORITHM}:{CHECKPOINT_SHA256}:mono22050-f32-playback-v1:dbn-false"


def outcome(status: str, code: str | None = None) -> dict:
    """Create a scoped outcome without implying silence on missing inference."""
    return dict(
        status=status,
        algorithmVersion=ALGORITHM,
        checkpointSha256=CHECKPOINT_SHA256,
        sampleRate=SAMPLE_RATE,
        frameHopSec=0.02,
        timestampOrigin="decoded-playback-start",
        evidenceKind="model-estimate",
        beats=[],
        warningCodes=[code] if code else [],
        productionQualified=False,
    )


def validate_events(times, duration: float) -> list[dict]:
    """Reject nonfinite, unordered, duplicate or out-of-source model events.

    sampleIndex is a rounded coordinate conversion, not an acoustic-accuracy
    claim. Events keep their predicted times; no BPM folding or grid fabrication.
    """
    events = []
    previous = -1.0
    for value in times:
        time_sec = float(value)
        if not math.isfinite(time_sec) or not previous < time_sec < duration:
            raise ValueError("INVALID_NEURAL_BEAT_TIME")
        if time_sec < 0:
            raise ValueError("INVALID_NEURAL_BEAT_TIME")
        events.append(dict(timeSec=time_sec, sampleIndex=round(time_sec * SAMPLE_RATE)))
        previous = time_sec
    return events


class BeatThisAnalyzer:
    """Own one warmed CPU predictor and serialize bounded inference calls.

    Instantiation is for startup only. Optional model failure must not disable
    legacy audio processing. Cancellation is checked while waiting and around
    decoding/inference; an active Torch operation cannot be interrupted here.
    """

    def __init__(self, predictor, max_duration: float):
        self.predictor = predictor
        self.max_duration = max_duration
        self.lock = threading.Lock()

    @classmethod
    def load(cls, checkpoint: Path, max_duration: float):
        """Verify immutable bytes before importing/loading the local checkpoint."""
        if not checkpoint.is_file() or checkpoint.stat().st_size != 8_451_101:
            raise ValueError("BEAT_THIS_CHECKPOINT_UNAVAILABLE")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("BEAT_THIS_CHECKPOINT_CHECKSUM")
        import torch
        from beat_this.inference import Audio2Beats

        torch.set_num_threads(4)
        predictor = Audio2Beats(
            checkpoint_path=str(checkpoint), device="cpu", float16=False, dbn=False
        )
        predictor(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
        return cls(predictor, max_duration)

    def analyze(self, source: Path, duration: float, guard) -> dict:
        """Decode directly from playback, then validate source-local timestamps.

        FFmpeg bounds emitted PCM by duration and deadline. Source metadata must
        already pass the worker probe limits. No upsampling of legacy PCM occurs.
        """
        if not math.isfinite(duration) or not 0 < duration <= self.max_duration:
            raise AnalyzeError(LIMIT_EXCEEDED, "Neural beat duration exceeds pilot limit")
        while not self.lock.acquire(timeout=0.1):
            guard()
        try:
            guard()
            decoded = run_bounded_subprocess(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-i",
                    str(source),
                    "-t",
                    str(self.max_duration + 0.1),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-f",
                    "f32le",
                    "-acodec",
                    "pcm_f32le",
                    "pipe:1",
                ],
                timeout_sec=guard.remaining(),
                cancel_check=guard,
            )
            if decoded.returncode:
                raise AnalyzeError(DECODE_FAILED, "Neural PCM decode failed")
            pcm = np.frombuffer(decoded.stdout, dtype="<f4").copy()
            if (
                not len(pcm)
                or len(pcm) > math.ceil((self.max_duration + 0.1) * SAMPLE_RATE)
                or not np.isfinite(pcm).all()
                or abs(len(pcm) / SAMPLE_RATE - duration) > 0.05
            ):
                raise ValueError("INVALID_NEURAL_PCM_CLOCK")
            guard()
            beats, downbeats = self.predictor(pcm, SAMPLE_RATE)
            guard()
            result = outcome("completed")
            result["beats"] = validate_events(beats, duration)
            result["diagnostics"] = dict(
                downbeatTimesSec=[e["timeSec"] for e in validate_events(downbeats, duration)],
                downbeatsQualified=False,
            )
            return result
        finally:
            self.lock.release()
