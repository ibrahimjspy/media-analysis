"""v1.1 audio analyzer — provisional deterministic policy behind benchmark gates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from media_analysis.errors import CANCELLED, FEATURE_UNAVAILABLE, AnalyzeError
from media_analysis.features.audio_dsp import (
    BPM_POLICY_VERSION,
    LOUDNESS_POLICY_VERSION,
    ONSET_POLICY_VERSION,
    RMS_POLICY_VERSION,
    compute_onsets,
    compute_rms_series,
    estimate_bpm,
    measure_loudness_ffmpeg,
    pcm_has_measurable_signal,
)
from media_analysis.features.audio_pcm import AnalysisPcm
from media_analysis.features.silero_vad import (
    SILERO_VAD_PREPROCESSING_VERSION,
    SileroVadSessionLike,
    merge_speech_segments,
    score_vad_windows,
    speech_segments_to_payload,
)
from media_analysis.frames import Rational
from media_analysis.models_manifest import ModelEntry

AUDIO_ANALYZER_VERSION = "audio-v1.1.0-provisional"
VAD_POLICY_VERSION = "silero-merge-threshold0.5-1.0.0-provisional"
AUDIO_ABSENT_WARNING = "AUDIO_ABSENT"


@dataclass(frozen=True, slots=True)
class AudioAnalysisResult:
    duration_sec: float
    rms_hop_sec: float
    rms_values: list[float]
    onsets: list[dict[str, object]]
    speech: list[dict[str, object]]
    integrated_lufs: float | None
    true_peak_db: float | None
    bpm: float | None
    warning_codes: tuple[str, ...]
    status: str

    def to_payload(self) -> dict[str, object]:
        return {
            "durationSec": self.duration_sec,
            "rms": {"hopSec": self.rms_hop_sec, "values": self.rms_values},
            "onsets": self.onsets,
            "speech": self.speech,
            "integratedLufs": self.integrated_lufs,
            "truePeakDb": self.true_peak_db,
            "bpm": self.bpm,
        }


def empty_audio_analysis(
    *, warning_codes: tuple[str, ...] = (AUDIO_ABSENT_WARNING,)
) -> AudioAnalysisResult:
    hop_sec, values = compute_rms_series(np.zeros(0, dtype=np.float32), sample_rate=16_000)
    return AudioAnalysisResult(
        duration_sec=0.0,
        rms_hop_sec=hop_sec,
        rms_values=values,
        onsets=[],
        speech=[],
        integrated_lufs=None,
        true_peak_db=None,
        bpm=None,
        warning_codes=warning_codes,
        status="completed",
    )


def analyze_audio(
    pcm: AnalysisPcm,
    *,
    source_path: Path | None = None,
    fps: Rational | None = None,
    vad_session: SileroVadSessionLike | None = None,
    require_vad: bool = True,
    timeout_sec: float | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> AudioAnalysisResult:
    """Measure audio facts from canonical analysis PCM."""
    if cancel_check:
        cancel_check()

    if not pcm.has_audio or pcm.samples.size == 0:
        return empty_audio_analysis()

    if require_vad and vad_session is None:
        raise AnalyzeError(
            FEATURE_UNAVAILABLE,
            "Silero VAD session is required for audio analysis",
        )

    if cancel_check:
        cancel_check()

    hop_sec, rms_values = compute_rms_series(pcm.samples, sample_rate=pcm.sample_rate)
    onsets = compute_onsets(pcm.samples, sample_rate=pcm.sample_rate, fps=fps)

    speech: list[dict[str, object]] = []
    if vad_session is not None:
        windows = score_vad_windows(
            vad_session,
            pcm.samples,
            sample_rate=pcm.sample_rate,
            cancel_check=cancel_check,
        )
        segments = merge_speech_segments(
            windows,
            max_duration_sec=pcm.duration_sec,
        )
        speech = speech_segments_to_payload(segments)

    integrated_lufs: float | None = None
    true_peak_db: float | None = None
    if source_path is not None:
        loudness = measure_loudness_ffmpeg(
            source_path,
            timeout_sec=timeout_sec,
            cancel_check=cancel_check,
        )
        integrated_lufs = loudness.integrated_lufs
        true_peak_db = loudness.true_peak_db

    bpm = None
    if pcm_has_measurable_signal(pcm):
        bpm = estimate_bpm(pcm.samples, sample_rate=pcm.sample_rate)

    return AudioAnalysisResult(
        duration_sec=pcm.duration_sec,
        rms_hop_sec=hop_sec,
        rms_values=rms_values,
        onsets=onsets,
        speech=speech,
        integrated_lufs=integrated_lufs,
        true_peak_db=true_peak_db,
        bpm=bpm,
        warning_codes=(),
        status="completed",
    )


def audio_capability(
    *, status: str = "completed", warning_codes: list[str] | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "version": AUDIO_ANALYZER_VERSION,
    }
    if warning_codes:
        payload["warningCodes"] = warning_codes
    return payload


def audio_provenance(
    *,
    vad_entry: ModelEntry | None = None,
    vad_runtime: str = "onnxruntime-cpu",
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "audioAnalyzerVersion": AUDIO_ANALYZER_VERSION,
        "audioRmsPolicyVersion": RMS_POLICY_VERSION,
        "audioOnsetPolicyVersion": ONSET_POLICY_VERSION,
        "audioVadPolicyVersion": VAD_POLICY_VERSION,
        "audioBpmPolicyVersion": BPM_POLICY_VERSION,
        "audioLoudnessPolicyVersion": LOUDNESS_POLICY_VERSION,
    }
    if vad_entry is not None:
        provenance["audioVadModel"] = {
            "name": vad_entry.name,
            "version": vad_entry.version,
            "artifactSha256": vad_entry.sha256,
            "runtimeProvider": vad_runtime,
            "preprocessingVersion": SILERO_VAD_PREPROCESSING_VERSION,
            "license": vad_entry.license,
        }
    return provenance


def rethrow_cancelled(exc: BaseException) -> None:
    if isinstance(exc, AnalyzeError) and exc.code == CANCELLED:
        raise exc
