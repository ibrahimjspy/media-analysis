"""Deterministic audio DSP helpers for v1.1 (NumPy-first; librosa optional for BPM)."""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from media_analysis.errors import DECODE_FAILED, AnalyzeError
from media_analysis.features.audio_pcm import AnalysisPcm
from media_analysis.features.audio_subprocess import run_bounded_subprocess
from media_analysis.frames import Rational

RMS_POLICY_VERSION = "rms-hop50ms-mean-square-1.0.0-provisional"
RMS_HOP_SEC = 0.05

ONSET_POLICY_VERSION = "stft-flux-bands-1.0.0-provisional"
ONSET_STFT_NFFT = 2048
ONSET_STFT_HOP = 512
ONSET_FLUX_THRESHOLD = 0.08
ONSET_MIN_GAP_SEC = 0.04

BPM_POLICY_VERSION = "flux-autocorr-confidence0.45-1.0.0-provisional"
BPM_MIN = 60.0
BPM_MAX = 200.0
BPM_MIN_SIGNAL_SEC = 2.0
BPM_MIN_CONFIDENCE = 0.45
BPM_MIN_FLUX_VARIANCE = 1e-4

LOUDNESS_POLICY_VERSION = "ffmpeg-ebur128-peak-1.0.1-provisional"

BAND_EDGES_HZ = {
    "low": (20.0, 200.0),
    "mid": (200.0, 2000.0),
    "high": (2000.0, 8000.0),
}


@dataclass(frozen=True, slots=True)
class LoudnessMeasurement:
    integrated_lufs: float | None
    true_peak_db: float | None


def compute_rms_series(
    samples: np.ndarray,
    *,
    sample_rate: int,
    hop_sec: float = RMS_HOP_SEC,
) -> tuple[float, list[float]]:
    if samples.size == 0:
        return hop_sec, []
    hop = max(int(round(hop_sec * sample_rate)), 1)
    values: list[float] = []
    for start in range(0, samples.size, hop):
        window = samples[start : start + hop]
        if window.size == 0:
            continue
        values.append(float(np.sqrt(np.mean(np.square(window), dtype=np.float64))))
    return hop_sec, values


def compute_onsets(
    samples: np.ndarray,
    *,
    sample_rate: int,
    fps: Rational | None = None,
) -> list[dict[str, object]]:
    if samples.size < ONSET_STFT_NFFT:
        return []

    frames = _stft_mag(samples, n_fft=ONSET_STFT_NFFT, hop=ONSET_STFT_HOP)
    if frames.size == 0:
        return []

    flux = np.maximum(frames[:, 1:] - frames[:, :-1], 0.0).sum(axis=0)
    if flux.size == 0:
        return []
    flux = flux / (float(np.max(flux)) + 1e-12)
    frame_times = (np.arange(flux.size) * ONSET_STFT_HOP) / sample_rate
    peaks = _pick_peaks(flux, frame_times, min_gap_sec=ONSET_MIN_GAP_SEC)

    band_flux = {
        name: _band_flux(samples, sample_rate, low=low, high=high)
        for name, (low, high) in BAND_EDGES_HZ.items()
    }

    onsets: list[dict[str, object]] = []
    for sec, strength in peaks:
        band = _dominant_band(sec, band_flux, sample_rate)
        item: dict[str, object] = {
            "sec": sec,
            "strength": strength,
        }
        if band is not None:
            item["band"] = band
        if fps is not None and fps.numerator > 0:
            item["sourceFrameApprox"] = int(sec * fps.numerator / fps.denominator)
        onsets.append(item)
    return onsets


def estimate_bpm(samples: np.ndarray, *, sample_rate: int) -> float | None:
    if samples.size < int(BPM_MIN_SIGNAL_SEC * sample_rate):
        return None
    if float(np.max(np.abs(samples))) < 1e-4:
        return None

    flux = _onset_flux_envelope(samples)
    if flux.size < 8 or float(np.var(flux)) < BPM_MIN_FLUX_VARIANCE:
        return None

    candidate = _estimate_bpm_from_flux(flux, sample_rate=sample_rate)
    if candidate is None:
        candidate = _estimate_bpm_librosa(samples, sample_rate)
    if candidate is None:
        return None

    confidence = _bpm_confidence(flux, sample_rate=sample_rate, bpm=candidate)
    if confidence < BPM_MIN_CONFIDENCE:
        return None
    return candidate


def measure_loudness_ffmpeg(
    path: Path,
    *,
    timeout_sec: float | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> LoudnessMeasurement:
    if cancel_check:
        cancel_check()
    if shutil.which("ffmpeg") is None:
        raise AnalyzeError(DECODE_FAILED, "ffmpeg is not installed")
    args = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-filter_complex",
        "ebur128=peak=true",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = run_bounded_subprocess(
            args,
            timeout_sec=timeout_sec,
            cancel_check=cancel_check,
            text_mode=True,
        )
    except AnalyzeError:
        raise
    except Exception as exc:
        raise AnalyzeError(DECODE_FAILED, "Could not decode source") from exc

    if completed.returncode != 0:
        raise AnalyzeError(DECODE_FAILED, "Could not decode source")

    text = (completed.stderr or "") + (completed.stdout or "")
    integrated = _parse_integrated_lufs(text)
    true_peak = _parse_true_peak_db(text)
    return LoudnessMeasurement(integrated_lufs=integrated, true_peak_db=true_peak)


def pcm_has_measurable_signal(pcm: AnalysisPcm, *, threshold: float = 1e-4) -> bool:
    """True when a present audio stream contains non-zero samples."""
    if not pcm.has_audio or pcm.samples.size == 0:
        return False
    return float(np.max(np.abs(pcm.samples))) >= threshold


def _stft_mag(samples: np.ndarray, *, n_fft: int, hop: int) -> np.ndarray:
    if samples.size < n_fft:
        return np.zeros((0, 0), dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    frames: list[np.ndarray] = []
    for start in range(0, samples.size - n_fft + 1, hop):
        chunk = samples[start : start + n_fft] * window
        spectrum = np.fft.rfft(chunk)
        frames.append(np.abs(spectrum).astype(np.float32))
    if not frames:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(frames, axis=1)


def _onset_flux_envelope(samples: np.ndarray) -> np.ndarray:
    frames = _stft_mag(samples, n_fft=ONSET_STFT_NFFT, hop=ONSET_STFT_HOP)
    if frames.size == 0:
        return np.zeros(0, dtype=np.float32)
    flux = np.maximum(frames[:, 1:] - frames[:, :-1], 0.0).sum(axis=0)
    if flux.size == 0:
        return np.zeros(0, dtype=np.float32)
    return flux.astype(np.float64)


def _estimate_bpm_from_flux(flux: np.ndarray, *, sample_rate: int) -> float | None:
    hop = ONSET_STFT_HOP
    centered = flux - float(np.mean(flux))
    if float(np.max(np.abs(centered))) < 1e-8:
        return None
    corr = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    corr[0] = 0.0
    min_lag = int((60.0 / BPM_MAX) * sample_rate / hop)
    max_lag = int((60.0 / BPM_MIN) * sample_rate / hop)
    if max_lag >= corr.size or min_lag >= max_lag:
        return None
    segment = corr[min_lag : max_lag + 1]
    if segment.size == 0:
        return None
    lag = int(min_lag + int(np.argmax(segment)))
    if lag <= 0:
        return None
    bpm = 60.0 * sample_rate / (lag * hop)
    if not BPM_MIN <= bpm <= BPM_MAX:
        return None
    return round(float(bpm), 2)


def _bpm_confidence(flux: np.ndarray, *, sample_rate: int, bpm: float) -> float:
    hop = ONSET_STFT_HOP
    centered = flux - float(np.mean(flux))
    corr = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    if corr.size == 0:
        return 0.0
    lag = int(round((60.0 / bpm) * sample_rate / hop))
    min_lag = int((60.0 / BPM_MAX) * sample_rate / hop)
    if lag <= 0 or lag >= corr.size:
        return 0.0
    peak = float(corr[lag])
    if peak <= 0:
        return 0.0
    local = corr[max(lag - 2, 0) : min(lag + 3, corr.size)]
    baseline = float(np.median(corr[1 : max(5, min_lag)])) if min_lag > 1 else 0.0
    return peak / (baseline + float(np.std(local)) + 1e-8)


def _band_flux(
    samples: np.ndarray,
    sample_rate: int,
    *,
    low: float,
    high: float,
) -> np.ndarray:
    frames = _stft_mag(samples, n_fft=ONSET_STFT_NFFT, hop=ONSET_STFT_HOP)
    if frames.size == 0:
        return np.zeros(0, dtype=np.float32)
    freqs = np.fft.rfftfreq(ONSET_STFT_NFFT, d=1.0 / sample_rate)
    mask = (freqs >= low) & (freqs < high)
    band = frames[mask, :]
    if band.size == 0:
        return np.zeros(frames.shape[1] - 1, dtype=np.float32)
    flux = np.maximum(band[:, 1:] - band[:, :-1], 0.0).sum(axis=0)
    return flux.astype(np.float32)


def _dominant_band(
    sec: float,
    band_flux: dict[str, np.ndarray],
    sample_rate: int,
) -> str | None:
    frame = int(round(sec * sample_rate / ONSET_STFT_HOP))
    scores: dict[str, float] = {}
    for name, flux in band_flux.items():
        if flux.size == 0:
            continue
        index = min(max(frame, 0), flux.size - 1)
        scores[name] = float(flux[index])
    if not scores:
        return None
    return max(scores, key=scores.get)


def _pick_peaks(
    values: np.ndarray,
    times: np.ndarray,
    *,
    min_gap_sec: float,
) -> list[tuple[float, float]]:
    peaks: list[tuple[float, float]] = []
    last_sec = -1.0
    for index in range(1, values.size - 1):
        if values[index] < ONSET_FLUX_THRESHOLD:
            continue
        if values[index] <= values[index - 1] or values[index] < values[index + 1]:
            continue
        sec = float(times[index])
        if sec - last_sec < min_gap_sec:
            if peaks and values[index] > peaks[-1][1]:
                peaks[-1] = (sec, float(values[index]))
            continue
        peaks.append((sec, float(values[index])))
        last_sec = sec
    return peaks


def _estimate_bpm_librosa(samples: np.ndarray, sample_rate: int) -> float | None:
    try:
        import librosa
    except ImportError:
        return None
    try:
        onset_env = librosa.onset.onset_strength(y=samples, sr=sample_rate)
        tempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)
    except Exception:
        return None
    if isinstance(tempo, tuple):
        bpm = tempo[0]
    else:
        bpm = tempo
    if bpm is None:
        return None
    value = float(np.asarray(bpm).reshape(-1)[0])
    if not BPM_MIN <= value <= BPM_MAX:
        return None
    return round(value, 2)


def _parse_integrated_lufs(text: str) -> float | None:
    patterns = (
        r"Integrated loudness:\s*\n\s*I:\s*(-inf|-?\d+(?:\.\d+)?)\s*LUFS",
        r"Integrated loudness:\s*(-inf|-?\d+(?:\.\d+)?)\s*LUFS",
        r"Integrated loudness \(I\):\s*(-inf|-?\d+(?:\.\d+)?)\s*LUFS",
    )
    raw = _last_metric_value(text, patterns)
    if raw is None:
        return None
    if raw == "-inf":
        return None
    return float(raw)


def _parse_true_peak_db(text: str) -> float | None:
    patterns = (
        r"True peak:\s*\n\s*Peak:\s*(-inf|-?\d+(?:\.\d+)?)\s*dBFS",
        r"True peak:\s*(-inf|-?\d+(?:\.\d+)?)\s*dBFS",
        r"Peak:\s*(-inf|-?\d+(?:\.\d+)?)\s*dBFS",
    )
    raw = _last_metric_value(text, patterns)
    if raw is None:
        return None
    if raw == "-inf":
        return None
    return float(raw)


def _last_metric_value(text: str, patterns: tuple[str, ...]) -> str | None:
    matches = [
        (match.start(), match.group(1).lower())
        for pattern in patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def parse_ebur128_output(text: str) -> LoudnessMeasurement:
    """Parse ffmpeg ebur128 stderr/stdout for tests and diagnostics."""
    return LoudnessMeasurement(
        integrated_lufs=_parse_integrated_lufs(text),
        true_peak_db=_parse_true_peak_db(text),
    )
