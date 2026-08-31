"""Silero VAD ONNX session wrapper (MIT, vendored v6.2 artifact at image build)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SILERO_VAD_MODEL_FILENAME = "silero_vad.onnx"
SILERO_VAD_MODEL_VERSION = "6.2"
SILERO_VAD_CAPABILITY_VERSION = "silero-vad-onnx-6.2-provisional"
SILERO_VAD_PREPROCESSING_VERSION = "silero-chunk512-context64-mono16k-v6"
SILERO_VAD_WINDOW_SAMPLES = 512
SILERO_VAD_CONTEXT_SAMPLES = 64
SILERO_VAD_SAMPLE_RATE = 16_000
SILERO_VAD_STATE_SHAPE = (2, 1, 128)

DEFAULT_SPEECH_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_SEC = 0.10
DEFAULT_MIN_SILENCE_SEC = 0.15


class SileroVadSessionLike(Protocol):
    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]: ...

    def get_inputs(self) -> list[Any]: ...

    def get_outputs(self) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class VadWindowScore:
    start_sec: float
    end_sec_exclusive: float
    score: float


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    start_sec: float
    end_sec_exclusive: float
    score: float


@dataclass(frozen=True, slots=True)
class _FakeIO:
    name: str


@dataclass
class FakeSileroVadV6Session:
    """v6.2-style test double: input/state/sr in, output/stateN out."""

    probabilities: list[float] | None = None
    constant: float = 0.0
    calls: list[np.ndarray] = field(default_factory=list)
    state_shape: tuple[int, int, int] = SILERO_VAD_STATE_SHAPE

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        audio = np.asarray(input_feed["input"], dtype=np.float32).reshape(-1)
        self.calls.append(audio.copy())
        if self.probabilities is not None:
            index = min(len(self.calls) - 1, len(self.probabilities) - 1)
            prob = float(self.probabilities[index])
        else:
            prob = float(self.constant)
        prob = _clamp_probability(prob)
        state = np.asarray(feed_get_state(input_feed), dtype=np.float32)
        return [np.array([[prob]], dtype=np.float32), state.copy()]

    def get_inputs(self) -> list[_FakeIO]:
        return [_FakeIO("input"), _FakeIO("state"), _FakeIO("sr")]

    def get_outputs(self) -> list[_FakeIO]:
        return [_FakeIO("output"), _FakeIO("stateN")]


# Backward-compatible alias for existing tests.
FakeSileroVadSession = FakeSileroVadV6Session


def feed_get_state(input_feed: dict[str, Any]) -> np.ndarray:
    state = input_feed.get("state")
    if state is None:
        return np.zeros(SILERO_VAD_STATE_SHAPE, dtype=np.float32)
    return np.asarray(state, dtype=np.float32)


def load_silero_vad_session(model_path: Path) -> Any:
    """Load vendored Silero VAD ONNX via ONNX Runtime CPU."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Silero VAD model not found: {model_path}")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for Silero VAD") from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def warmup_silero_vad(session: SileroVadSessionLike) -> None:
    """Deterministic warmup compatible with runtime /ready checks."""
    zeros = np.zeros(
        SILERO_VAD_CONTEXT_SAMPLES + SILERO_VAD_WINDOW_SAMPLES,
        dtype=np.float32,
    )
    _run_vad_chunk(session, zeros, sample_rate=SILERO_VAD_SAMPLE_RATE, state=None)


def score_vad_windows(
    session: SileroVadSessionLike,
    samples: np.ndarray,
    *,
    sample_rate: int = SILERO_VAD_SAMPLE_RATE,
    window_samples: int = SILERO_VAD_WINDOW_SAMPLES,
    cancel_check: Callable[[], None] | None = None,
) -> list[VadWindowScore]:
    if sample_rate != SILERO_VAD_SAMPLE_RATE:
        raise ValueError("Silero VAD expects 16 kHz analysis PCM")
    if samples.size == 0:
        return []

    duration_sec = pcm_duration_sec(samples.size, sample_rate)
    window_sec = window_samples / sample_rate
    count = (samples.size + window_samples - 1) // window_samples
    padded = np.pad(
        samples.astype(np.float32, copy=False),
        (0, count * window_samples - samples.size),
    )
    state = np.zeros(SILERO_VAD_STATE_SHAPE, dtype=np.float32)
    context = np.zeros(SILERO_VAD_CONTEXT_SAMPLES, dtype=np.float32)
    scores: list[VadWindowScore] = []
    for index in range(0, count * window_samples, window_samples):
        if cancel_check:
            cancel_check()
        chunk = padded[index : index + window_samples]
        model_input = np.concatenate((context, chunk))
        prob, state = _run_vad_chunk(
            session,
            model_input,
            sample_rate=sample_rate,
            state=state,
        )
        context = chunk[-SILERO_VAD_CONTEXT_SAMPLES :].copy()
        start = index / sample_rate
        end = min(start + window_sec, duration_sec)
        scores.append(
            VadWindowScore(
                start_sec=start,
                end_sec_exclusive=end,
                score=prob,
            )
        )
    return scores


def pcm_duration_sec(sample_count: int, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return sample_count / sample_rate if sample_count > 0 else 0.0


def merge_speech_segments(
    windows: list[VadWindowScore],
    *,
    threshold: float = DEFAULT_SPEECH_THRESHOLD,
    min_speech_sec: float = DEFAULT_MIN_SPEECH_SEC,
    min_silence_sec: float = DEFAULT_MIN_SILENCE_SEC,
    max_duration_sec: float | None = None,
) -> list[SpeechSegment]:
    if not windows:
        return []

    raw: list[tuple[float, float, list[float]]] = []
    in_speech = False
    start_sec = 0.0
    scores: list[float] = []

    for window in windows:
        if window.score >= threshold:
            if not in_speech:
                start_sec = window.start_sec
                scores = []
                in_speech = True
            scores.append(window.score)
        elif in_speech:
            end_sec = window.start_sec
            if end_sec - start_sec >= min_speech_sec:
                raw.append((start_sec, end_sec, list(scores)))
            in_speech = False

    if in_speech:
        end_sec = windows[-1].end_sec_exclusive
        if end_sec - start_sec >= min_speech_sec:
            raw.append((start_sec, end_sec, list(scores)))

    if not raw:
        return []

    merged: list[tuple[float, float, list[float]]] = [raw[0]]
    for start, end, seg_scores in raw[1:]:
        prev_start, prev_end, prev_scores = merged[-1]
        if start - prev_end < min_silence_sec:
            merged[-1] = (prev_start, end, prev_scores + seg_scores)
        else:
            merged.append((start, end, seg_scores))

    cap = max_duration_sec if max_duration_sec is not None else math.inf
    return [
        SpeechSegment(
            start_sec=start,
            end_sec_exclusive=min(end, cap),
            score=float(max(seg_scores) if seg_scores else threshold),
        )
        for start, end, seg_scores in merged
        if min(end, cap) > start
    ]


def speech_segments_to_payload(segments: list[SpeechSegment]) -> list[dict[str, object]]:
    return [
        {
            "startSec": segment.start_sec,
            "endSecExclusive": segment.end_sec_exclusive,
            "score": segment.score,
            "scoreType": "raw_model",
        }
        for segment in segments
    ]


def _clamp_probability(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def _run_vad_chunk(
    session: SileroVadSessionLike,
    chunk: np.ndarray,
    *,
    sample_rate: int,
    state: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    input_names = {item.name for item in session.get_inputs()}
    output_names = [item.name for item in session.get_outputs()]
    current_state = (
        np.zeros(SILERO_VAD_STATE_SHAPE, dtype=np.float32)
        if state is None
        else np.asarray(state, dtype=np.float32)
    )
    feed: dict[str, Any] = {
        "input": chunk.reshape(1, -1).astype(np.float32),
        "sr": np.array([sample_rate], dtype=np.int64),
    }

    if "state" not in input_names:
        raise RuntimeError("Silero VAD session must expose v6.2 state input")

    feed["state"] = current_state
    outputs = session.run(None, feed)
    prob = _clamp_probability(float(np.asarray(outputs[0]).reshape(-1)[0]))

    state_output_name = next(
        (name for name in output_names if name.lower() in {"staten", "state"}),
        None,
    )
    if state_output_name is None:
        raise RuntimeError("Silero VAD session must expose v6.2 stateN output")
    index = output_names.index(state_output_name)
    next_state = np.asarray(outputs[index], dtype=np.float32)
    return prob, next_state
