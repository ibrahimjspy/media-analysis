"""Neural beat pilot tests: clock validation, bounded decoding and legacy hashes."""

import time
import wave

import numpy as np
import pytest

from media_analysis.errors import CANCELLED, AnalyzeError
from media_analysis.features.beat_this import BeatThisAnalyzer, validate_events
from media_analysis.request_hash import request_hash
from media_analysis.schemas import AnalyzeRequest


class Guard:
    """Small stage deadline for real local FFmpeg tests."""

    def __init__(self):
        self.deadline = time.monotonic() + 20

    def __call__(self):
        assert self.remaining() > 0

    def remaining(self):
        return self.deadline - time.monotonic()


@pytest.mark.parametrize(
    "events", [[float("nan")], [float("inf")], [-0.1], [1.0], [0.2, 0.2], [0.4, 0.2]]
)
def test_rejects_invalid_model_timestamps(events):
    with pytest.raises(ValueError):
        validate_events(events, 1)


def test_zero_and_coordinate_conversion_are_preserved():
    assert validate_events([0, 0.234], 1) == [
        dict(timeSec=0, sampleIndex=0),
        dict(timeSec=0.234, sampleIndex=5160),
    ]
    assert validate_events([], 1) == []


def test_legacy_option_hash_does_not_change_for_default_false():
    old = dict(rhythmOptions=dict(minBpm=50, maxBpm=200))
    assert request_hash(old) == request_hash(
        dict(rhythmOptions=dict(**old["rhythmOptions"], neuralBeats=False))
    )
    assert request_hash(old) != request_hash(
        dict(rhythmOptions=dict(**old["rhythmOptions"], neuralBeats=True))
    )


def test_neural_is_audio_only():
    with pytest.raises(ValueError):
        AnalyzeRequest(
            idempotencyKey="test",
            mediaKind="video",
            features=["rhythm"],
            source=dict(signedGetUrl="http://localhost/a", expiresAt="2099-01-01T00:00:00Z"),
            rhythmOptions=dict(neuralBeats=True),
        )


@pytest.mark.ffmpeg
def test_decodes_original_stereo_to_22050_and_preserves_impulse_clock(tmp_path):
    # An asymmetric stereo impulse is independently placed at 0.5 s in 44.1 kHz.
    samples = np.zeros((44100, 2), dtype="<i2")
    samples[22050, 0] = 24000
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(samples.tobytes())

    def predictor(pcm, rate):
        assert rate == 22050 and pcm.dtype == np.float32 and pcm.ndim == 1
        assert abs(int(np.argmax(np.abs(pcm))) / rate - 0.5) < 0.002
        return [0.5], [0.5]

    result = BeatThisAnalyzer(predictor, 2).analyze(path, 1, Guard())
    assert result["beats"] == [dict(timeSec=0.5, sampleIndex=11025)]
    assert result["diagnostics"]["downbeatsQualified"] is False


def test_limits_and_cancellation_prevent_inference(tmp_path):
    def forbidden(*args):
        pytest.fail("inference should not run")

    analyzer = BeatThisAnalyzer(forbidden, 2)
    with pytest.raises(AnalyzeError):
        analyzer.analyze(tmp_path / "none", 3, Guard())

    def cancelled():
        raise AnalyzeError(CANCELLED, "cancelled")

    with pytest.raises(AnalyzeError, match="cancelled"):
        analyzer.analyze(tmp_path / "none", 1, cancelled)


def test_bad_checkpoint_fails_before_loading(tmp_path):
    checkpoint = tmp_path / "bad.ckpt"
    checkpoint.write_bytes(b"not weights")
    with pytest.raises(ValueError):
        BeatThisAnalyzer.load(checkpoint, 10)


def test_optional_startup_loads_once_only_when_enabled(settings, monkeypatch):
    from media_analysis.runtime import load_runtime

    calls = []
    sentinel = object()
    monkeypatch.setattr(BeatThisAnalyzer, "load", lambda *args: calls.append(1) or sentinel)
    disabled = load_runtime(
        settings.model_copy(update={"media_analysis_neural_beats_enabled": False})
    )
    assert disabled.neural_beats is None and not calls
    enabled = load_runtime(
        settings.model_copy(update={"media_analysis_neural_beats_enabled": True})
    )
    assert enabled.neural_beats is sentinel and calls == [1]


def test_optional_startup_failure_preserves_ordinary_readiness(settings, monkeypatch):
    from media_analysis.runtime import load_runtime

    def fail(*args):
        raise ValueError("bad checkpoint")

    baseline = load_runtime(settings)
    monkeypatch.setattr(BeatThisAnalyzer, "load", fail)
    state = load_runtime(settings.model_copy(update={"media_analysis_neural_beats_enabled": True}))
    assert state.neural_beats is None
    assert state.ready == baseline.ready
    assert state.neural_beats_error == "NEURAL_BEATS_UNAVAILABLE"
