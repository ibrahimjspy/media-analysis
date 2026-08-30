import sys

import numpy as np
import pytest

from media_analysis.errors import CANCELLED, AnalyzeError
from media_analysis.features.silero_vad import (
    DEFAULT_SPEECH_THRESHOLD,
    SILERO_VAD_CAPABILITY_VERSION,
    SILERO_VAD_MODEL_FILENAME,
    SILERO_VAD_MODEL_VERSION,
    SILERO_VAD_STATE_SHAPE,
    FakeSileroVadV6Session,
    VadWindowScore,
    load_silero_vad_session,
    merge_speech_segments,
    score_vad_windows,
    speech_segments_to_payload,
    warmup_silero_vad,
)


@pytest.mark.unit
def test_silero_model_expectations_for_v6_2() -> None:
    assert SILERO_VAD_MODEL_FILENAME == "silero_vad.onnx"
    assert SILERO_VAD_MODEL_VERSION == "6.2"
    assert SILERO_VAD_CAPABILITY_VERSION == "silero-vad-onnx-6.2-provisional"
    assert SILERO_VAD_STATE_SHAPE == (2, 1, 128)


@pytest.mark.unit
def test_load_silero_vad_requires_existing_model_file(tmp_path) -> None:
    missing = tmp_path / SILERO_VAD_MODEL_FILENAME
    with pytest.raises(FileNotFoundError, match="Silero VAD model not found"):
        load_silero_vad_session(missing)


@pytest.mark.unit
def test_fake_silero_v6_warmup_runs_single_chunk() -> None:
    session = FakeSileroVadV6Session(constant=0.2)
    warmup_silero_vad(session)
    assert len(session.calls) == 1
    assert session.calls[0].shape == (512,)


@pytest.mark.unit
def test_fake_silero_v6_io_contract() -> None:
    session = FakeSileroVadV6Session(constant=0.5)
    assert [item.name for item in session.get_inputs()] == ["input", "state", "sr"]
    assert [item.name for item in session.get_outputs()] == ["output", "stateN"]


@pytest.mark.unit
def test_score_vad_windows_empty_pcm() -> None:
    session = FakeSileroVadV6Session(constant=0.9)
    assert score_vad_windows(session, np.zeros(0, dtype=np.float32)) == []


@pytest.mark.unit
def test_score_vad_windows_uses_injected_probabilities() -> None:
    session = FakeSileroVadV6Session(probabilities=[0.1, 0.9, 0.85])
    samples = np.ones(512 * 3, dtype=np.float32) * 0.01
    windows = score_vad_windows(session, samples)
    assert len(windows) == 3
    assert windows[1].score == pytest.approx(0.9)


@pytest.mark.unit
def test_score_vad_windows_clamps_probability_and_duration() -> None:
    session = FakeSileroVadV6Session(probabilities=[1.5, -0.2])
    samples = np.ones(700, dtype=np.float32) * 0.01
    windows = score_vad_windows(session, samples)
    assert windows[0].score == pytest.approx(1.0)
    assert windows[1].score == pytest.approx(0.0)
    assert windows[-1].end_sec_exclusive == pytest.approx(700 / 16_000)


@pytest.mark.unit
def test_score_vad_windows_cancel_per_chunk() -> None:
    session = FakeSileroVadV6Session(probabilities=[0.9, 0.9, 0.9])
    samples = np.ones(512 * 3, dtype=np.float32)
    calls = {"count": 0}

    def cancel() -> None:
        calls["count"] += 1
        if calls["count"] >= 2:
            raise AnalyzeError(CANCELLED, "Analysis cancelled")

    with pytest.raises(AnalyzeError) as exc:
        score_vad_windows(session, samples, cancel_check=cancel)
    assert exc.value.code == CANCELLED


@pytest.mark.unit
def test_merge_speech_segments_joins_short_gaps() -> None:
    windows = [
        VadWindowScore(0.0, 0.032, 0.9),
        VadWindowScore(0.032, 0.064, 0.2),
        VadWindowScore(0.064, 0.096, 0.88),
    ]
    segments = merge_speech_segments(
        windows,
        threshold=DEFAULT_SPEECH_THRESHOLD,
        min_speech_sec=0.03,
        min_silence_sec=0.05,
        max_duration_sec=0.096,
    )
    assert len(segments) == 1
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec_exclusive == pytest.approx(0.096)
    assert segments[0].score == pytest.approx(0.9)


@pytest.mark.unit
def test_merge_speech_segments_clamps_to_pcm_duration() -> None:
    windows = [
        VadWindowScore(0.0, 0.5, 0.95),
        VadWindowScore(0.5, 1.0, 0.95),
    ]
    segments = merge_speech_segments(windows, min_speech_sec=0.01, max_duration_sec=0.75)
    assert segments[0].end_sec_exclusive == pytest.approx(0.75)


@pytest.mark.unit
def test_merge_speech_segments_splits_on_long_silence() -> None:
    windows = [
        VadWindowScore(0.0, 0.032, 0.95),
        VadWindowScore(0.032, 0.064, 0.1),
        VadWindowScore(0.064, 0.096, 0.1),
        VadWindowScore(0.096, 0.128, 0.1),
        VadWindowScore(0.128, 0.160, 0.1),
        VadWindowScore(0.160, 0.192, 0.92),
    ]
    segments = merge_speech_segments(windows, min_speech_sec=0.03, min_silence_sec=0.10)
    assert len(segments) == 2
    assert segments[0].end_sec_exclusive <= 0.064
    assert segments[1].start_sec >= 0.128


@pytest.mark.unit
def test_speech_segments_to_payload_raw_model() -> None:
    from media_analysis.features.silero_vad import SpeechSegment

    payload = speech_segments_to_payload(
        [SpeechSegment(0.5, 1.25, 0.87)],
    )
    assert payload == [
        {
            "startSec": 0.5,
            "endSecExclusive": 1.25,
            "score": 0.87,
            "scoreType": "raw_model",
        }
    ]


@pytest.mark.unit
def test_load_silero_vad_uses_onnxruntime_contract(tmp_path, monkeypatch) -> None:
    model = tmp_path / SILERO_VAD_MODEL_FILENAME
    model.write_bytes(b"not-a-real-onnx")
    calls: list[tuple] = []

    class _FakeSession:
        pass

    class _FakeSessionOptions:
        intra_op_num_threads = 1

    class _FakeOrt:
        SessionOptions = _FakeSessionOptions

        @staticmethod
        def InferenceSession(path, sess_options=None, providers=None):
            calls.append((path, providers))
            return _FakeSession()

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt())
    session = load_silero_vad_session(model)
    assert isinstance(session, _FakeSession)
    assert calls
    assert str(model) in str(calls[0][0])
