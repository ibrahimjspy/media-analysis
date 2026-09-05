from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from tests.unit.test_frame_access import make_probe_mp4
from tests.unit.test_subject_pipeline import ScriptedDetector

import media_analysis.analyze as analyze_module
from media_analysis.config import Settings
from media_analysis.decode import ProbedMedia
from media_analysis.errors import TIMEOUT, AnalyzeError
from media_analysis.features.yolox import Detection
from media_analysis.frame_access import BoundedFrameAccess
from media_analysis.frames import Rational
from media_analysis.jobs import Job
from media_analysis.models_manifest import ManifestState, ModelEntry
from media_analysis.runtime import RuntimeState
from media_analysis.schemas import AnalyzeRequest
from media_analysis.telemetry import JobTelemetry


@pytest.mark.unit
def test_ocr_and_audio_run_concurrently_in_bounded_pool(monkeypatch) -> None:
    barrier = threading.Barrier(2)

    def feature(name: str):
        def run(**_kwargs):
            barrier.wait(timeout=1)
            time.sleep(0.01)
            return ({name: {}}, {name: {"status": "completed"}}, [])

        return run

    monkeypatch.setattr(analyze_module, "_run_ocr_feature", feature("reservedRegions"))
    monkeypatch.setattr(analyze_module, "_run_audio_features", feature("waveform"))
    monkeypatch.setattr(
        analyze_module,
        "analyze_shots",
        lambda *_args, **_kwargs: [{"startFrame": 0, "endFrameExclusive": 2}],
    )
    request = AnalyzeRequest.model_validate(
        {
            "idempotencyKey": "parallel",
            "features": ["ocr", "waveform"],
            "source": {
                "signedGetUrl": "https://example.test/video.mp4",
                "expiresAt": "2099-01-01T00:00:00Z",
            },
        }
    )
    manifest = ManifestState(entries=(), ready=True, loaded=(), errors=())
    runtime = RuntimeState(
        manifest=manifest,
        ready=True,
        loaded_models=(),
        warmup_complete=True,
    )
    media = ProbedMedia(
        width=32,
        height=32,
        fps=Rational(30, 1),
        frame_count=2,
        duration=2 / 30,
        codec="h264",
        has_audio=False,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channel_count=None,
    )

    result = analyze_module._compute_cpu(
        request,
        media,
        Path("unused.mp4"),
        runtime,
        Settings(media_analysis_feature_workers=3),
        Job(idempotency_key="parallel", request_hash="hash"),
        time.monotonic() + 2,
        telemetry=JobTelemetry(),
        frame_access=object(),
    )

    assert result["capabilities"]["reservedRegions"]["status"] == "completed"
    assert result["capabilities"]["waveform"]["status"] == "completed"


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_subject_feature_uses_shared_frame_provider(tmp_path, monkeypatch) -> None:
    media = ProbedMedia(160, 120, Rational(30, 1), 12, 0.4, "h264", False, None, None, None)
    path = make_probe_mp4(tmp_path / "clip.mp4", frames=12)
    detector = ScriptedDetector({0: [Detection(32, 24, 160, 192, 0.95)]})
    entry = ModelEntry("yolox-tiny", "unused.onnx", "0" * 64, "Apache-2.0", stub=False)
    runtime = RuntimeState(
        ManifestState((entry,), True, (entry.name,), ()),
        True,
        (entry.name,),
        True,
        subject_detector=detector,
    )
    request = _request(["subjects"])
    with BoundedFrameAccess(path) as access:
        result = analyze_module._compute_cpu(
            request,
            media,
            path,
            runtime,
            Settings(),
            Job("subjects", "hash"),
            time.monotonic() + 5,
            telemetry=JobTelemetry(),
            frame_access=access,
        )
        assert result["capabilities"]["subjects"]["status"] == "completed"
        assert result["subjects"]
        assert access.unique_decodes > 0


def _request(features):
    return AnalyzeRequest.model_validate(
        {
            "idempotencyKey": "review-regression",
            "features": features,
            "source": {
                "signedGetUrl": "http://127.0.0.1/unused.mp4",
                "expiresAt": "2099-01-01T00:00:00Z",
            },
        }
    )


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_visual_failure_cancels_and_joins_workers_before_closing_media(tmp_path, monkeypatch):
    entered, finished = threading.Event(), threading.Event()
    closed_during_worker = []
    media = ProbedMedia(160, 120, Rational(30, 1), 12, 0.4, "h264", False, None, None, None)
    path = make_probe_mp4(tmp_path / "clip.mp4", frames=12)
    runtime = RuntimeState(ManifestState((), True, (), ()), True, (), True)
    job = Job("failure", "hash")

    def background(**kwargs):
        entered.set()
        try:
            assert kwargs["job"].cancel.wait(2), "foreground failure did not signal shutdown"
            closed_during_worker.append(kwargs["frame_access"]._closed)
            return {}, {}, []
        finally:
            finished.set()

    def fail_motion(*args, **kwargs):
        assert entered.wait(2)
        raise AnalyzeError(TIMEOUT, "visual deadline")

    monkeypatch.setattr(analyze_module, "_run_ocr_feature", background)
    monkeypatch.setattr(analyze_module, "analyze_motion", fail_motion)
    with pytest.raises(AnalyzeError) as exc:
        analyze_module._compute(
            _request(["ocr", "motion"]),
            media,
            path,
            runtime,
            Settings(media_analysis_feature_workers=2),
            job,
            time.monotonic() + 10,
            work_dir=tmp_path,
            telemetry=JobTelemetry(),
        )
    assert exc.value.code == TIMEOUT
    assert finished.is_set()
    assert closed_during_worker == [False]
