from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest
from tests.unit.ocr_fake_ort import FakeOrtSession

from media_analysis.decode import ProbedMedia
from media_analysis.features.ocr import (
    OCR_ERROR_DECODE,
    OCR_ERROR_INFERENCE,
    OCR_WARNING_PARTIAL_DECODE,
    OcrAnalyzeResult,
    VideoFrameReader,
    analyze_ocr,
    polygons_to_detections,
)
from media_analysis.features.ppocr import DetPolygon, DetPreprocessMeta, postprocess_db_map
from media_analysis.frames import Rational


def _media(*, frames: int = 90) -> ProbedMedia:
    return ProbedMedia(
        width=320,
        height=240,
        fps=Rational(30, 1),
        frame_count=frames,
        duration=frames / 30.0,
        codec="h264",
        has_audio=False,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channel_count=None,
    )


def _prob_map_with_top_box() -> np.ndarray:
    pytest.importorskip("pyclipper")
    meta = DetPreprocessMeta(240, 320, 1.0, 240, 320)
    prob = np.zeros((240, 320), dtype=np.float32)
    prob[8:32, 40:280] = 0.95
    polys = postprocess_db_map(prob, meta, thresh=0.5, box_thresh=0.5, min_size=2)
    assert polys
    tensor = np.zeros((1, 1, 240, 320), dtype=np.float32)
    tensor[0, 0, 8:32, 40:280] = 0.95
    return tensor


@pytest.mark.unit
def test_analyze_ocr_returns_empty_list_when_no_roi_detections() -> None:
    prob = np.zeros((1, 1, 64, 64), dtype=np.float32)
    session = FakeOrtSession(prob)

    def frames(_path: Path, _index: int) -> np.ndarray:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    result = analyze_ocr(
        Path("unused.mp4"),
        _media(),
        shots=[{"startFrame": 0, "endFrameExclusive": 90}],
        model_path=Path("unused.onnx"),
        session=session,
        frame_provider=frames,
    )
    assert result.status == "completed"
    assert result.regions == []


@pytest.mark.unit
def test_analyze_ocr_propagates_cancellation_between_samples() -> None:
    class Cancelled(Exception):
        pass

    def cancel() -> None:
        raise Cancelled

    with pytest.raises(Cancelled):
        analyze_ocr(
            Path("unused.mp4"),
            _media(),
            model_path=Path("unused.onnx"),
            session=FakeOrtSession(np.zeros((1, 1, 8, 8), dtype=np.float32)),
            frame_provider=lambda _path, _index: np.zeros((64, 64, 3), dtype=np.uint8),
            cancel_check=cancel,
        )


@pytest.mark.unit
def test_analyze_ocr_merges_injected_detections_into_reserved_region() -> None:
    pytest.importorskip("pyclipper")
    session = FakeOrtSession(_prob_map_with_top_box())

    def frames(_path: Path, _index: int) -> np.ndarray:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    result = analyze_ocr(
        Path("unused.mp4"),
        _media(),
        shots=[{"startFrame": 0, "endFrameExclusive": 90}],
        model_path=Path("unused.onnx"),
        session=session,
        frame_provider=frames,
    )
    assert result.status == "completed"
    assert len(result.regions) >= 1
    region = result.regions[0]
    assert region["kind"] in {"ocr_text", "persistent_overlay"}
    sample = region["samples"][0]
    assert sample["scoreType"] == "raw_model"
    assert "polygon" in sample and "box" in sample
    assert all(0.0 <= point["x"] <= 1.0 for point in sample["polygon"])


@pytest.mark.unit
def test_analyze_ocr_model_failure_returns_stable_error_code() -> None:
    session = FakeOrtSession(np.zeros((1, 1, 8, 8), dtype=np.float32), fail_on_run=True)

    def frames(_path: Path, _index: int) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    result = analyze_ocr(
        Path("unused.mp4"),
        _media(frames=30),
        model_path=Path("unused.onnx"),
        session=session,
        frame_provider=frames,
    )
    assert isinstance(result, OcrAnalyzeResult)
    assert result.status == "failed"
    assert result.error == OCR_ERROR_INFERENCE


@pytest.mark.unit
def test_polygons_to_detections_filters_middle_band() -> None:
    top = DetPolygon(points=((0.1, 0.05), (0.3, 0.05), (0.3, 0.1), (0.1, 0.1)), score=0.9)
    middle = DetPolygon(points=((0.1, 0.4), (0.3, 0.4), (0.3, 0.5), (0.1, 0.5)), score=0.9)
    detections = polygons_to_detections([top, middle], source_frame=0)
    assert len(detections) == 1
    assert detections[0].box["y"] <= 0.1


@pytest.mark.unit
def test_analyze_ocr_all_decode_failures_return_failed() -> None:
    session = FakeOrtSession(np.zeros((1, 1, 8, 8), dtype=np.float32))

    def frames(_path: Path, _index: int) -> None:
        return None

    result = analyze_ocr(
        Path("unused.mp4"),
        _media(frames=30),
        model_path=Path("unused.onnx"),
        session=session,
        frame_provider=frames,
    )
    assert result.status == "failed"
    assert result.error == OCR_ERROR_DECODE


@pytest.mark.unit
def test_analyze_ocr_partial_decode_emits_internal_warning() -> None:
    pytest.importorskip("pyclipper")
    session = FakeOrtSession(_prob_map_with_top_box())

    def frames(_path: Path, index: int) -> np.ndarray | None:
        if index == 0:
            return None
        return np.zeros((240, 320, 3), dtype=np.uint8)

    result = analyze_ocr(
        Path("unused.mp4"),
        _media(),
        shots=[{"startFrame": 0, "endFrameExclusive": 90}],
        model_path=Path("unused.onnx"),
        session=session,
        frame_provider=frames,
    )
    assert result.status == "completed"
    assert OCR_WARNING_PARTIAL_DECODE in result.warning_codes


class _FakeCapture:
    instances: ClassVar[list["_FakeCapture"]] = []

    def __init__(self, _path: str) -> None:
        self.released = False
        type(self).instances.append(self)

    def isOpened(self) -> bool:
        return True

    def set(self, *_args: object, **_kwargs: object) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return True, np.zeros((240, 320, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


@pytest.mark.unit
def test_default_frame_reader_opens_capture_once_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCapture.instances = []
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    session = FakeOrtSession(np.zeros((1, 1, 8, 8), dtype=np.float32))

    result = analyze_ocr(
        Path("clip.mp4"),
        _media(frames=30),
        model_path=Path("unused.onnx"),
        session=session,
    )
    assert result.status == "completed"
    assert len(_FakeCapture.instances) == 1
    assert _FakeCapture.instances[0].released


@pytest.mark.unit
def test_video_frame_reader_closes_on_context_exit() -> None:
    reader = VideoFrameReader(Path("missing.mp4"))
    reader.close()
    assert reader.read(0) is None
