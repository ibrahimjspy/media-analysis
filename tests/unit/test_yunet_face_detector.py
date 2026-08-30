import cv2
import numpy as np
import pytest

from media_analysis.features.yunet import (
    DEFAULT_SCORE_THRESHOLD,
    FakeYuNetDetector,
    RawFaceDetection,
    clip_normalized_box,
    load_yunet_detector,
    normalize_face_box,
)


@pytest.mark.unit
def test_load_yunet_requires_existing_model_file(tmp_path) -> None:
    missing = tmp_path / "face_detection_yunet_2023mar.onnx"
    with pytest.raises(FileNotFoundError, match="YuNet model not found"):
        load_yunet_detector(missing)


@pytest.mark.unit
def test_load_yunet_uses_face_detector_yn_create_contract(tmp_path, monkeypatch) -> None:
    model = tmp_path / "face_detection_yunet_2023mar.onnx"
    model.write_bytes(b"not-a-real-onnx")

    calls: list[tuple] = []

    class _FakeBackend:
        pass

    def fake_create(*args):
        calls.append(args)
        return _FakeBackend()

    monkeypatch.setattr(cv2.FaceDetectorYN, "create", fake_create)

    detector = load_yunet_detector(
        model,
        input_size=(640, 480),
        score_threshold=0.55,
        nms_threshold=0.4,
        top_k=100,
    )
    assert isinstance(detector, object)
    assert calls == [
        (
            str(model),
            "",
            (640, 480),
            0.55,
            0.4,
            100,
        )
    ]


@pytest.mark.unit
def test_fake_yunet_updates_input_size_for_variable_frames() -> None:
    detector = FakeYuNetDetector()
    small = np.zeros((120, 160, 3), dtype=np.uint8)
    large = np.zeros((240, 320, 3), dtype=np.uint8)

    detector.detect(small)
    detector.detect(large)

    assert detector.input_sizes == [(160, 120), (320, 240)]


@pytest.mark.unit
def test_fake_yunet_applies_score_threshold() -> None:
    detector = FakeYuNetDetector(
        score_threshold=0.7,
        sequence=[
            [
                RawFaceDetection(0, 0, 10, 10, 0.69),
                RawFaceDetection(20, 20, 10, 10, 0.71),
            ]
        ],
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.71)


@pytest.mark.unit
def test_normalize_face_box_clips_to_unit_square() -> None:
    detection = RawFaceDetection(x=-10, y=90, width=200, height=200, score=0.9)
    box = normalize_face_box(detection, frame_width=100, frame_height=100)
    assert box["x"] == 0.0
    assert box["y"] == pytest.approx(0.9)
    assert box["width"] == 1.0
    assert box["height"] == pytest.approx(0.1)


@pytest.mark.unit
def test_clip_normalized_box_shrinks_partially_out_of_frame_negative_origin() -> None:
    clipped = clip_normalized_box({"x": -0.2, "y": 0.1, "width": 0.5, "height": 0.3})
    assert clipped == {
        "x": 0.0,
        "y": pytest.approx(0.1),
        "width": pytest.approx(0.3),
        "height": pytest.approx(0.3),
    }


@pytest.mark.unit
def test_clip_normalized_box_shrinks_partially_out_of_frame_positive_overflow() -> None:
    clipped = clip_normalized_box({"x": 0.9, "y": 0.8, "width": 0.5, "height": 0.5})
    assert clipped["width"] == pytest.approx(0.1)
    assert clipped["height"] == pytest.approx(0.2)


@pytest.mark.unit
def test_default_score_threshold_is_documented() -> None:
    assert DEFAULT_SCORE_THRESHOLD == 0.6
