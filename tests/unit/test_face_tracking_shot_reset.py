import numpy as np
import pytest

from media_analysis.decode import ProbedMedia
from media_analysis.features.faces import DictFrameReader, analyze_faces
from media_analysis.features.yunet import FakeYuNetDetector, RawFaceDetection
from media_analysis.frames import Rational


def _media(*, width: int = 320, height: int = 240, frame_count: int = 60) -> ProbedMedia:
    return ProbedMedia(
        width=width,
        height=height,
        fps=Rational(30, 1),
        frame_count=frame_count,
        duration=frame_count / 30.0,
        codec="h264",
        has_audio=False,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channel_count=None,
    )


def _blank_frames(indices: list[int], *, width: int = 320, height: int = 240) -> DictFrameReader:
    return DictFrameReader(
        frames={
            index: np.zeros((height, width, 3), dtype=np.uint8)
            for index in indices
        }
    )


@pytest.mark.unit
def test_no_faces_returns_completed_empty_list() -> None:
    shots = [{"startFrame": 0, "endFrameExclusive": 30}]
    reader = _blank_frames([0, 6, 12, 18, 24, 29])
    detector = FakeYuNetDetector(sequence=[[]] * 6)

    faces = analyze_faces(
        path=__file__,
        media=_media(frame_count=30),
        shots=shots,
        detector=detector,
        frame_reader=reader,
        max_sample_gap_ms=200,
    )

    assert faces == []


@pytest.mark.unit
def test_face_analysis_propagates_cancellation_between_samples() -> None:
    class Cancelled(Exception):
        pass

    def cancel() -> None:
        raise Cancelled

    with pytest.raises(Cancelled):
        analyze_faces(
            path=__file__,
            media=_media(frame_count=30),
            shots=[{"startFrame": 0, "endFrameExclusive": 30}],
            detector=FakeYuNetDetector(sequence=[]),
            frame_reader=_blank_frames([0]),
            cancel_check=cancel,
        )


@pytest.mark.unit
def test_face_analysis_all_decode_failures_raise() -> None:
    with pytest.raises(RuntimeError, match="decode sampled face frames"):
        analyze_faces(
            path=__file__,
            media=_media(frame_count=30),
            shots=[{"startFrame": 0, "endFrameExclusive": 30}],
            detector=FakeYuNetDetector(sequence=[]),
            frame_reader=_blank_frames([]),
        )


@pytest.mark.unit
def test_face_tracks_reset_at_shot_cuts() -> None:
    shots = [
        {"startFrame": 0, "endFrameExclusive": 15},
        {"startFrame": 15, "endFrameExclusive": 30},
    ]
    reader = _blank_frames([0, 14, 15, 29])
    face = RawFaceDetection(x=40, y=30, width=50, height=60, score=0.95)
    detector = FakeYuNetDetector(
        sequence=[
            [face],
            [face],
            [face],
            [face],
        ]
    )

    faces = analyze_faces(
        path=__file__,
        media=_media(frame_count=30),
        shots=shots,
        detector=detector,
        frame_reader=reader,
        max_sample_gap_ms=500,
    )

    assert len(faces) == 2
    assert faces[0]["trackId"] == "face-000-001"
    assert faces[1]["trackId"] == "face-001-001"


@pytest.mark.unit
def test_track_samples_use_detected_and_interpolated_between_real_detections() -> None:
    shots = [{"startFrame": 0, "endFrameExclusive": 12}]
    reader = _blank_frames([0, 6, 11])
    first = RawFaceDetection(x=10, y=10, width=20, height=20, score=0.92)
    detector = FakeYuNetDetector(sequence=[[first], [first], [first]])

    faces = analyze_faces(
        path=__file__,
        media=_media(frame_count=12),
        shots=shots,
        detector=detector,
        frame_reader=reader,
        max_sample_gap_ms=200,
    )

    assert len(faces) == 1
    kinds = {sample["sampleKind"] for sample in faces[0]["samples"]}
    assert kinds == {"detected", "interpolated"}
    assert "tracked" not in kinds


@pytest.mark.unit
def test_missed_sample_frame_finishes_track_and_allows_new_track() -> None:
    shots = [{"startFrame": 0, "endFrameExclusive": 12}]
    reader = _blank_frames([0, 6, 11])
    face = RawFaceDetection(x=10, y=10, width=20, height=20, score=0.92)
    detector = FakeYuNetDetector(sequence=[[face], [], [face]])

    faces = analyze_faces(
        path=__file__,
        media=_media(frame_count=12),
        shots=shots,
        detector=detector,
        frame_reader=reader,
        max_sample_gap_ms=200,
    )

    assert len(faces) == 2
    assert faces[0]["trackId"] == "face-000-001"
    assert faces[1]["trackId"] == "face-000-002"
    assert all(sample["sampleKind"] == "detected" for sample in faces[0]["samples"])
    assert all(sample["sampleKind"] == "detected" for sample in faces[1]["samples"])


@pytest.mark.unit
def test_matched_detection_records_actual_iou_as_association_score() -> None:
    shots = [{"startFrame": 0, "endFrameExclusive": 12}]
    reader = _blank_frames([0, 6])
    first = RawFaceDetection(x=10, y=10, width=20, height=20, score=0.92)
    shifted = RawFaceDetection(x=12, y=10, width=20, height=20, score=0.88)
    detector = FakeYuNetDetector(sequence=[[first], [shifted]])

    faces = analyze_faces(
        path=__file__,
        media=_media(frame_count=12),
        shots=shots,
        detector=detector,
        frame_reader=reader,
        max_sample_gap_ms=200,
    )

    detected = [s for s in faces[0]["samples"] if s["sampleKind"] == "detected"]
    assert len(detected) == 2
    assert detected[0].get("associationScore") is None
    assert detected[1]["associationScore"] == pytest.approx(0.818181818, rel=1e-4)


@pytest.mark.unit
def test_face_track_sample_shape_matches_contract() -> None:
    shots = [{"startFrame": 0, "endFrameExclusive": 6}]
    reader = _blank_frames([0, 5])
    detection = RawFaceDetection(x=16, y=12, width=32, height=40, score=0.91)
    detector = FakeYuNetDetector(sequence=[[detection], [detection]])

    faces = analyze_faces(
        path=__file__,
        media=_media(frame_count=6),
        shots=shots,
        detector=detector,
        frame_reader=reader,
        max_sample_gap_ms=200,
    )

    sample = faces[0]["samples"][0]
    assert set(sample) >= {
        "sourceFrame",
        "presentationTimeSecApprox",
        "box",
        "sampleKind",
        "scoreType",
        "occluded",
    }
    assert sample["sourceFrame"] == 0
    assert sample["presentationTimeSecApprox"] == pytest.approx(0.0)
    assert sample["sampleKind"] == "detected"
    assert sample["scoreType"] == "raw_model"
    assert sample["occluded"] is False
    assert sample["detectorScore"] == pytest.approx(0.91)

    interpolated = next(s for s in faces[0]["samples"] if s["sampleKind"] == "interpolated")
    assert interpolated["scoreType"] == "heuristic"
    assert interpolated["occluded"] is False
    assert "detectorScore" not in interpolated


@pytest.mark.unit
def test_non_uniform_analysis_resolution_is_rejected() -> None:
    shots = [{"startFrame": 0, "endFrameExclusive": 6}]
    reader = _blank_frames([0])
    detector = FakeYuNetDetector(sequence=[[]])

    with pytest.raises(ValueError, match="uniform scale"):
        analyze_faces(
            path=__file__,
            media=_media(width=320, height=240),
            shots=shots,
            detector=detector,
            frame_reader=reader,
            analysis_width=160,
            analysis_height=200,
        )


@pytest.mark.unit
def test_uniform_analysis_resolution_is_accepted() -> None:
    shots = [{"startFrame": 0, "endFrameExclusive": 6}]
    reader = _blank_frames([0, 5], width=160, height=120)
    detection = RawFaceDetection(x=16, y=12, width=32, height=40, score=0.91)
    detector = FakeYuNetDetector(sequence=[[detection], [detection]])

    faces = analyze_faces(
        path=__file__,
        media=_media(width=320, height=240, frame_count=6),
        shots=shots,
        detector=detector,
        frame_reader=reader,
        analysis_width=160,
        analysis_height=120,
        max_sample_gap_ms=200,
    )

    assert len(faces) == 1
    assert faces[0]["samples"][0]["box"]["x"] == pytest.approx(0.1)
    assert faces[0]["samples"][0]["box"]["y"] == pytest.approx(0.1)
