import numpy as np
import pytest

from media_analysis.decode import ProbedMedia
from media_analysis.features.faces import DictFrameReader, VideoFrameReader, analyze_faces
from media_analysis.features.yunet import FakeYuNetDetector, RawFaceDetection
from media_analysis.frames import Rational


def _media(frame_count: int = 12) -> ProbedMedia:
    return ProbedMedia(
        width=320,
        height=240,
        fps=Rational(30, 1),
        frame_count=frame_count,
        duration=frame_count / 30.0,
        codec="h264",
        has_audio=False,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channel_count=None,
    )


def _subject(track_id: str, frame: int, box: dict[str, float]) -> dict:
    return {
        "trackId": track_id,
        "type": "person",
        "scoreType": "raw_model",
        "samples": [
            {
                "sourceFrame": frame,
                "presentationTimeSecApprox": frame / 30.0,
                "box": box,
                "sampleKind": "detected",
                "scoreType": "raw_model",
                "occluded": False,
                "detectorScore": 0.9,
            }
        ],
    }


@pytest.mark.unit
def test_face_associates_to_containing_subject_track() -> None:
    shots = [{"startFrame": 0, "endFrameExclusive": 6}]
    reader = DictFrameReader(
        frames={
            0: np.zeros((240, 320, 3), dtype=np.uint8),
            5: np.zeros((240, 320, 3), dtype=np.uint8),
        }
    )
    face_box = RawFaceDetection(x=80, y=60, width=40, height=40, score=0.93)
    detector = FakeYuNetDetector(sequence=[[face_box], [face_box]])

    subjects = [
        _subject(
            "person-000-001",
            0,
            {"x": 0.2, "y": 0.2, "width": 0.5, "height": 0.6},
        ),
        _subject(
            "person-000-001",
            5,
            {"x": 0.2, "y": 0.2, "width": 0.5, "height": 0.6},
        ),
    ]

    faces = analyze_faces(
        path=__file__,
        media=_media(),
        shots=shots,
        subjects=subjects,
        detector=detector,
        frame_reader=reader,
        max_sample_gap_ms=200,
    )

    assert len(faces) == 1
    assert faces[0]["subjectTrackId"] == "person-000-001"


@pytest.mark.unit
def test_face_without_subject_overlap_has_no_subject_track_id() -> None:
    shots = [{"startFrame": 0, "endFrameExclusive": 6}]
    reader = DictFrameReader(
        frames={
            0: np.zeros((240, 320, 3), dtype=np.uint8),
            5: np.zeros((240, 320, 3), dtype=np.uint8),
        }
    )
    face_box = RawFaceDetection(x=10, y=10, width=20, height=20, score=0.93)
    detector = FakeYuNetDetector(sequence=[[face_box], [face_box]])

    subjects = [
        _subject(
            "person-000-002",
            0,
            {"x": 0.7, "y": 0.7, "width": 0.2, "height": 0.2},
        ),
        _subject(
            "person-000-002",
            5,
            {"x": 0.7, "y": 0.7, "width": 0.2, "height": 0.2},
        ),
    ]

    faces = analyze_faces(
        path=__file__,
        media=_media(),
        shots=shots,
        subjects=subjects,
        detector=detector,
        frame_reader=reader,
        max_sample_gap_ms=200,
    )

    assert len(faces) == 1
    assert "subjectTrackId" not in faces[0]


@pytest.mark.unit
def test_subject_association_requires_temporal_overlap() -> None:
    shots = [{"startFrame": 0, "endFrameExclusive": 12}]
    reader = DictFrameReader(
        frames={
            0: np.zeros((240, 320, 3), dtype=np.uint8),
            6: np.zeros((240, 320, 3), dtype=np.uint8),
            11: np.zeros((240, 320, 3), dtype=np.uint8),
        }
    )
    face_box = RawFaceDetection(x=80, y=60, width=40, height=40, score=0.93)
    detector = FakeYuNetDetector(sequence=[[face_box], [face_box], [face_box]])

    subjects = [
        _subject("person-000-003", 0, {"x": 0.2, "y": 0.2, "width": 0.5, "height": 0.6}),
    ]

    faces = analyze_faces(
        path=__file__,
        media=_media(frame_count=12),
        shots=shots,
        subjects=subjects,
        detector=detector,
        frame_reader=reader,
        max_sample_gap_ms=200,
    )

    assert len(faces) == 1
    assert "subjectTrackId" not in faces[0]


@pytest.mark.unit
def test_injected_frame_reader_is_not_closed_by_analyze_faces() -> None:
    class _Reader:
        def __init__(self) -> None:
            self.closed = False

        def read(self, frame_index: int) -> np.ndarray | None:
            return np.zeros((240, 320, 3), dtype=np.uint8)

        def close(self) -> None:
            self.closed = True

    reader = _Reader()
    detector = FakeYuNetDetector(sequence=[[]])
    analyze_faces(
        path=__file__,
        media=_media(frame_count=6),
        shots=[{"startFrame": 0, "endFrameExclusive": 6}],
        detector=detector,
        frame_reader=reader,
    )
    assert reader.closed is False


@pytest.mark.unit
def test_video_frame_reader_releases_capture_on_close() -> None:
    released = {"called": False}

    class _FakeCapture:
        def isOpened(self) -> bool:
            return True

        def set(self, *_args: object) -> None:
            return None

        def read(self) -> tuple[bool, None]:
            return False, None

        def release(self) -> None:
            released["called"] = True

    reader = VideoFrameReader.__new__(VideoFrameReader)
    reader.path = __file__
    reader.width = 320
    reader.height = 240
    reader._capture = _FakeCapture()

    reader.close()
    assert released["called"] is True
    assert reader._capture is None
