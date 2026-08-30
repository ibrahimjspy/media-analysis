"""Unit tests for subject pipeline orchestration and provenance metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from media_analysis.decode import ProbedMedia
from media_analysis.features.subjects import (
    SAMPLING_POLICY_VERSION,
    TRACKER_VERSION,
    analyze_subjects,
    count_distinct_persons,
    subject_model_provenance,
)
from media_analysis.features.yolox import Detection, LetterboxMeta
from media_analysis.frames import Rational
from media_analysis.models_manifest import ModelEntry


class ScriptedDetector:
    def __init__(self, schedule: dict[int, list[Detection]]) -> None:
        self.schedule = schedule

    def detect(self, image_bgr: np.ndarray) -> tuple[list[Detection], LetterboxMeta]:
        frame = int(getattr(self, "_frame", 0))
        return self.schedule.get(frame, []), LetterboxMeta(
            scale=1.0,
            orig_width=320,
            orig_height=240,
        )

    def detect_normalized(self, image_bgr: np.ndarray):
        detections, meta = self.detect(image_bgr)
        boxes = [
            {
                "x": det.x1 / 320,
                "y": det.y1 / 240,
                "width": (det.x2 - det.x1) / 320,
                "height": (det.y2 - det.y1) / 240,
            }
            for det in detections
        ]
        return list(zip(detections, boxes, strict=True)), meta


@pytest.fixture
def probed_media() -> ProbedMedia:
    return ProbedMedia(
        width=320,
        height=240,
        fps=Rational(30, 1),
        frame_count=60,
        duration=2.0,
        codec="h264",
        has_audio=False,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channel_count=None,
    )


@pytest.mark.unit
def test_analyze_subjects_open_failure_raises(
    tmp_path: Path,
    probed_media: ProbedMedia,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"not-a-real-video")

    class EmptyCapture:
        def isOpened(self) -> bool:
            return False

        def read(self):
            return False, None

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        "media_analysis.features.subjects.cv2.VideoCapture",
        lambda _: EmptyCapture(),
    )
    detector = ScriptedDetector({})
    with pytest.raises(RuntimeError, match="open video for subject analysis"):
        analyze_subjects(path, probed_media, detector=detector)  # type: ignore[arg-type]


@pytest.mark.unit
def test_analyze_subjects_all_sample_decodes_fail(
    tmp_path: Path,
    probed_media: ProbedMedia,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenCapture:
        def isOpened(self) -> bool:
            return True

        def set(self, *_args) -> None:
            return None

        def read(self):
            return False, None

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        "media_analysis.features.subjects.cv2.VideoCapture",
        lambda _: BrokenCapture(),
    )
    with pytest.raises(RuntimeError, match="decode sampled subject frames"):
        analyze_subjects(
            tmp_path / "broken.mp4",
            probed_media,
            detector=ScriptedDetector({}),  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_track_ids_reset_at_shot_boundaries(
    tmp_path: Path,
    probed_media: ProbedMedia,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "clip.mp4"
    path.touch()
    schedule = {
        0: [Detection(10, 10, 60, 120, 0.9)],
        30: [Detection(12, 12, 62, 122, 0.92)],
    }
    detector = ScriptedDetector(schedule)

    class FakeCapture:
        def __init__(self, *_args, **_kwargs) -> None:
            self._index = 0

        def isOpened(self) -> bool:
            return True

        def set(self, _prop, frame_index: float) -> None:
            self._index = int(frame_index)
            detector._frame = self._index

        def read(self):
            return True, np.zeros((240, 320, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

    monkeypatch.setattr("media_analysis.features.subjects.cv2.VideoCapture", FakeCapture)
    shots = [
        {"startFrame": 0, "endFrameExclusive": 30},
        {"startFrame": 30, "endFrameExclusive": 60},
    ]
    subjects = analyze_subjects(path, probed_media, shots=shots, detector=detector)  # type: ignore[arg-type]
    track_ids = {item["trackId"] for item in subjects}
    assert "0-1" in track_ids
    assert "1-1" in track_ids


@pytest.mark.unit
def test_subject_samples_are_normalized(
    tmp_path: Path,
    probed_media: ProbedMedia,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "clip.mp4"
    path.touch()
    detector = ScriptedDetector({0: [Detection(32, 24, 160, 192, 0.95)]})

    class FakeCapture:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop, frame_index: float) -> None:
            detector._frame = int(frame_index)

        def read(self):
            return True, np.zeros((240, 320, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

    monkeypatch.setattr("media_analysis.features.subjects.cv2.VideoCapture", FakeCapture)
    subjects = analyze_subjects(path, probed_media, detector=detector)  # type: ignore[arg-type]
    box = subjects[0]["samples"][0]["box"]
    assert 0.0 <= box["x"] <= 1.0
    assert 0.0 <= box["y"] <= 1.0
    assert 0.0 <= box["width"] <= 1.0
    assert 0.0 <= box["height"] <= 1.0
    assert subjects[0]["samples"][0]["sampleKind"] == "detected"


@pytest.mark.unit
def test_subject_model_provenance_metadata() -> None:
    entry = ModelEntry(
        name="yolox-tiny",
        file="yolox_tiny.onnx",
        sha256="abc",
        license="Apache-2.0",
        stub=False,
        version="0.1.1",
    )
    provenance = subject_model_provenance(entry)
    assert provenance is not None
    assert provenance["name"] == "yolox-tiny"
    assert provenance["runtimeProvider"] == "onnxruntime-cpu"
    assert provenance["preprocessingVersion"] == "letterbox-416"
    assert provenance["artifactSha256"] == "abc"

    stub = ModelEntry(
        name="yolox-tiny",
        file="yolox_tiny.onnx",
        sha256="def",
        license="Apache-2.0",
        stub=True,
        version="stub",
    )
    assert subject_model_provenance(stub)["runtimeProvider"] == "stub"


@pytest.mark.unit
def test_version_constants_match_analyze_contract() -> None:
    assert TRACKER_VERSION == "bytetrack-shot-reset-1.0.0"
    assert SAMPLING_POLICY_VERSION == "max-gap-200ms-1.0.0"


@pytest.mark.unit
def test_count_distinct_persons() -> None:
    subjects = [{"trackId": "0-1"}, {"trackId": "0-2"}]
    assert count_distinct_persons(subjects) == 2
