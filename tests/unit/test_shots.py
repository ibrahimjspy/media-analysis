from __future__ import annotations

from pathlib import Path

import pytest

import media_analysis.features.shots as shots_module
from media_analysis.decode import ProbedMedia
from media_analysis.features.shots import analyze_shots, classify_shots_with_subjects
from media_analysis.frames import Rational


@pytest.mark.unit
def test_shot_classification_uses_shot_local_subject_count() -> None:
    media = ProbedMedia(
        width=320,
        height=240,
        fps=Rational(30, 1),
        frame_count=90,
        duration=3,
        codec="h264",
        has_audio=False,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channel_count=None,
    )
    shots = [
        {"startFrame": 0, "endFrameExclusive": 30},
        {"startFrame": 30, "endFrameExclusive": 90},
    ]
    subjects = [
        {"samples": [{"sourceFrame": 35}]},
        {"samples": [{"sourceFrame": 40}]},
    ]

    classify_shots_with_subjects(shots, subjects, media)

    assert shots[0]["classification"] == "no_person"
    assert shots[0]["classificationScore"] == 1.0
    assert shots[1]["classification"] == "multiple_people"
    assert shots[1]["classificationScore"] == 0.6


@pytest.mark.unit
def test_shot_analysis_checks_cancellation_during_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = ProbedMedia(
        width=320,
        height=240,
        fps=Rational(30, 1),
        frame_count=30,
        duration=1,
        codec="h264",
        has_audio=False,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channel_count=None,
    )
    checks = 0

    def check() -> None:
        nonlocal checks
        checks += 1

    def detect(_path, _media, *, cancel_check=None):
        assert cancel_check is check
        cancel_check()
        return [(0, 30, "start", 1.0)]

    monkeypatch.setattr(shots_module, "_detect_boundaries", detect)
    analyze_shots(Path("clip.mp4"), media, cancel_check=check)
    assert checks >= 4
