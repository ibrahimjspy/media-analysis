from __future__ import annotations

from pathlib import Path

from media_analysis.decode import ProbedMedia
from media_analysis.frames import duration_sec

SHOT_ANALYZER_VERSION = "pyscenedetect-adaptive-1.0.0"
SHOT_CLASSIFIER_VERSION = "heuristic-person-ratio-1.0.0"


def _classify(person_count: int | None, duration: float) -> tuple[str, float, str]:
    if person_count is None:
        return "other", 0.0, "heuristic"
    if person_count == 0:
        return "no_person", 1.0, "heuristic"
    if person_count >= 3:
        return "multiple_people", 0.7, "heuristic"
    if person_count == 2:
        return "multiple_people", 0.6, "heuristic"
    if duration >= 2.0:
        return "talking_head", 0.4, "heuristic"
    return "other", 0.3, "heuristic"


def analyze_shots(
    path: Path,
    media: ProbedMedia,
    *,
    person_count: int | None = None,
) -> list[dict]:
    boundaries = _detect_boundaries(path, media)
    shots: list[dict] = []
    for index, (start, end, kind, score) in enumerate(boundaries):
        length = duration_sec(end - start, media.fps)
        label, class_score, score_type = _classify(person_count, length)
        if index == 0 and kind == "hard_cut":
            kind = "start"
        shots.append(
            {
                "startFrame": start,
                "endFrameExclusive": end,
                "startSecApprox": duration_sec(start, media.fps),
                "endSecApprox": duration_sec(end, media.fps),
                "boundaryKind": kind,
                "boundaryScore": score,
                "boundaryScoreType": "heuristic",
                "classification": label,
                "classificationScore": class_score,
                "scoreType": score_type,
            }
        )
    if shots:
        shots[-1]["boundaryKind"] = "end" if len(shots) == 1 else shots[-1]["boundaryKind"]
    return shots


def classify_shots_with_subjects(
    shots: list[dict],
    subjects: list[dict],
    media: ProbedMedia,
) -> None:
    """Update shot-local heuristic labels from measured subject tracks."""
    for shot in shots:
        start = int(shot["startFrame"])
        end = int(shot["endFrameExclusive"])
        person_count = sum(
            1
            for subject in subjects
            if any(
                start <= int(sample["sourceFrame"]) < end
                for sample in subject.get("samples", [])
            )
        )
        duration = duration_sec(end - start, media.fps)
        label, score, score_type = _classify(person_count, duration)
        shot["classification"] = label
        shot["classificationScore"] = score
        shot["scoreType"] = score_type


def _detect_boundaries(path: Path, media: ProbedMedia) -> list[tuple[int, int, str, float]]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector, ThresholdDetector
    except Exception:
        return [(0, media.frame_count, "start", 1.0)]

    video = open_video(str(path))
    manager = SceneManager()
    manager.add_detector(AdaptiveDetector())
    manager.add_detector(ThresholdDetector())
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    if not scenes:
        return [(0, media.frame_count, "start", 1.0)]

    out: list[tuple[int, int, str, float]] = []
    for start, end in scenes:
        start_frame = int(start.get_frames())
        end_frame = int(end.get_frames())
        end_frame = min(end_frame, media.frame_count)
        if end_frame <= start_frame:
            continue
        kind = "hard_cut" if out else "start"
        out.append((start_frame, end_frame, kind, 0.8))
    if not out:
        return [(0, media.frame_count, "start", 1.0)]
    if out[-1][1] < media.frame_count:
        last = out[-1]
        out[-1] = (last[0], media.frame_count, last[2], last[3])
    return out
