"""Shared helpers for deterministic v1.2 visual measurements."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from media_analysis.errors import INVALID_REQUEST, AnalyzeError
from media_analysis.schemas import PriorFactsIn, ShotAnalysisIn

FrameProvider = Callable[[int], np.ndarray | None]

PriorFactsInput = PriorFactsIn | dict[str, Any] | None
ShotsInput = list[dict[str, Any]] | list[ShotAnalysisIn] | None


class ReadableFrameProvider(Protocol):
    def read(self, frame_index: int) -> np.ndarray | None: ...


def normalize_prior_facts(prior_facts: PriorFactsInput) -> dict[str, Any] | None:
    """Accept typed `PriorFactsIn` from the parent or a validated dict (`model_dump`)."""
    if prior_facts is None:
        return None
    if isinstance(prior_facts, PriorFactsIn):
        return prior_facts.model_dump(mode="python", by_alias=True)
    if isinstance(prior_facts, dict):
        return prior_facts
    raise TypeError("prior_facts must be PriorFactsIn, dict, or None")


def normalize_shots(shots: ShotsInput) -> list[dict[str, Any]]:
    if shots is None:
        return []
    normalized: list[dict[str, Any]] = []
    for shot in shots:
        if isinstance(shot, ShotAnalysisIn):
            normalized.append(shot.model_dump(mode="python", by_alias=True))
        elif isinstance(shot, dict):
            normalized.append(shot)
        else:
            raise TypeError("shots entries must be ShotAnalysisIn or dict")
    return normalized


def shot_ranges(
    shots: list[dict[str, Any]] | None,
    frame_count: int,
) -> list[tuple[int, int]]:
    if not shots:
        return [(0, frame_count)]
    ranges: list[tuple[int, int]] = []
    for shot in shots:
        start = int(shot["startFrame"])
        end = int(shot["endFrameExclusive"])
        if end > start:
            ranges.append((start, min(end, frame_count)))
    if not ranges and frame_count > 0:
        return [(0, frame_count)]
    return ranges


def shots_from_prior_facts(
    prior_facts: PriorFactsInput,
    *,
    frame_count: int,
) -> list[dict[str, Any]] | None:
    """Return canonical shot facts supplied by the caller for fill-only requests."""
    payload = normalize_prior_facts(prior_facts)
    if payload is None:
        return None
    raw = payload.get("shots")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("priorFacts.shots must be a list")
    shots: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, ShotAnalysisIn):
            item = item.model_dump(mode="python", by_alias=True)
        if not isinstance(item, dict):
            raise ValueError("priorFacts.shots entries must be objects")
        start = int(item["startFrame"])
        end = int(item["endFrameExclusive"])
        if end <= start:
            continue
        if start < 0 or end > frame_count:
            raise ValueError("priorFacts.shots ranges must lie inside canonical frameCount")
        shots.append(item)
    if not shots:
        raise ValueError("priorFacts.shots must contain at least one valid range")
    return shots


def resolve_shots(
    *,
    prior_facts: PriorFactsInput = None,
    supplied_shots: ShotsInput = None,
    frame_count: int,
    fill_only: bool = False,
) -> list[dict[str, Any]]:
    """Resolve shot ranges for measurement.

    Parent integration:
    - Fresh decode path: pass explicit ``supplied_shots`` (typed or dict).
    - Fill-only path: pass ``prior_facts=PriorFactsIn(...)`` with ``fill_only=True``.
      Missing ``priorFacts.shots`` is rejected; the whole video is never inferred.
    """
    if supplied_shots:
        return normalize_shots(supplied_shots)
    prior = shots_from_prior_facts(prior_facts, frame_count=frame_count)
    if prior is not None:
        return prior
    if fill_only or prior_facts is not None:
        raise AnalyzeError(
            INVALID_REQUEST,
            "priorFacts.shots is required for fill-only exposure/thumbnail analysis",
        )
    return [{"startFrame": 0, "endFrameExclusive": frame_count}]


def middle_third_bounds(start: int, end: int) -> tuple[int, int]:
    """Half-open middle-third bounds [lo, hi) inside [start, end)."""
    length = end - start
    if length <= 0:
        return start, start
    if length <= 2:
        return start, end
    lo = start + length // 3
    hi = start + (2 * length + 2) // 3
    hi = min(max(hi, lo + 1), end)
    return lo, hi


def middle_third_frames(start: int, end: int) -> list[int]:
    """Every frame in the middle third. Prefer ``middle_third_sample_frames`` for decode."""
    lo, hi = middle_third_bounds(start, end)
    return list(range(lo, hi))


def middle_third_sample_frames(
    start: int,
    end: int,
    *,
    max_frames: int,
) -> list[int]:
    """Evenly spaced, capped frames inside the middle third (deterministic)."""
    lo, hi = middle_third_bounds(start, end)
    length = hi - lo
    if length <= 0:
        return []
    if length <= max_frames:
        return list(range(lo, hi))
    if max_frames <= 1:
        return [lo + length // 2]
    step = (length - 1) / (max_frames - 1)
    frames = {lo + int(round(step * index)) for index in range(max_frames)}
    frames.add(lo)
    frames.add(hi - 1)
    return sorted(frame for frame in frames if lo <= frame < hi)


def representative_sample_frames(start: int, end: int, *, max_samples: int = 5) -> list[int]:
    """Deterministic evenly spaced representatives inside a shot."""
    length = end - start
    if length <= 0:
        return []
    if length <= max_samples:
        return list(range(start, end))
    if max_samples <= 1:
        return [start + length // 2]
    step = (length - 1) / (max_samples - 1)
    frames = {start + int(round(step * index)) for index in range(max_samples)}
    frames.add(start)
    frames.add(end - 1)
    return sorted(frame for frame in frames if start <= frame < end)
