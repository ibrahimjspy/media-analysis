"""Stage 1 matte request validation. Stage 2 modes fail explicitly."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from media_analysis.errors import INVALID_REQUEST, AnalyzeError
from media_analysis.person_matte.types import MatteOutputGrant, ShotBoundary


def coerce_mapping(value: Any, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if not isinstance(dumped, dict):
            raise AnalyzeError(INVALID_REQUEST, f"{field_name} must serialize to an object")
        return dumped
    if isinstance(value, dict):
        return value
    raise AnalyzeError(INVALID_REQUEST, f"{field_name} must be an object")


def _parse_shot(raw: Any) -> ShotBoundary:
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        raise AnalyzeError(INVALID_REQUEST, "priorFacts.shots entries are invalid")
    try:
        start = int(raw["startFrame"])
        end = int(raw["endFrameExclusive"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalyzeError(INVALID_REQUEST, "priorFacts.shots entries are invalid") from exc
    if end <= start:
        raise AnalyzeError(INVALID_REQUEST, "priorFacts.shots has non-positive span")
    return ShotBoundary(start_frame=start, end_frame_exclusive=end)


def parse_shot_boundaries(prior_facts: dict[str, Any] | None) -> tuple[ShotBoundary, ...]:
    if not prior_facts:
        raise AnalyzeError(
            INVALID_REQUEST,
            "priorFacts.shots is required for shot_boundaries reset policy",
        )
    shots_raw = prior_facts.get("shots")
    if not isinstance(shots_raw, list) or not shots_raw:
        raise AnalyzeError(
            INVALID_REQUEST,
            "priorFacts.shots is required for shot_boundaries reset policy",
        )
    return tuple(_parse_shot(item) for item in shots_raw)


def validate_shot_timeline(
    shots: tuple[ShotBoundary, ...],
    *,
    frame_count: int,
) -> tuple[ShotBoundary, ...]:
    if frame_count <= 0:
        raise AnalyzeError(INVALID_REQUEST, "canonical frameCount must be positive")
    ordered = sorted(shots, key=lambda item: item.start_frame)
    if tuple(ordered) != shots:
        raise AnalyzeError(INVALID_REQUEST, "priorFacts.shots must be sorted by startFrame")
    if ordered[0].start_frame != 0:
        raise AnalyzeError(INVALID_REQUEST, "priorFacts.shots must start at frame 0")
    if ordered[-1].end_frame_exclusive != frame_count:
        raise AnalyzeError(
            INVALID_REQUEST,
            "priorFacts.shots must cover the full canonical timeline",
        )
    for index, shot in enumerate(ordered):
        if shot.start_frame < 0 or shot.end_frame_exclusive > frame_count:
            raise AnalyzeError(INVALID_REQUEST, "priorFacts.shots exceed canonical frameCount")
        if index > 0:
            previous = ordered[index - 1]
            if shot.start_frame < previous.end_frame_exclusive:
                raise AnalyzeError(INVALID_REQUEST, "priorFacts.shots must not overlap")
            if shot.start_frame > previous.end_frame_exclusive:
                raise AnalyzeError(INVALID_REQUEST, "priorFacts.shots must not leave gaps")
    return tuple(ordered)


def validate_contiguous_frame_indices(
    frames: Iterable[tuple[int, Any]],
    *,
    frame_count: int,
) -> None:
    expected = 0
    seen: set[int] = set()
    for source_frame, _payload in frames:
        if source_frame in seen:
            raise AnalyzeError(INVALID_REQUEST, "duplicate sourceFrame in matte frame stream")
        if source_frame != expected:
            raise AnalyzeError(
                INVALID_REQUEST,
                "sourceFrame indices must be contiguous from 0",
            )
        seen.add(source_frame)
        expected += 1
    if expected != frame_count:
        raise AnalyzeError(
            INVALID_REQUEST,
            "full_duration matte frame count must match canonical frameCount",
        )


def parse_matte_grant(output_grants: dict[str, Any] | None) -> MatteOutputGrant:
    if not output_grants:
        raise AnalyzeError(INVALID_REQUEST, "outputGrants.matteAssets is required for person_matte")
    assets = output_grants.get("matteAssets")
    if not isinstance(assets, list) or len(assets) != 1:
        raise AnalyzeError(
            INVALID_REQUEST,
            "Stage 1 person_matte requires exactly one matteAssets grant (full_duration)",
        )
    raw = assets[0]
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        raise AnalyzeError(INVALID_REQUEST, "matteAssets grant must be an object")
    label = raw.get("label")
    put_url = raw.get("signedPutUrl")
    expires = raw.get("expiresAt")
    if not label or not put_url or not expires:
        raise AnalyzeError(
            INVALID_REQUEST,
            "matteAssets grant missing label, signedPutUrl, or expiresAt",
        )
    if raw.get("segmentFrames") is not None or raw.get("bucketIndex") is not None:
        raise AnalyzeError(
            INVALID_REQUEST,
            "segmented matte delivery is Stage 2 and is not implemented",
        )
    return MatteOutputGrant(label=str(label), signed_put_url=str(put_url), expires_at=str(expires))


def validate_matte_target(matte_target: dict[str, Any] | None) -> dict[str, Any]:
    if not matte_target:
        raise AnalyzeError(INVALID_REQUEST, "matteTarget is required for person_matte")
    if hasattr(matte_target, "model_dump"):
        matte_target = matte_target.model_dump()
    if not isinstance(matte_target, dict):
        raise AnalyzeError(INVALID_REQUEST, "matteTarget must be an object")
    mode = matte_target.get("mode")
    if mode == "subject_tracks":
        raise AnalyzeError(
            INVALID_REQUEST,
            "matteTarget.mode subject_tracks is Stage 2 and is not implemented",
        )
    if mode != "all_people":
        raise AnalyzeError(INVALID_REQUEST, "matteTarget.mode must be all_people for Stage 1")
    return {"mode": "all_people"}


def validate_matte_stage1_request(
    *,
    matte_target: dict[str, Any] | None,
    output_grants: dict[str, Any] | None,
    prior_facts: dict[str, Any] | None = None,
    delivery_mode: str | None = None,
    frame_count: int | None = None,
) -> tuple[dict[str, Any], MatteOutputGrant, tuple[ShotBoundary, ...]]:
    """Validate Stage 1-only matte inputs. Stage 2 requests raise INVALID_REQUEST."""
    if delivery_mode == "segmented":
        raise AnalyzeError(
            INVALID_REQUEST,
            "segmented matte delivery is Stage 2 and is not implemented",
        )
    target = validate_matte_target(matte_target)
    grant = parse_matte_grant(coerce_mapping(output_grants, field_name="outputGrants"))
    facts = coerce_mapping(prior_facts, field_name="priorFacts")
    shots = parse_shot_boundaries(facts)
    if frame_count is not None:
        shots = validate_shot_timeline(shots, frame_count=frame_count)
    return target, grant, shots
