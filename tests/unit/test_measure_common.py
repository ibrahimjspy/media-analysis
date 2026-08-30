from __future__ import annotations

import pytest

from media_analysis.errors import INVALID_REQUEST, AnalyzeError
from media_analysis.features.measure_common import (
    middle_third_bounds,
    middle_third_sample_frames,
    normalize_prior_facts,
    resolve_shots,
)
from media_analysis.schemas import PriorFactsIn, ShotAnalysisIn


def _shot(start: int, end: int) -> dict[str, object]:
    return {
        "startFrame": start,
        "endFrameExclusive": end,
        "startSecApprox": start / 30.0,
        "endSecApprox": end / 30.0,
        "boundaryKind": "hard_cut",
        "classification": "other",
    }


@pytest.mark.unit
def test_normalize_prior_facts_accepts_typed_model() -> None:
    prior = PriorFactsIn(
        shots=[
            ShotAnalysisIn(
                startFrame=0,
                endFrameExclusive=30,
                startSecApprox=0.0,
                endSecApprox=1.0,
                boundaryKind="start",
                classification="other",
            )
        ]
    )
    payload = normalize_prior_facts(prior)
    assert payload is not None
    assert payload["shots"][0]["startFrame"] == 0


@pytest.mark.unit
def test_resolve_shots_fill_only_rejects_missing_prior_shots() -> None:
    with pytest.raises(AnalyzeError) as exc:
        resolve_shots(
            prior_facts={"subjects": []},
            frame_count=120,
            fill_only=True,
        )
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_resolve_shots_fill_only_accepts_typed_prior_facts() -> None:
    prior = PriorFactsIn(
        shots=[
            ShotAnalysisIn(
                startFrame=0,
                endFrameExclusive=60,
                startSecApprox=0.0,
                endSecApprox=2.0,
                boundaryKind="start",
                classification="other",
            ),
            ShotAnalysisIn(
                startFrame=60,
                endFrameExclusive=120,
                startSecApprox=2.0,
                endSecApprox=4.0,
                boundaryKind="hard_cut",
                classification="other",
            ),
        ]
    )
    shots = resolve_shots(prior_facts=prior, frame_count=120, fill_only=True)
    assert len(shots) == 2


@pytest.mark.unit
def test_resolve_shots_explicit_internal_shots_without_prior() -> None:
    shots = resolve_shots(supplied_shots=[_shot(0, 45)], frame_count=90)
    assert shots[0]["endFrameExclusive"] == 45


@pytest.mark.unit
def test_middle_third_bounds_and_sample_boundaries() -> None:
    assert middle_third_bounds(0, 30) == (10, 20)
    sampled = middle_third_sample_frames(0, 30, max_frames=3)
    assert sampled == [10, 14, 19]
