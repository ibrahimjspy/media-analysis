from __future__ import annotations

import pytest

from media_analysis.errors import INVALID_REQUEST, AnalyzeError
from media_analysis.person_matte.types import ShotBoundary
from media_analysis.person_matte.validation import (
    validate_contiguous_frame_indices,
    validate_matte_stage1_request,
    validate_shot_timeline,
)


def _grants() -> dict:
    return {
        "matteAssets": [
            {
                "label": "full",
                "signedPutUrl": "https://example.test/put",
                "expiresAt": "2099-01-01T00:00:00Z",
            }
        ]
    }


def _shots(frame_count: int = 30) -> dict:
    return {"shots": [{"startFrame": 0, "endFrameExclusive": frame_count}]}


@pytest.mark.unit
def test_stage1_accepts_all_people_single_grant() -> None:
    target, grant, shots = validate_matte_stage1_request(
        matte_target={"mode": "all_people"},
        output_grants=_grants(),
        prior_facts=_shots(),
        frame_count=30,
    )
    assert target == {"mode": "all_people"}
    assert grant.label == "full"
    assert grant.expires_at.startswith("2099")
    assert len(shots) == 1


@pytest.mark.unit
def test_prior_facts_shots_required() -> None:
    with pytest.raises(AnalyzeError) as exc:
        validate_matte_stage1_request(
            matte_target={"mode": "all_people"},
            output_grants=_grants(),
            prior_facts=None,
            frame_count=30,
        )
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_shots_must_cover_full_timeline() -> None:
    with pytest.raises(AnalyzeError) as exc:
        validate_shot_timeline(
            (ShotBoundary(0, 20),),
            frame_count=30,
        )
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_shots_must_be_sorted_non_overlapping() -> None:
    with pytest.raises(AnalyzeError) as exc:
        validate_shot_timeline(
            (
                ShotBoundary(10, 20),
                ShotBoundary(0, 10),
            ),
            frame_count=20,
        )
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_contiguous_frame_indices_reject_gap() -> None:
    frames = [(0, object()), (2, object())]
    with pytest.raises(AnalyzeError) as exc:
        validate_contiguous_frame_indices(frames, frame_count=3)
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_contiguous_frame_indices_reject_duplicate() -> None:
    frames = [(0, object()), (0, object())]
    with pytest.raises(AnalyzeError) as exc:
        validate_contiguous_frame_indices(frames, frame_count=2)
    assert "duplicate" in exc.value.message


@pytest.mark.unit
def test_subject_tracks_is_stage2_invalid() -> None:
    with pytest.raises(AnalyzeError) as exc:
        validate_matte_stage1_request(
            matte_target={"mode": "subject_tracks", "subjectTrackIds": ["t1"]},
            output_grants=_grants(),
            prior_facts=_shots(),
        )
    assert exc.value.code == INVALID_REQUEST
    assert "Stage 2" in exc.value.message


@pytest.mark.unit
def test_segmented_delivery_is_stage2_invalid() -> None:
    with pytest.raises(AnalyzeError) as exc:
        validate_matte_stage1_request(
            matte_target={"mode": "all_people"},
            output_grants={
                "matteAssets": [
                    {
                        "label": "seg-0",
                        "signedPutUrl": "https://example.test/put0",
                        "expiresAt": "2099-01-01T00:00:00Z",
                        "bucketIndex": 0,
                    },
                    {
                        "label": "seg-1",
                        "signedPutUrl": "https://example.test/put1",
                        "expiresAt": "2099-01-01T00:00:00Z",
                        "bucketIndex": 1,
                    },
                ]
            },
            prior_facts=_shots(),
        )
    assert exc.value.code == INVALID_REQUEST
