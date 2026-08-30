import pytest
from pydantic import ValidationError

from media_analysis.schemas import AnalyzeRequest


def _minimal_source() -> dict:
    return {
        "signedGetUrl": "https://assets.example/clip.mp4",
        "expiresAt": "2099-01-01T00:00:00Z",
    }


def _sample_shot() -> dict:
    return {
        "startFrame": 0,
        "endFrameExclusive": 30,
        "startSecApprox": 0.0,
        "endSecApprox": 1.0,
        "boundaryKind": "start",
        "classification": "other",
    }


def _sample_subject() -> dict:
    return {
        "trackId": "0-1",
        "type": "person",
        "samples": [
            {
                "sourceFrame": 0,
                "presentationTimeSecApprox": 0.0,
                "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                "sampleKind": "detected",
                "scoreType": "raw_model",
                "occluded": False,
            }
        ],
    }


@pytest.mark.unit
def test_output_grants_require_unique_thumbnail_indices() -> None:
    with pytest.raises(ValidationError, match="indices must be unique"):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["thumbnails"],
                "outputGrants": {
                    "thumbnails": [
                        {
                            "index": 0,
                            "signedPutUrl": "https://assets.example/t0",
                            "expiresAt": "2099-01-01T00:00:00Z",
                        },
                        {
                            "index": 0,
                            "signedPutUrl": "https://assets.example/t1",
                            "expiresAt": "2099-01-01T00:00:00Z",
                        },
                    ]
                },
            }
        )


@pytest.mark.unit
def test_output_grants_require_unique_matte_labels() -> None:
    with pytest.raises(ValidationError, match="labels must be unique"):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["person_matte"],
                "outputGrants": {
                    "matteAssets": [
                        {
                            "label": "full",
                            "signedPutUrl": "https://assets.example/m0",
                            "expiresAt": "2099-01-01T00:00:00Z",
                        },
                        {
                            "label": "full",
                            "signedPutUrl": "https://assets.example/m1",
                            "expiresAt": "2099-01-01T00:00:00Z",
                        },
                    ]
                },
            }
        )


@pytest.mark.unit
def test_matte_frame_ranges_must_be_half_open_positive() -> None:
    with pytest.raises(ValidationError, match="greater than startFrame"):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["person_matte"],
                "matteFrameRanges": [{"startFrame": 10, "endFrameExclusive": 10}],
            }
        )


@pytest.mark.unit
def test_subject_tracks_matte_target_requires_nonempty_unique_ids() -> None:
    with pytest.raises(ValidationError, match="subjectTrackIds"):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["person_matte"],
                "matteTarget": {"mode": "subject_tracks", "subjectTrackIds": ["a", "a"]},
            }
        )


@pytest.mark.unit
def test_prior_facts_requires_shots_or_subjects() -> None:
    with pytest.raises(ValidationError, match="priorFacts must include"):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["quality"],
                "priorFacts": {"policyVersions": {"qualityPolicyVersion": "q-1"}},
            }
        )


@pytest.mark.unit
def test_prior_facts_accepts_canonical_shots_and_subjects() -> None:
    request = AnalyzeRequest.model_validate(
        {
            "idempotencyKey": "k",
            "source": _minimal_source(),
            "features": ["quality", "thumbnails"],
            "priorFacts": {
                "shots": [_sample_shot()],
                "subjects": [_sample_subject()],
                "policyVersions": {"qualityPolicyVersion": "quality-provisional-1.0.0"},
            },
            "outputGrants": {
                "thumbnails": [
                    {
                        "index": 0,
                        "signedPutUrl": "https://assets.example/t0",
                        "expiresAt": "2099-01-01T00:00:00Z",
                    }
                ]
            },
        }
    )
    assert request.priorFacts is not None
    assert request.priorFacts.shots is not None
    assert request.priorFacts.subjects is not None
    assert request.outputGrants is not None
    assert request.outputGrants.thumbnails is not None
    assert request.outputGrants.thumbnails[0].index == 0


@pytest.mark.unit
def test_grant_expiry_must_be_iso_timestamp() -> None:
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["thumbnails"],
                "outputGrants": {
                    "canonicalMp4": {
                        "signedPutUrl": "https://assets.example/out.mp4",
                        "expiresAt": "not-a-date",
                    }
                },
            }
        )


@pytest.mark.unit
def test_rational_denominator_must_be_nonzero() -> None:
    with pytest.raises(ValidationError, match="denominator"):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["shots"],
                "canonicalMedia": {"fps": {"numerator": 30, "denominator": 0}},
            }
        )


@pytest.mark.unit
def test_canonical_fps_numerator_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="fps.numerator"):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["shots"],
                "canonicalMedia": {"fps": {"numerator": 0, "denominator": 1}},
            }
        )


@pytest.mark.unit
def test_normalized_box_must_be_finite_and_in_unit_square() -> None:
    with pytest.raises(ValidationError, match="exceeds canonical width"):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["quality"],
                "priorFacts": {
                    "subjects": [
                        {
                            "trackId": "0-1",
                            "samples": [
                                {
                                    "sourceFrame": 0,
                                    "presentationTimeSecApprox": 0.0,
                                    "box": {"x": 0.8, "y": 0.1, "width": 0.3, "height": 0.2},
                                    "sampleKind": "detected",
                                    "scoreType": "raw_model",
                                    "occluded": False,
                                }
                            ],
                        }
                    ]
                },
            }
        )


@pytest.mark.unit
def test_prior_facts_shots_must_be_sorted_and_non_overlapping() -> None:
    with pytest.raises(ValidationError, match="sorted by startFrame"):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["quality"],
                "priorFacts": {
                    "shots": [
                        {
                            **_sample_shot(),
                            "startFrame": 30,
                            "endFrameExclusive": 60,
                        },
                        _sample_shot(),
                    ]
                },
            }
        )
    with pytest.raises(ValidationError, match="must not overlap"):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["quality"],
                "priorFacts": {
                    "shots": [
                        _sample_shot(),
                        {
                            **_sample_shot(),
                            "startFrame": 15,
                            "endFrameExclusive": 45,
                        },
                    ]
                },
            }
        )


@pytest.mark.unit
def test_prior_facts_subject_samples_sorted_in_range_when_frame_count_known() -> None:
    with pytest.raises(ValidationError, match="sorted by sourceFrame"):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["person_matte"],
                "canonicalMedia": {"frameCount": 60},
                "priorFacts": {
                    "subjects": [
                        {
                            "trackId": "0-1",
                            "samples": [
                                {
                                    "sourceFrame": 10,
                                    "presentationTimeSecApprox": 0.33,
                                    "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                                    "sampleKind": "detected",
                                    "scoreType": "raw_model",
                                    "occluded": False,
                                },
                                {
                                    "sourceFrame": 0,
                                    "presentationTimeSecApprox": 0.0,
                                    "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                                    "sampleKind": "detected",
                                    "scoreType": "raw_model",
                                    "occluded": False,
                                },
                            ],
                        }
                    ]
                },
            }
        )
    with pytest.raises(ValidationError, match="out of canonical frame range"):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "k",
                "source": _minimal_source(),
                "features": ["person_matte"],
                "canonicalMedia": {"frameCount": 30},
                "priorFacts": {
                    "subjects": [
                        {
                            **_sample_subject(),
                            "samples": [
                                {
                                    "sourceFrame": 30,
                                    "presentationTimeSecApprox": 1.0,
                                    "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                                    "sampleKind": "detected",
                                    "scoreType": "raw_model",
                                    "occluded": False,
                                }
                            ],
                        }
                    ]
                },
            }
        )
