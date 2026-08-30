import pytest
from pydantic import ValidationError

from media_analysis.analyze import validate_analyze_request, validate_features
from media_analysis.config import Settings
from media_analysis.errors import FEATURE_UNAVAILABLE, INVALID_REQUEST, AnalyzeError
from media_analysis.schemas import AnalyzeRequest


@pytest.mark.unit
def test_person_matte_is_invalid_on_cpu_image() -> None:
    settings = Settings(media_analysis_image="analysis-cpu")
    with pytest.raises(AnalyzeError) as exc:
        validate_features(["person_matte"], settings)
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_cpu_features_are_unavailable_on_matte_image() -> None:
    settings = Settings(media_analysis_image="matte-gpu")
    with pytest.raises(AnalyzeError) as exc:
        validate_features(["shots"], settings)
    assert exc.value.code == FEATURE_UNAVAILABLE


@pytest.mark.unit
def test_matte_image_rejects_mixed_cpu_features() -> None:
    settings = Settings(media_analysis_image="matte-cpu")
    with pytest.raises(AnalyzeError) as exc:
        validate_features(["person_matte", "shots"], settings)
    assert exc.value.code == FEATURE_UNAVAILABLE


@pytest.mark.unit
def test_thumbnails_requires_output_grants_at_boundary() -> None:
    settings = Settings(media_analysis_image="analysis-cpu")
    request = AnalyzeRequest.model_validate(
        {
            "idempotencyKey": "k",
            "source": {
                "signedGetUrl": "https://assets.example/clip.mp4",
                "expiresAt": "2099-01-01T00:00:00Z",
            },
            "features": ["thumbnails"],
        }
    )
    with pytest.raises(AnalyzeError) as exc:
        validate_analyze_request(request, settings)
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
@pytest.mark.parametrize(
    ("target", "ranges"),
    [
        ({"mode": "subject_tracks", "subjectTrackIds": ["track-1"]}, None),
        ({"mode": "all_people"}, [{"startFrame": 0, "endFrameExclusive": 10}]),
    ],
)
def test_matte_stage2_modes_are_rejected_at_boundary(
    target: dict[str, object],
    ranges: list[dict[str, int]] | None,
) -> None:
    request = AnalyzeRequest.model_validate(
        {
            "idempotencyKey": "matte-stage2",
            "source": {
                "signedGetUrl": "https://assets.example/clip.mp4",
                "expiresAt": "2099-01-01T00:00:00Z",
            },
            "features": ["person_matte"],
            "matteTarget": target,
            "matteFrameRanges": ranges,
            "priorFacts": {
                "shots": [
                    {
                        "startFrame": 0,
                        "endFrameExclusive": 10,
                        "startSecApprox": 0.0,
                        "endSecApprox": 1.0,
                        "boundaryKind": "start",
                        "classification": "other",
                    }
                ]
            },
            "outputGrants": {
                "matteAssets": [
                    {
                        "label": "full",
                        "signedPutUrl": "https://assets.example/matte.mp4",
                        "expiresAt": "2099-01-01T00:00:00Z",
                    }
                ]
            },
        }
    )
    with pytest.raises(AnalyzeError) as exc:
        validate_analyze_request(
            request,
            Settings(media_analysis_image="matte-cpu"),
        )
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
@pytest.mark.parametrize(
    "features,resolution",
    [
        (["shots", "shots"], {"width": 160, "height": 120}),
        (["shots"], {"width": 0, "height": 120}),
    ],
)
def test_request_rejects_duplicate_features_and_nonpositive_resolution(
    features: list[str],
    resolution: dict[str, int],
) -> None:
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(
            {
                "idempotencyKey": "invalid",
                "source": {
                    "signedGetUrl": "https://assets.example/clip.mp4",
                    "expiresAt": "2026-08-29T00:00:00Z",
                },
                "features": features,
                "analysisResolution": resolution,
            }
        )
