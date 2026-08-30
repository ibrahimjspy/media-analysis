import pytest

from media_analysis.request_hash import request_hash


def _base() -> dict:
    return {
        "canonicalize": False,
        "features": ["shots", "subjects"],
        "canonicalMedia": {"frameCount": 30, "width": 320, "height": 240},
        "source": {
            "signedGetUrl": "https://bucket.s3.amazonaws.com/a?X-Amz-Signature=one",
            "expiresAt": "2099-01-01T00:00:00Z",
            "sha256": "abc",
        },
    }


@pytest.mark.unit
def test_hash_ignores_signed_url_and_expiry() -> None:
    first = _base()
    second = _base()
    second["source"]["signedGetUrl"] = "https://bucket.s3.amazonaws.com/a?X-Amz-Signature=two"
    second["source"]["expiresAt"] = "2099-06-01T00:00:00Z"
    assert request_hash(first) == request_hash(second)


@pytest.mark.unit
def test_hash_changes_when_features_change() -> None:
    first = _base()
    second = _base()
    second["features"] = ["shots"]
    assert request_hash(first) != request_hash(second)


@pytest.mark.unit
def test_feature_order_does_not_change_hash() -> None:
    first = _base()
    second = _base()
    second["features"] = ["subjects", "shots"]
    assert request_hash(first) == request_hash(second)


@pytest.mark.unit
def test_hash_ignores_output_grant_urls_and_expiry() -> None:
    first = _base()
    first["outputGrants"] = {
        "canonicalMp4": {
            "signedPutUrl": "https://bucket.s3.amazonaws.com/out?sig=one",
            "expiresAt": "2099-01-01T00:00:00Z",
        },
        "thumbnails": [
            {
                "index": 0,
                "signedPutUrl": "https://bucket.s3.amazonaws.com/t0?sig=one",
                "expiresAt": "2099-01-01T00:00:00Z",
            }
        ],
        "matteAssets": [
            {
                "label": "full",
                "signedPutUrl": "https://bucket.s3.amazonaws.com/m0?sig=one",
                "expiresAt": "2099-01-01T00:00:00Z",
            }
        ],
    }
    second = dict(first)
    second["outputGrants"] = {
        "canonicalMp4": {
            "signedPutUrl": "https://bucket.s3.amazonaws.com/out?sig=two",
            "expiresAt": "2099-06-01T00:00:00Z",
        },
        "thumbnails": [
            {
                "index": 0,
                "signedPutUrl": "https://bucket.s3.amazonaws.com/t0?sig=two",
                "expiresAt": "2099-06-01T00:00:00Z",
            }
        ],
        "matteAssets": [
            {
                "label": "full",
                "signedPutUrl": "https://bucket.s3.amazonaws.com/m0?sig=two",
                "expiresAt": "2099-06-01T00:00:00Z",
            }
        ],
    }
    assert request_hash(first) == request_hash(second)


@pytest.mark.unit
def test_hash_changes_when_output_grant_shape_changes() -> None:
    first = _base()
    first["outputGrants"] = {
        "thumbnails": [
            {
                "index": 0,
                "signedPutUrl": "https://bucket.s3.amazonaws.com/t0",
                "expiresAt": "2099-01-01T00:00:00Z",
            }
        ]
    }
    second = _base()
    second["outputGrants"] = {
        "thumbnails": [
            {
                "index": 1,
                "signedPutUrl": "https://bucket.s3.amazonaws.com/t1",
                "expiresAt": "2099-01-01T00:00:00Z",
            }
        ]
    }
    third = _base()
    third["outputGrants"] = {
        "matteAssets": [
            {
                "label": "full",
                "signedPutUrl": "https://bucket.s3.amazonaws.com/m0",
                "expiresAt": "2099-01-01T00:00:00Z",
            }
        ]
    }
    assert request_hash(first) != request_hash(second)
    assert request_hash(first) != request_hash(third)


@pytest.mark.unit
def test_hash_includes_prior_facts() -> None:
    shot_a = {
        "startFrame": 0,
        "endFrameExclusive": 30,
        "startSecApprox": 0.0,
        "endSecApprox": 1.0,
        "boundaryKind": "start",
        "classification": "other",
    }
    shot_b = {
        "startFrame": 0,
        "endFrameExclusive": 60,
        "startSecApprox": 0.0,
        "endSecApprox": 2.0,
        "boundaryKind": "start",
        "classification": "other",
    }
    first = _base()
    first["priorFacts"] = {"shots": [shot_a]}
    second = _base()
    second["priorFacts"] = {"shots": [shot_b]}
    assert request_hash(first) != request_hash(second)
