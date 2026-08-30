import pytest

from media_analysis.errors import STABLE_CODES, AnalyzeError, error_body


@pytest.mark.unit
def test_error_body_is_code_and_message_only() -> None:
    body = error_body("DECODE_FAILED", "Could not decode source")
    assert body == {"error": "Could not decode source", "code": "DECODE_FAILED"}


@pytest.mark.unit
def test_unstable_code_cannot_be_constructed() -> None:
    with pytest.raises(ValueError, match="unstable"):
        AnalyzeError("WHOOPS", "nope")


@pytest.mark.unit
def test_locked_codes_are_exactly_the_contract() -> None:
    assert STABLE_CODES == frozenset(
        {
            "UNAUTHORIZED",
            "INVALID_REQUEST",
            "SOURCE_EXPIRED",
            "SOURCE_FETCH_FAILED",
            "DECODE_FAILED",
            "FEATURE_UNAVAILABLE",
            "MODEL_NOT_READY",
            "TIMEOUT",
            "CANCELLED",
            "UPLOAD_FAILED",
            "CHECKSUM_MISMATCH",
            "LIMIT_EXCEEDED",
            "INTERNAL_ERROR",
        }
    )
