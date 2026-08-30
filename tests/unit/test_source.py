from datetime import UTC, datetime

import pytest

from media_analysis.config import Settings
from media_analysis.errors import (
    CANCELLED,
    INVALID_REQUEST,
    SOURCE_EXPIRED,
    SOURCE_FETCH_FAILED,
    AnalyzeError,
)
from media_analysis.source import assert_fetch_url, assert_source_fresh, fetch_source, host_allowed


@pytest.mark.unit
def test_expired_source_is_rejected_before_fetch() -> None:
    with pytest.raises(AnalyzeError) as exc:
        assert_source_fresh(
            "2020-01-01T00:00:00Z",
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )
    assert exc.value.code == SOURCE_EXPIRED
    assert "http" not in exc.value.message.lower()


@pytest.mark.unit
def test_invalid_expiry_is_a_stable_invalid_request() -> None:
    with pytest.raises(AnalyzeError) as exc:
        assert_source_fresh("not-a-timestamp")
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_host_allowlist_rejects_other_hosts_and_schemes() -> None:
    allowed = frozenset({"127.0.0.1", "assets.example.test"})
    assert host_allowed("http://127.0.0.1/clip.mp4", allowed)
    assert host_allowed("https://assets.example.test/a", allowed)
    assert not host_allowed("https://evil.test/a", allowed)
    assert not host_allowed("file:///tmp/clip.mp4", allowed)
    assert not host_allowed("s3://bucket/key", allowed)


@pytest.mark.unit
def test_host_allowlist_supports_explicit_subdomain_wildcards_only() -> None:
    allowed = frozenset({"*.storage.example"})
    assert host_allowed("https://bucket.storage.example/clip.mp4", allowed)
    assert not host_allowed("https://storage.example/clip.mp4", allowed)
    assert not host_allowed("https://storage.example.evil.test/clip.mp4", allowed)


@pytest.mark.unit
def test_disallowed_host_error_does_not_echo_url() -> None:
    settings = Settings(media_analysis_allowed_hosts="127.0.0.1")
    with pytest.raises(AnalyzeError) as exc:
        assert_fetch_url("https://secret.example/clip.mp4?token=abc", settings)
    assert exc.value.code == SOURCE_FETCH_FAILED
    assert "token=abc" not in exc.value.message
    assert "secret.example" not in exc.value.message


@pytest.mark.unit
def test_download_propagates_job_cancellation() -> None:
    def cancel() -> None:
        raise AnalyzeError(CANCELLED, "Analysis cancelled")

    with pytest.raises(AnalyzeError) as exc:
        fetch_source(
            "https://assets.example/clip.mp4",
            settings=Settings(media_analysis_allowed_hosts="assets.example"),
            expires_at="2099-01-01T00:00:00Z",
            cancel_check=cancel,
        )
    assert exc.value.code == CANCELLED
