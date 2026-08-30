from __future__ import annotations

import hashlib
from collections.abc import Callable
from urllib.parse import urlparse

import httpx

from media_analysis.config import Settings
from media_analysis.errors import (
    CHECKSUM_MISMATCH,
    INVALID_REQUEST,
    LIMIT_EXCEEDED,
    UPLOAD_FAILED,
    AnalyzeError,
)
from media_analysis.source import host_allowed, parse_expires_at


def assert_grant_fresh(expires_at: str, *, now=None) -> None:
    from datetime import UTC, datetime

    try:
        expiry = parse_expires_at(expires_at)
    except ValueError as exc:
        raise AnalyzeError(
            INVALID_REQUEST,
            "output grant expiresAt must be an ISO-8601 timestamp",
        ) from exc
    current = now or datetime.now(UTC)
    if current >= expiry:
        raise AnalyzeError(UPLOAD_FAILED, "Signed output grant expired before upload")


def assert_upload_url(url: str, settings: Settings) -> None:
    if not host_allowed(url, settings.allowed_hosts):
        raise AnalyzeError(UPLOAD_FAILED, "Upload host is not on the allowlist")


def _safe_upload_error(url: str, exc: BaseException) -> AnalyzeError:
    host = urlparse(url).hostname or "unknown-host"
    error = AnalyzeError(UPLOAD_FAILED, f"Could not upload artifact to {host}")
    error.__cause__ = exc
    return error


def upload_artifact(
    data: bytes,
    url: str,
    *,
    settings: Settings,
    expires_at: str,
    expected_sha256: str | None = None,
    mime_type: str = "application/octet-stream",
    timeout_sec: float | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, int | str]:
    """PUT bytes once to a scoped signed URL. Returns sha256, byteCount, mimeType."""
    assert_grant_fresh(expires_at)
    assert_upload_url(url, settings)
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 and digest != expected_sha256.lower():
        raise AnalyzeError(CHECKSUM_MISMATCH, "Artifact sha256 did not match")
    if len(data) > settings.media_analysis_max_bytes:
        raise AnalyzeError(LIMIT_EXCEEDED, "Artifact exceeds MEDIA_ANALYSIS_MAX_BYTES")

    timeout = (
        timeout_sec if timeout_sec is not None else settings.media_analysis_download_timeout_sec
    )
    headers = {
        "Content-Type": mime_type,
        "Content-Length": str(len(data)),
    }

    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            if cancel_check:
                cancel_check()
            response = client.put(url, content=data, headers=headers)
            if response.is_redirect:
                raise AnalyzeError(UPLOAD_FAILED, "Upload redirect is not allowed")
            response.raise_for_status()
    except AnalyzeError:
        raise
    except Exception as exc:
        raise _safe_upload_error(url, exc) from exc

    return {"sha256": digest, "byteCount": len(data), "mimeType": mime_type}
