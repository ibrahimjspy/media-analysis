from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from media_analysis.config import Settings
from media_analysis.errors import (
    CHECKSUM_MISMATCH,
    INVALID_REQUEST,
    LIMIT_EXCEEDED,
    SOURCE_EXPIRED,
    SOURCE_FETCH_FAILED,
    AnalyzeError,
)


def parse_expires_at(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def assert_source_fresh(expires_at: str, *, now: datetime | None = None) -> None:
    try:
        expiry = parse_expires_at(expires_at)
    except ValueError as exc:
        raise AnalyzeError(
            INVALID_REQUEST,
            "source.expiresAt must be an ISO-8601 timestamp",
        ) from exc
    current = now or datetime.now(UTC)
    if current >= expiry:
        raise AnalyzeError(SOURCE_EXPIRED, "Signed source URL expired before fetch")


def host_allowed(url: str, allowed_hosts: frozenset[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in allowed_hosts:
        return True
    wildcard_suffixes = (pattern[2:] for pattern in allowed_hosts if pattern.startswith("*."))
    return any(host.endswith(f".{suffix}") and host != suffix for suffix in wildcard_suffixes)


def assert_fetch_url(url: str, settings: Settings) -> None:
    if not host_allowed(url, settings.allowed_hosts):
        raise AnalyzeError(SOURCE_FETCH_FAILED, "Source host is not on the allowlist")


def _safe_error(url: str, exc: BaseException) -> AnalyzeError:
    host = urlparse(url).hostname or "unknown-host"
    error = AnalyzeError(SOURCE_FETCH_FAILED, f"Could not fetch source from {host}")
    error.__cause__ = exc
    return error


@dataclass(frozen=True, slots=True)
class DownloadedSource:
    byte_count: int
    sha256: str


def _stream_source(
    url: str,
    *,
    settings: Settings,
    expires_at: str,
    expected_sha256: str | None,
    timeout_sec: float | None,
    cancel_check: Callable[[], None] | None,
    write: Callable[[bytes], object],
) -> DownloadedSource:
    assert_source_fresh(expires_at)
    assert_fetch_url(url, settings)
    timeout = (
        timeout_sec if timeout_sec is not None else settings.media_analysis_download_timeout_sec
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            current_url = url
            for redirect_count in range(4):
                if cancel_check:
                    cancel_check()
                with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        if redirect_count == 3:
                            raise AnalyzeError(SOURCE_FETCH_FAILED, "Too many source redirects")
                        location = response.headers.get("location", "")
                        next_url = urljoin(current_url, location)
                        if not host_allowed(next_url, settings.allowed_hosts):
                            raise AnalyzeError(
                                SOURCE_FETCH_FAILED,
                                "Source redirect left the host allowlist",
                            )
                        current_url = next_url
                        continue

                    response.raise_for_status()
                    content_length = response.headers.get("content-length")
                    if (
                        content_length
                        and content_length.isdigit()
                        and int(content_length) > settings.media_analysis_max_bytes
                    ):
                        raise AnalyzeError(
                            LIMIT_EXCEEDED,
                            "Source exceeds MEDIA_ANALYSIS_MAX_BYTES",
                        )

                    for chunk in response.iter_bytes():
                        if cancel_check:
                            cancel_check()
                        size += len(chunk)
                        if size > settings.media_analysis_max_bytes:
                            raise AnalyzeError(
                                LIMIT_EXCEEDED,
                                "Source exceeds MEDIA_ANALYSIS_MAX_BYTES",
                            )
                        digest.update(chunk)
                        write(chunk)
                    break
            else:  # pragma: no cover - loop always breaks or raises
                raise AnalyzeError(SOURCE_FETCH_FAILED, "Could not fetch source")
    except AnalyzeError:
        raise
    except Exception as exc:
        raise _safe_error(url, exc) from exc

    actual_sha256 = digest.hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256.lower():
        raise AnalyzeError(CHECKSUM_MISMATCH, "Source sha256 did not match")
    return DownloadedSource(byte_count=size, sha256=actual_sha256)


def fetch_source(
    url: str,
    *,
    settings: Settings,
    expires_at: str,
    expected_sha256: str | None = None,
    timeout_sec: float | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> bytes:
    buffer = io.BytesIO()
    _stream_source(
        url,
        settings=settings,
        expires_at=expires_at,
        expected_sha256=expected_sha256,
        timeout_sec=timeout_sec,
        cancel_check=cancel_check,
        write=buffer.write,
    )
    return buffer.getvalue()


def fetch_source_to_path(
    url: str,
    dest: Path,
    *,
    settings: Settings,
    expires_at: str,
    expected_sha256: str | None = None,
    timeout_sec: float | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> DownloadedSource:
    """Stream a bounded source to disk without retaining the media body in RAM."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as handle:
            return _stream_source(
                url,
                settings=settings,
                expires_at=expires_at,
                expected_sha256=expected_sha256,
                timeout_sec=timeout_sec,
                cancel_check=cancel_check,
                write=handle.write,
            )
    except Exception:
        dest.unlink(missing_ok=True)
        raise
