from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from media_analysis.config import Settings
from media_analysis.errors import (
    CANCELLED,
    CHECKSUM_MISMATCH,
    INVALID_REQUEST,
    LIMIT_EXCEEDED,
    UPLOAD_FAILED,
    AnalyzeError,
)
from media_analysis.upload import assert_grant_fresh, assert_upload_url, upload_artifact


class _PutHandler(BaseHTTPRequestHandler):
    received: bytes = b""
    content_type: str = ""
    last_path: str = ""

    def do_PUT(self) -> None:  # noqa: N802
        type(self).last_path = self.path
        length = int(self.headers.get("Content-Length", "0"))
        type(self).received = self.rfile.read(length)
        type(self).content_type = self.headers.get("Content-Type", "")
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        return


@pytest.fixture
def put_server(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setenv("MEDIA_ANALYSIS_ALLOWED_HOSTS", "127.0.0.1,localhost")
    from media_analysis.config import reset_settings

    reset_settings()
    _PutHandler.received = b""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PutHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield {
        "url": f"http://{host}:{port}/upload/object?X-Amz-Signature=fake",
        "handler": _PutHandler,
        "settings": Settings(),
    }
    server.shutdown()
    thread.join(timeout=2)


@pytest.mark.unit
def test_expired_grant_is_rejected_before_upload() -> None:
    with pytest.raises(AnalyzeError) as exc:
        assert_grant_fresh("2020-01-01T00:00:00Z", now=datetime(2026, 8, 28, tzinfo=UTC))
    assert exc.value.code == UPLOAD_FAILED
    assert "http" not in exc.value.message.lower()
    assert "source" not in exc.value.message.lower()


@pytest.mark.unit
def test_invalid_grant_expiry_is_invalid_request() -> None:
    with pytest.raises(AnalyzeError) as exc:
        assert_grant_fresh("not-a-timestamp")
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_disallowed_upload_host_error_does_not_echo_url() -> None:
    settings = Settings(media_analysis_allowed_hosts="127.0.0.1")
    with pytest.raises(AnalyzeError) as exc:
        assert_upload_url("https://secret.example/upload?token=abc", settings)
    assert exc.value.code == UPLOAD_FAILED
    assert "token=abc" not in exc.value.message
    assert "secret.example" not in exc.value.message


@pytest.mark.unit
def test_upload_puts_bytes_once_to_local_http_fixture(put_server: dict) -> None:
    payload = b"canonical-bytes"
    result = upload_artifact(
        payload,
        put_server["url"],
        settings=put_server["settings"],
        expires_at="2099-01-01T00:00:00Z",
        mime_type="video/mp4",
    )
    assert put_server["handler"].received == payload
    assert put_server["handler"].content_type == "video/mp4"
    assert result["byteCount"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["mimeType"] == "video/mp4"


@pytest.mark.unit
def test_upload_checksum_mismatch_is_stable() -> None:
    settings = Settings(media_analysis_allowed_hosts="127.0.0.1")
    with pytest.raises(AnalyzeError) as exc:
        upload_artifact(
            b"payload",
            "http://127.0.0.1/upload",
            settings=settings,
            expires_at="2099-01-01T00:00:00Z",
            expected_sha256="0" * 64,
        )
    assert exc.value.code == CHECKSUM_MISMATCH


@pytest.mark.unit
def test_upload_limit_exceeded_before_network() -> None:
    settings = Settings(media_analysis_allowed_hosts="127.0.0.1", media_analysis_max_bytes=4)
    with pytest.raises(AnalyzeError) as exc:
        upload_artifact(
            b"12345",
            "http://127.0.0.1/upload",
            settings=settings,
            expires_at="2099-01-01T00:00:00Z",
        )
    assert exc.value.code == LIMIT_EXCEEDED


@pytest.mark.unit
def test_upload_propagates_job_cancellation(put_server: dict) -> None:
    def cancel() -> None:
        raise AnalyzeError(CANCELLED, "Analysis cancelled")

    with pytest.raises(AnalyzeError) as exc:
        upload_artifact(
            b"payload",
            put_server["url"],
            settings=put_server["settings"],
            expires_at="2099-01-01T00:00:00Z",
            cancel_check=cancel,
        )
    assert exc.value.code == CANCELLED


@pytest.mark.unit
def test_upload_issues_exactly_one_put(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(media_analysis_allowed_hosts="127.0.0.1")
    calls: list[str] = []

    class FakeResponse:
        status_code = 200
        headers: dict[str, str] = {}
        is_redirect = False

        def raise_for_status(self) -> None:
            return

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put(self, url: str, *, content: bytes, headers: dict[str, str]):
            calls.append(url)
            return FakeResponse()

    monkeypatch.setattr("media_analysis.upload.httpx.Client", FakeClient)
    upload_artifact(
        b"abc",
        "http://127.0.0.1/object",
        settings=settings,
        expires_at="2099-01-01T00:00:00Z",
    )
    assert calls == ["http://127.0.0.1/object"]


@pytest.mark.unit
def test_upload_rejects_redirect_without_second_put(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(media_analysis_allowed_hosts="127.0.0.1")
    calls: list[str] = []

    class FakeResponse:
        status_code = 302
        headers = {"location": "http://127.0.0.1/second"}
        is_redirect = True

        def raise_for_status(self) -> None:
            return

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put(self, url: str, *, content: bytes, headers: dict[str, str]):
            calls.append(url)
            return FakeResponse()

    monkeypatch.setattr("media_analysis.upload.httpx.Client", FakeClient)
    with pytest.raises(AnalyzeError) as exc:
        upload_artifact(
            b"abc",
            "http://127.0.0.1/first",
            settings=settings,
            expires_at="2099-01-01T00:00:00Z",
        )
    assert exc.value.code == UPLOAD_FAILED
    assert "redirect" in exc.value.message.lower()
    assert calls == ["http://127.0.0.1/first"]
