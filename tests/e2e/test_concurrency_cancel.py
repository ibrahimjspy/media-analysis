from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import media_analysis.app as app_module
from media_analysis.errors import CANCELLED, AnalyzeError
from media_analysis.telemetry import JobTelemetry


@pytest.fixture(autouse=True)
def prepared_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep existing gate tests independent of HTTP source availability."""

    @contextmanager
    def prepare(*_args, deadline, **_kwargs):
        yield SimpleNamespace(deadline=deadline, telemetry=JobTelemetry())

    monkeypatch.setattr(app_module, "prepare_analysis_source", prepare)


def _payload(key: str) -> dict[str, Any]:
    return {
        "idempotencyKey": key,
        "features": ["shots"],
        "source": {
            "signedGetUrl": "http://127.0.0.1/unused.mp4",
            "expiresAt": "2099-01-01T00:00:00Z",
        },
    }


@pytest.mark.e2e
def test_worker_serializes_inference_jobs(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_run(*_args, **_kwargs) -> dict[str, Any]:
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1
        return {"overallStatus": "completed"}

    monkeypatch.setattr(app_module, "run_analyze", fake_run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda key: client.post(
                    "/analyze",
                    headers=auth_headers,
                    json=_payload(key),
                ),
                ("serial-1", "serial-2"),
            )
        )

    assert [response.status_code for response in responses] == [200, 200]
    assert max_active == 1


@pytest.mark.e2e
def test_cancel_interrupts_an_active_job(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()

    def fake_run(*_args, job, **_kwargs) -> dict[str, Any]:
        entered.set()
        if not job.cancel.wait(timeout=2):
            raise AssertionError("cancel endpoint did not signal the active job")
        raise AnalyzeError(CANCELLED, "Analysis cancelled")

    monkeypatch.setattr(app_module, "run_analyze", fake_run)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            client.post,
            "/analyze",
            headers=auth_headers,
            json=_payload("cancel-active"),
        )
        assert entered.wait(timeout=1)
        cancel = client.post("/analyze/cancel-active/cancel", headers=auth_headers)
        response = pending.result(timeout=2)

    assert cancel.json()["cancelled"] is True
    assert response.status_code == 409
    assert response.json()["code"] == CANCELLED


@pytest.mark.e2e
def test_download_of_next_job_overlaps_serial_analysis(client, auth_headers, monkeypatch):
    """Second source becomes ready while the first model still owns its gate."""
    first_inference = threading.Event()
    second_prepared = threading.Event()
    active = 0
    peak = 0
    lock = threading.Lock()

    @contextmanager
    def prepare(payload, *, deadline, **_kwargs):
        if payload.idempotencyKey == "overlap-2":
            assert first_inference.wait(2)
            second_prepared.set()
        yield SimpleNamespace(deadline=deadline, telemetry=JobTelemetry())

    def run(payload, **_kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        if payload.idempotencyKey == "overlap-1":
            first_inference.set()
            assert second_prepared.wait(2), "Download did not overlap inference"
        with lock:
            active -= 1
        return {"overallStatus": "completed"}

    monkeypatch.setattr(app_module, "prepare_analysis_source", prepare)
    monkeypatch.setattr(app_module, "run_analyze", run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            client.post, "/analyze", headers=auth_headers, json=_payload("overlap-1")
        )
        assert first_inference.wait(2)
        second = pool.submit(
            client.post, "/analyze", headers=auth_headers, json=_payload("overlap-2")
        )
        assert first.result(3).status_code == second.result(3).status_code == 200
    assert peak == 1


@pytest.mark.e2e
def test_cancel_waiting_job_cleans_prefetch_and_never_enters_inference(
    client, auth_headers, monkeypatch
):
    """A cancelled prefetched job cannot run after the preceding job completes."""
    first_entered = threading.Event()
    second_ready = threading.Event()
    release = threading.Event()
    cleaned = threading.Event()
    calls = []

    @contextmanager
    def prepare(payload, *, deadline, **_kwargs):
        if payload.idempotencyKey == "waiting-2":
            second_ready.set()
        try:
            yield SimpleNamespace(deadline=deadline, telemetry=JobTelemetry())
        finally:
            if payload.idempotencyKey == "waiting-2":
                cleaned.set()

    def run(payload, **_kwargs):
        calls.append(payload.idempotencyKey)
        first_entered.set()
        assert release.wait(3)
        return {"overallStatus": "completed"}

    monkeypatch.setattr(app_module, "prepare_analysis_source", prepare)
    monkeypatch.setattr(app_module, "run_analyze", run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            client.post, "/analyze", headers=auth_headers, json=_payload("waiting-1")
        )
        assert first_entered.wait(2)
        second = pool.submit(
            client.post, "/analyze", headers=auth_headers, json=_payload("waiting-2")
        )
        assert second_ready.wait(2)
        assert client.post("/analyze/waiting-2/cancel", headers=auth_headers).json()["cancelled"]
        assert second.result(2).json()["code"] == CANCELLED
        release.set()
        assert first.result(2).status_code == 200
    assert cleaned.is_set()
    assert calls == ["waiting-1"]


@pytest.mark.e2e
def test_third_job_waits_before_prefetch_and_can_be_cancelled(client, auth_headers, monkeypatch):
    """At most two job directories may exist, including the active inference job."""
    first_entered = threading.Event()
    second_ready = threading.Event()
    third_registered = threading.Event()
    release = threading.Event()
    prepared_keys = []
    original_begin = app_module.registry.begin

    def begin(key, *args, **kwargs):
        value = original_begin(key, *args, **kwargs)
        if key == "bounded-3":
            third_registered.set()
        return value

    @contextmanager
    def prepare(payload, *, deadline, **_kwargs):
        prepared_keys.append(payload.idempotencyKey)
        if payload.idempotencyKey == "bounded-2":
            second_ready.set()
        yield SimpleNamespace(deadline=deadline, telemetry=JobTelemetry())

    def run(payload, **_kwargs):
        if payload.idempotencyKey == "bounded-1":
            first_entered.set()
            assert release.wait(3)
        return {"overallStatus": "completed"}

    monkeypatch.setattr(app_module.registry, "begin", begin)
    monkeypatch.setattr(app_module, "prepare_analysis_source", prepare)
    monkeypatch.setattr(app_module, "run_analyze", run)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(
            client.post, "/analyze", headers=auth_headers, json=_payload("bounded-1")
        )
        assert first_entered.wait(2)
        second = pool.submit(
            client.post, "/analyze", headers=auth_headers, json=_payload("bounded-2")
        )
        assert second_ready.wait(2)
        third = pool.submit(
            client.post, "/analyze", headers=auth_headers, json=_payload("bounded-3")
        )
        assert third_registered.wait(2)
        assert client.post("/analyze/bounded-3/cancel", headers=auth_headers).json()["cancelled"]
        assert third.result(2).json()["code"] == CANCELLED
        assert prepared_keys == ["bounded-1", "bounded-2"]
        release.set()
        assert first.result(2).status_code == second.result(2).status_code == 200
