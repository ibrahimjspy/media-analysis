from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi.testclient import TestClient

import media_analysis.app as app_module
from media_analysis.errors import CANCELLED, AnalyzeError


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
