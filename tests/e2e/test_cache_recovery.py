from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import media_analysis.analyze as analyze_module
from media_analysis.app import create_app, reset_manifest


def _payload(source_server, key):
    return {
        "idempotencyKey": key,
        "features": ["shots"],
        "source": {
            "signedGetUrl": source_server["url"],
            "expiresAt": "2099-01-01T00:00:00Z",
            "sha256": source_server["sha256"],
        },
    }


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_cache_disabled_with_fingerprint_still_completes(settings, auth_headers, source_server):
    cfg = settings.model_copy(update={"media_analysis_cache_enabled": False})
    with TestClient(create_app(cfg)) as client:
        response = client.post(
            "/analyze", headers=auth_headers, json=_payload(source_server, "off")
        )
    assert response.status_code == 200, response.text
    assert response.json()["overallStatus"] == "completed"
    assert response.json()["telemetry"]["resultCacheHit"] is False


@pytest.mark.e2e
@pytest.mark.ffmpeg
@pytest.mark.parametrize("status", ["failed", "partial"])
def test_failed_features_are_recomputed_on_retry(
    settings,
    auth_headers,
    source_server,
    tmp_path,
    monkeypatch,
    status,
):
    cfg = settings.model_copy(update={"media_analysis_cache_dir": tmp_path / "cache"})
    actual_compute = analyze_module._compute
    calls = []

    def transient_failure(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {"overallStatus": status, "canonicalMedia": {}}
        return actual_compute(*args, **kwargs)

    monkeypatch.setattr(analyze_module, "_compute", transient_failure)
    with TestClient(create_app(cfg)) as client:
        first = client.post("/analyze", headers=auth_headers, json=_payload(source_server, "fail"))
        second = client.post(
            "/analyze", headers=auth_headers, json=_payload(source_server, "retry")
        )
    assert first.json()["overallStatus"] == status
    assert second.json()["overallStatus"] == "completed", second.text
    assert second.json()["telemetry"]["resultCacheHit"] is False
    assert len(calls) == 2


@pytest.mark.e2e
@pytest.mark.ffmpeg
def test_cached_source_still_enforces_current_byte_limit(
    settings,
    auth_headers,
    source_server,
    tmp_path,
):
    cfg = settings.model_copy(update={"media_analysis_cache_dir": tmp_path / "cache"})
    with TestClient(create_app(cfg)) as client:
        first = client.post("/analyze", headers=auth_headers, json=_payload(source_server, "warm"))
        assert first.status_code == 200
    limited = cfg.model_copy(update={"media_analysis_max_bytes": 1})
    reset_manifest()
    with TestClient(create_app(limited)) as client:
        second = client.post(
            "/analyze", headers=auth_headers, json=_payload(source_server, "limit")
        )
    assert second.status_code == 413
    assert second.json()["code"] == "LIMIT_EXCEEDED"
