"""Actual HTTP/cache/grant lifecycle with a deterministic depth predictor.

The predictor stands in for expensive inference, not decoding, request hashing,
artifact encoding or uploads. Real pinned-model smoke is a separate opt-in check.
"""

import io
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from tests.e2e.test_native_media_pipeline import media_server, payload, png_bytes

from media_analysis.app import create_app
from media_analysis.features.depth import DepthAnalyzer, encode_depth


class Predictor:
    """Count executions to prove refreshed output grants reuse inference."""

    def __init__(self):
        self.calls = 0

    def analyze(self, image, guard):
        guard()
        self.calls += 1
        return encode_depth(np.arange(32, dtype=np.float32).reshape(4, 8), *image.size)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("_encoded", "AAAA"),
        ("policyVersion", "old-policy"),
        ("model", {"id": "unverified-model"}),
        ("canonicalWidth", 1),
        ("encoding", "uint16be_inverse_relative"),
        ("normalization", {"policy": "percentile-1-99", "low": 9, "high": 3}),
        ("mimeType", "image/png"),
        ("diagnostics", {"edgeFraction": 2}),
        ("productionQualified", True),
    ],
)
def test_depth_upload_and_cache_replay(
    settings, auth_headers, tmp_path, monkeypatch, field, replacement
):
    """Renew grants without inference, but recompute corrupt bytes or metadata.

    The valid private cache intentionally lacks mimeType until upload. Keeping
    binary bytes unchanged in metadata cases catches persistent cache hits that
    would otherwise be rejected repeatedly by backend artifact validation.
    """
    import media_analysis.app as app_module

    original = app_module.load_runtime
    predictor = Predictor()
    monkeypatch.setattr(
        app_module,
        "load_runtime",
        lambda cfg: replace(
            original(cfg.model_copy(update={"media_analysis_depth_enabled": False})),
            depth=predictor,
        ),
    )
    cfg = settings.model_copy(
        update={
            "media_analysis_depth_enabled": True,
            "media_analysis_cache_dir": tmp_path / "cache",
        }
    )
    data = png_bytes()
    with media_server(data) as (url, uploads), TestClient(create_app(cfg)) as client:
        body = payload(
            url,
            data,
            "image",
            ["quality", "relative_depth"],
            canonicalize=True,
            outputGrants={
                "depthMap": {
                    "signedPutUrl": url + "/depth-one",
                    "expiresAt": "2099-01-01T00:00:00Z",
                }
            },
        )
        first = client.post("/analyze", json=body, headers=auth_headers)
        assert first.status_code == 200, first.text
        result = first.json()
        assert result["capabilities"]["relative_depth"]["status"] == "completed"
        assert "_encoded" not in str(result)
        assert result["relative_depth"]["canonicalSha256"] == result["canonicalMedia"]["sha256"]
        assert result["relative_depth"]["byteCount"] == len(uploads[-1][1])
        body["idempotencyKey"] = "depth-replay"
        body["outputGrants"]["depthMap"]["signedPutUrl"] = url + "/depth-two"
        second = client.post("/analyze", json=body, headers=auth_headers)
        assert second.status_code == 200, second.text
        assert predictor.calls == 1
        assert uploads[-1][0] == "/depth-two"
        assert uploads[-1][1] == uploads[0][1]
        import json

        cached = next((tmp_path / "cache" / "native-result").glob("*.json"))
        content = json.loads(cached.read_text())
        assert "mimeType" not in content["relative_depth"]
        content["relative_depth"][field] = replacement
        cached.write_text(json.dumps(content))
        body["idempotencyKey"] = "corrupt-cache-retry"
        third = client.post("/analyze", json=body, headers=auth_headers)
        assert third.status_code == 200, third.text
        assert predictor.calls == 2
        assert not third.json()["telemetry"]["resultCacheHit"]
        assert third.json()["relative_depth"] == first.json()["relative_depth"]
        body["idempotencyKey"] = "repaired-cache-retry"
        fourth = client.post("/analyze", json=body, headers=auth_headers)
        assert fourth.status_code == 200, fourth.text
        assert fourth.json()["telemetry"]["resultCacheHit"]
        assert predictor.calls == 2


def test_depth_missing_model_preserves_photo(settings, auth_headers):
    data = png_bytes()
    with media_server(data) as (url, _), TestClient(create_app(settings)) as client:
        body = payload(
            url,
            data,
            "image",
            ["quality", "relative_depth"],
            canonicalize=True,
            outputGrants={
                "depthMap": {"signedPutUrl": url + "/depth", "expiresAt": "2099-01-01T00:00:00Z"}
            },
        )
        response = client.post("/analyze", json=body, headers=auth_headers)
        assert response.status_code == 200, response.text
        r = response.json()
        assert r["overallStatus"] == "partial"
        assert r["capabilities"]["quality"]["status"] == "completed"
        assert r["capabilities"]["relative_depth"]["status"] == "unavailable"
        assert not r.get("relative_depth")
        assert r["canonicalMedia"]["sha256"]
        body["mediaKind"] = "video"
        assert client.post("/analyze", json=body, headers=auth_headers).status_code == 400


@pytest.mark.parametrize("seed_legacy_cache", [False, True])
def test_uniform_photo_keeps_canonical_image_without_invented_depth(
    settings, auth_headers, monkeypatch, tmp_path, seed_legacy_cache
):
    """Decline blank-image depth before inference, including pre-guard cache hits."""
    import media_analysis.app as app_module

    original = app_module.load_runtime
    predictor = Predictor()
    analyzer = DepthAnalyzer(None, None)
    if not seed_legacy_cache:
        predictor.analyze = analyzer.analyze
    monkeypatch.setattr(
        app_module,
        "load_runtime",
        lambda cfg: replace(
            original(cfg.model_copy(update={"media_analysis_depth_enabled": False})),
            depth=predictor,
        ),
    )
    cfg = settings.model_copy(
        update={
            "media_analysis_depth_enabled": True,
            "media_analysis_cache_dir": tmp_path / "blank-cache",
        }
    )
    output = io.BytesIO()
    Image.new("RGB", (64, 32), (128, 128, 128)).save(output, format="PNG")
    data = output.getvalue()
    with media_server(data) as (url, uploads), TestClient(create_app(cfg)) as client:
        body = payload(
            url,
            data,
            "image",
            ["quality", "relative_depth"],
            canonicalize=True,
            outputGrants={
                "canonicalImage": {
                    "signedPutUrl": url + "/canonical",
                    "expiresAt": "2099-01-01T00:00:00Z",
                },
                "depthMap": {
                    "signedPutUrl": url + "/depth",
                    "expiresAt": "2099-01-01T00:00:00Z",
                },
            },
        )
        if seed_legacy_cache:
            first = client.post("/analyze", json=body, headers=auth_headers)
            assert first.status_code == 200, first.text
            assert first.json()["capabilities"]["relative_depth"]["status"] == "completed"
            assert predictor.calls == 1
            predictor.analyze = analyzer.analyze
            body["idempotencyKey"] = "decline-legacy-blank-cache"
            uploads.clear()

        response = client.post("/analyze", json=body, headers=auth_headers)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["overallStatus"] == "partial"
        assert result["capabilities"]["quality"]["status"] == "completed"
        assert result["capabilities"]["relative_depth"] == {
            "status": "failed",
            "warningCodes": ["DEPTH_NO_IMAGE_VARIATION"],
        }
        assert "DEPTH_NO_IMAGE_VARIATION" in result["warningCodes"]
        assert not result.get("relative_depth")
        assert result["canonicalMedia"]["sha256"]
        assert not result["telemetry"]["resultCacheHit"]
        assert [path for path, _, _ in uploads] == ["/canonical"]


def test_failed_depth_upload_does_not_fail_canonical_photo(
    settings, auth_headers, monkeypatch, tmp_path
):
    import media_analysis.app as app_module
    import media_analysis.native as native
    from media_analysis.errors import UPLOAD_FAILED, AnalyzeError

    original = app_module.load_runtime
    monkeypatch.setattr(
        app_module,
        "load_runtime",
        lambda cfg: replace(
            original(cfg.model_copy(update={"media_analysis_depth_enabled": False})),
            depth=Predictor(),
        ),
    )
    monkeypatch.setattr(
        native,
        "upload_artifact",
        lambda *a, **k: (_ for _ in ()).throw(AnalyzeError(UPLOAD_FAILED, "expired")),
    )
    cfg = settings.model_copy(
        update={
            "media_analysis_depth_enabled": True,
            "media_analysis_cache_dir": tmp_path / "depth-cache",
        }
    )
    data = png_bytes()
    with media_server(data) as (url, _), TestClient(create_app(cfg)) as client:
        body = payload(
            url,
            data,
            "image",
            ["quality", "relative_depth"],
            canonicalize=True,
            outputGrants={
                "depthMap": {"signedPutUrl": url + "/depth", "expiresAt": "2099-01-01T00:00:00Z"}
            },
        )
        response = client.post("/analyze", json=body, headers=auth_headers)
        assert response.status_code == 200, response.text
        assert response.json()["overallStatus"] == "partial"
        assert response.json()["canonicalMedia"]["sha256"]
        assert "DEPTH_DELIVERY_FAILED" in response.json()["warningCodes"]


@pytest.mark.parametrize("cancelled", [True, False])
def test_depth_cancellation_propagates_but_unknown_errors_are_sanitized(
    settings, auth_headers, monkeypatch, tmp_path, cancelled
):
    """Cancellation aborts; arbitrary model error text never enters public JSON."""
    import media_analysis.app as app_module
    from media_analysis.errors import CANCELLED, AnalyzeError

    original = app_module.load_runtime
    predictor = Predictor()
    error = (
        AnalyzeError(CANCELLED, "cancelled")
        if cancelled
        else ValueError("private-provider-body-and-url")
    )
    predictor.analyze = lambda *args: (_ for _ in ()).throw(error)
    monkeypatch.setattr(
        app_module,
        "load_runtime",
        lambda cfg: replace(
            original(cfg.model_copy(update={"media_analysis_depth_enabled": False})),
            depth=predictor,
        ),
    )
    cfg = settings.model_copy(
        update={
            "media_analysis_depth_enabled": True,
            "media_analysis_cache_dir": tmp_path / "depth-cache",
        }
    )
    data = png_bytes()
    with media_server(data) as (url, uploads), TestClient(create_app(cfg)) as client:
        body = payload(
            url,
            data,
            "image",
            ["quality", "relative_depth"],
            canonicalize=True,
            outputGrants={
                "depthMap": {"signedPutUrl": url + "/depth", "expiresAt": "2099-01-01T00:00:00Z"}
            },
        )
        response = client.post("/analyze", json=body, headers=auth_headers)
        if cancelled:
            assert response.status_code == 409, response.text
            assert response.json()["code"] == "CANCELLED"
        else:
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["overallStatus"] == "partial"
            assert result["capabilities"]["relative_depth"] == {
                "status": "failed",
                "warningCodes": ["RELATIVE_DEPTH_FAILED"],
            }
            assert "private-provider-body-and-url" not in response.text
            assert result["canonicalMedia"]["sha256"]
        assert uploads == []
