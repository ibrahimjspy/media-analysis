"""Idempotency hash. Signed URLs and expiresAt are excluded so retries hit."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _normalize_output_grants(grants: dict[str, Any] | None) -> dict[str, Any] | None:
    if not grants:
        return None
    stable: dict[str, Any] = {}
    if grants.get("canonicalMp4"):
        stable["canonicalMp4"] = True
    thumbnails = grants.get("thumbnails")
    if thumbnails:
        stable["thumbnails"] = sorted(
            [{"index": item["index"]} for item in thumbnails],
            key=lambda item: item["index"],
        )
    matte_assets = grants.get("matteAssets")
    if matte_assets:
        stable["matteAssets"] = sorted(
            [{"label": item["label"]} for item in matte_assets],
            key=lambda item: item["label"],
        )
    return stable or None


def request_hash(payload: dict[str, Any]) -> str:
    material = {
        "canonicalize": payload.get("canonicalize", False),
        "features": sorted(payload.get("features") or []),
        "canonicalMedia": payload.get("canonicalMedia"),
        "analysisResolution": payload.get("analysisResolution"),
        "matteFrameRanges": payload.get("matteFrameRanges"),
        "matteTarget": payload.get("matteTarget"),
        "sourceSha256": (payload.get("source") or {}).get("sha256"),
        "priorFacts": payload.get("priorFacts"),
        "outputGrants": _normalize_output_grants(payload.get("outputGrants")),
    }
    encoded = json.dumps(_normalize(material), separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
