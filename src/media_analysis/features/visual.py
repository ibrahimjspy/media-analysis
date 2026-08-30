"""subjects / faces / ocr. Stub models return completed empty lists."""

from __future__ import annotations

from media_analysis.models_manifest import ManifestState, ModelEntry


def model_provenance(entry: ModelEntry | None, *, runtime: str, preprocessing: str) -> dict | None:
    if entry is None:
        return None
    return {
        "name": entry.name,
        "version": entry.version,
        "artifactSha256": entry.sha256,
        "runtimeProvider": runtime,
        "preprocessingVersion": preprocessing,
        "license": entry.license,
    }


def empty_subjects() -> list:
    return []


def empty_faces() -> list:
    return []


def empty_reserved_regions() -> list:
    return []


def can_run_visual(manifest: ManifestState, name: str) -> bool:
    entry = manifest.by_name(name)
    return entry is not None
