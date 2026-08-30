from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    artifact_kind: str
    raw: dict[str, Any]

    @property
    def is_community_export(self) -> bool:
        return self.artifact_kind == "community-export"

    @property
    def owned_export_recorded(self) -> bool:
        if self.artifact_kind == "owned-export":
            return True
        return bool(self.raw.get("ownedExportRecorded", False))


@dataclass(frozen=True, slots=True)
class ProfileGates:
    owned_ppocr_export_recorded: bool
    matte_production_enabled: bool = False
    benchmark_gate_passed: bool = False
    silero_vad_production_enabled: bool = False
    note: str = ""


@dataclass(frozen=True, slots=True)
class LockedModel:
    name: str
    file: str
    url: str
    revision: str
    sha256: str
    license: str
    size_bytes: int | None = None
    stub: bool = False
    provenance: ModelProvenance | None = None


@dataclass(frozen=True, slots=True)
class ModelLock:
    lock_version: int
    profile: str
    models: tuple[LockedModel, ...]
    gates: ProfileGates | None = None


def _parse_provenance(raw: dict[str, Any] | None) -> ModelProvenance | None:
    if not raw:
        return None
    kind = str(raw.get("artifactKind", "unknown"))
    return ModelProvenance(artifact_kind=kind, raw=dict(raw))


def _parse_model(raw: dict[str, Any]) -> LockedModel:
    required = ("name", "file", "url", "revision", "sha256", "license")
    missing = [key for key in required if key not in raw]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"lock entry missing fields: {joined}")
    sha256 = str(raw["sha256"]).lower()
    if len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256):
        raise ValueError(f"invalid sha256 for {raw['name']}")
    size = raw.get("sizeBytes")
    return LockedModel(
        name=str(raw["name"]),
        file=str(raw["file"]),
        url=str(raw["url"]),
        revision=str(raw["revision"]),
        sha256=sha256,
        license=str(raw["license"]),
        size_bytes=int(size) if size is not None else None,
        stub=bool(raw.get("stub", False)),
        provenance=_parse_provenance(raw.get("provenance")),
    )


def _parse_gates(raw: dict[str, Any] | None) -> ProfileGates | None:
    if not raw:
        return None
    return ProfileGates(
        owned_ppocr_export_recorded=bool(raw.get("ownedPpOcrExportRecorded", False)),
        matte_production_enabled=bool(raw.get("matteProductionEnabled", False)),
        benchmark_gate_passed=bool(raw.get("benchmarkGatePassed", False)),
        silero_vad_production_enabled=bool(raw.get("sileroVadProductionEnabled", False)),
        note=str(raw.get("note", "")),
    )


def load_lock(path: Path, *, profile: str) -> ModelLock:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lock_version = int(payload.get("lockVersion", 0))
    if lock_version not in {1, 2}:
        raise ValueError(f"unsupported lockVersion: {lock_version}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f"profile not found in lock: {profile}")
    raw_profile = profiles[profile]
    models_raw = raw_profile.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise ValueError(f"profile {profile} has no models")
    models = tuple(_parse_model(item) for item in models_raw)
    return ModelLock(
        lock_version=lock_version,
        profile=profile,
        models=models,
        gates=_parse_gates(raw_profile.get("gates")),
    )


def write_runtime_manifest(model_dir: Path, models: tuple[LockedModel, ...]) -> Path:
    entries: list[dict[str, Any]] = []
    for item in models:
        entry: dict[str, Any] = {
            "name": item.name,
            "file": item.file,
            "sha256": item.sha256,
            "license": item.license,
            "version": item.revision,
            "stub": item.stub,
        }
        if item.provenance is not None:
            entry["provenance"] = {
                "artifactKind": item.provenance.artifact_kind,
                **item.provenance.raw,
            }
        entries.append(entry)
    manifest = {"models": entries}
    path = model_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
