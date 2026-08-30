from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REQUIRED_CPU_MODELS = ("yolox-tiny", "yunet", "PP-OCRv5_mobile_det", "silero-vad")
OPTIONAL_CPU_MODELS: tuple[str, ...] = ()
REQUIRED_MATTE_MODELS = ("modnet",)
SILERO_VAD_MODEL_NAME = "silero-vad"


@dataclass(frozen=True, slots=True)
class ModelEntry:
    name: str
    file: str
    sha256: str
    license: str
    stub: bool = False
    version: str = "unspecified"


@dataclass(frozen=True, slots=True)
class ManifestState:
    entries: tuple[ModelEntry, ...]
    ready: bool
    loaded: tuple[str, ...]
    errors: tuple[str, ...]

    def by_name(self, name: str) -> ModelEntry | None:
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(model_dir: Path) -> dict:
    path = model_dir / "manifest.json"
    if not path.is_file():
        return {"models": []}
    return json.loads(path.read_text(encoding="utf-8"))


def verify_models(model_dir: Path, *, image: str) -> ManifestState:
    raw = load_manifest(model_dir)
    entries = tuple(
        ModelEntry(
            name=item["name"],
            file=item["file"],
            sha256=item["sha256"].lower(),
            license=item.get("license", "unknown"),
            stub=bool(item.get("stub", False)),
            version=item.get("version", "unspecified"),
        )
        for item in raw.get("models", [])
    )
    required = REQUIRED_MATTE_MODELS if image.startswith("matte") else REQUIRED_CPU_MODELS
    optional = () if image.startswith("matte") else OPTIONAL_CPU_MODELS
    errors: list[str] = []
    loaded: list[str] = []
    for name in (*required, *optional):
        entry = next((item for item in entries if item.name == name), None)
        if entry is None:
            if name in required:
                errors.append(f"missing manifest entry: {name}")
            continue
        path = model_dir / entry.file
        if not path.is_file():
            if name in required:
                errors.append(f"missing file: {entry.file}")
            continue
        digest = sha256_file(path)
        if digest != entry.sha256:
            if name in required:
                errors.append(f"sha256 mismatch: {name}")
            continue
        loaded.append(name)
    if image == "analysis-cpu" and any(item.name == "modnet" for item in entries):
        errors.append("analysis-cpu must not ship matte weights")
    return ManifestState(
        entries=entries,
        ready=not errors and set(required).issubset(set(loaded)),
        loaded=tuple(loaded),
        errors=tuple(errors),
    )
