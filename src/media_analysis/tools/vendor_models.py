from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import httpx

from media_analysis.models_manifest import (
    OPTIONAL_CPU_MODELS,
    REQUIRED_CPU_MODELS,
    REQUIRED_MATTE_MODELS,
    verify_models,
)
from media_analysis.tools.model_lock import LockedModel, load_lock, write_runtime_manifest

DEFAULT_LOCK = Path("models/manifest.lock.json")
STUB_BLOBS: dict[str, tuple[str, bytes]] = {
    "yolox-tiny": ("yolox_tiny.onnx", b"stub-yolox-tiny"),
    "yunet": ("face_detection_yunet_2023mar.onnx", b"stub-yunet"),
    "PP-OCRv5_mobile_det": ("PP-OCRv5_mobile_det.onnx", b"stub-ocr-det"),
    "silero-vad": ("silero_vad.onnx", b"stub-silero-vad"),
    "modnet": ("modnet.onnx", b"stub-modnet"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_names(profile: str) -> tuple[str, ...]:
    if profile.startswith("matte"):
        return REQUIRED_MATTE_MODELS
    return REQUIRED_CPU_MODELS


def _optional_names(profile: str) -> tuple[str, ...]:
    if profile.startswith("matte"):
        return ()
    return OPTIONAL_CPU_MODELS


def _stub_models(profile: str) -> tuple[LockedModel, ...]:
    names = (*_required_names(profile), *_optional_names(profile))
    models: list[LockedModel] = []
    for name in names:
        filename, blob = STUB_BLOBS[name]
        models.append(
            LockedModel(
                name=name,
                file=filename,
                url="stub://local",
                revision="stub",
                sha256=sha256_bytes(blob),
                license="Apache-2.0" if name not in {"yunet", "silero-vad"} else "MIT",
                size_bytes=len(blob),
                stub=True,
            )
        )
    return tuple(models)


def _download(url: str, timeout_sec: float = 120.0) -> bytes:
    with httpx.Client(follow_redirects=True, timeout=timeout_sec) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def _materialize(entry: LockedModel, dest: Path, *, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force:
        if sha256_file(dest) == entry.sha256:
            return
        raise SystemExit(f"checksum mismatch for existing file: {dest}")
    if entry.url.startswith("stub://"):
        blob = STUB_BLOBS[entry.name][1]
    else:
        blob = _download(entry.url)
    digest = sha256_bytes(blob)
    if digest != entry.sha256:
        raise SystemExit(
            f"downloaded sha256 mismatch for {entry.name}: "
            f"expected {entry.sha256}, got {digest}"
        )
    if entry.size_bytes is not None and len(blob) != entry.size_bytes:
        raise SystemExit(
            f"downloaded size mismatch for {entry.name}: "
            f"expected {entry.size_bytes}, got {len(blob)}"
        )
    dest.write_bytes(blob)


def vendor(profile: str, lock_path: Path, dest: Path, *, stub: bool, force: bool) -> None:
    lock = None if stub else load_lock(lock_path, profile=profile)
    if stub:
        locked = _stub_models(profile)
    else:
        assert lock is not None
        locked = lock.models
        for entry in locked:
            if entry.provenance and entry.provenance.is_community_export:
                print(
                    f"warning: {entry.name} is a community export; "
                    "checksum match is not production validation — "
                    "see docs/ppocr-owned-export.md",
                    file=sys.stderr,
                )
        if (
            lock.gates
            and profile == "analysis-cpu"
            and not lock.gates.owned_ppocr_export_recorded
        ):
            print(
                f"warning: profile {profile} gates.ownedPpOcrExportRecorded=false; "
                "OCR production parity not recorded",
                file=sys.stderr,
            )
        if lock.gates and profile.startswith("matte"):
            if not lock.gates.matte_production_enabled or not lock.gates.benchmark_gate_passed:
                print(
                    f"warning: profile {profile} matte production gate is closed; "
                    f"{lock.gates.note or 'benchmark/broad-enable gate remains off'}",
                    file=sys.stderr,
                )
    required = set(_required_names(profile))
    present = {item.name for item in locked}
    missing = required - present
    if missing:
        joined = ", ".join(sorted(missing))
        raise SystemExit(f"lock profile {profile} missing required models: {joined}")
    if profile == "analysis-cpu" and "modnet" in present:
        raise SystemExit("analysis-cpu profile must not include matte weights (modnet)")
    if profile.startswith("matte"):
        forbidden = {"yolox-tiny", "yunet", "PP-OCRv5_mobile_det"}
        overlap = forbidden & present
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise SystemExit(f"{profile} profile must not include v1 analysis weights: {joined}")
    for entry in locked:
        _materialize(entry, dest / entry.file, force=force)
    write_runtime_manifest(dest, locked)
    state = verify_models(dest, image=profile)
    if not state.ready:
        joined = "; ".join(state.errors)
        raise SystemExit(f"post-vendor verification failed: {joined}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify vendored ONNX weights from manifest.lock.json "
            "(build/setup only)."
        )
    )
    parser.add_argument(
        "--profile",
        default="analysis-cpu",
        help="Lock profile to materialize (default: analysis-cpu)",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK,
        help=f"Path to manifest.lock.json (default: {DEFAULT_LOCK})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("models"),
        help="Output directory for weights and manifest.json (default: ./models)",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Write tiny stub weights for local tests (never used in production images).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when an existing file matches the lock checksum.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        vendor(args.profile, args.lock, args.dest, stub=args.stub, force=args.force)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(exc, file=sys.stderr)
            return 1 if isinstance(exc.code, int) else 1
        return 0
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        print(f"vendor failed: {exc}", file=sys.stderr)
        return 1
    print(f"vendored profile={args.profile} dest={args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
