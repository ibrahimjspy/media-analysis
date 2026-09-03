"""Download official PP-OCR inference weights and convert them to ONNX.

This is an operator/CI tool. The worker image does not install PaddlePaddle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

EXPORT_LOCK = Path("models/ppocr-export.lock.json")


def load_export_lock(path: Path = EXPORT_LOCK) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return dest


def extract_tar(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as handle:
        handle.extractall(dest, filter="data")
    return dest


def find_inference_dir(root: Path) -> Path:
    for path in root.rglob("inference.pdiparams"):
        return path.parent
    raise FileNotFoundError("inference.pdiparams not found in extracted archive")


def resolve_model_files(infer_dir: Path) -> tuple[str, str]:
    if (infer_dir / "inference.json").is_file():
        return "inference.json", "inference.pdiparams"
    if (infer_dir / "inference.pdmodel").is_file():
        return "inference.pdmodel", "inference.pdiparams"
    raise FileNotFoundError(f"no inference.json or inference.pdmodel in {infer_dir}")


def export_onnx(infer_dir: Path, dest: Path, *, opset: int) -> Path:
    try:
        import paddle2onnx
    except ImportError as exc:
        raise RuntimeError(
            "paddle2onnx (and packaging) are required for owned export; "
            "run this on linux/amd64 with the pinned toolchain"
        ) from exc

    model_filename, params_filename = resolve_model_files(infer_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    paddle2onnx.export(
        model_filename=str(infer_dir / model_filename),
        params_filename=str(infer_dir / params_filename),
        save_file=str(dest),
        opset_version=opset,
        enable_onnx_checker=True,
    )
    return dest


def write_recipe(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Owned PP-OCRv5 mobile det ONNX export")
    parser.add_argument("--lock", type=Path, default=EXPORT_LOCK)
    parser.add_argument("--work", type=Path, default=Path(".ppocr-export"))
    parser.add_argument("--out", type=Path, default=Path("models/PP-OCRv5_mobile_det.onnx"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    lock = load_export_lock(args.lock)
    upstream = lock["upstreamInference"]
    toolchain = lock["toolchain"]
    archive = args.work / upstream["name"]
    extracted = args.work / "infer"

    if not args.skip_download or not archive.is_file():
        download_file(upstream["url"], archive)
    digest = sha256_file(archive)
    expected = upstream.get("sha256")
    if expected and digest != expected:
        raise SystemExit(f"upstream tar sha256 mismatch: {digest} != {expected}")

    extract_tar(archive, extracted)
    infer_dir = find_inference_dir(extracted)
    export_onnx(infer_dir, args.out, opset=int(toolchain["opset"]))
    recipe = {
        "artifactKind": "owned-export",
        "onnxSha256": sha256_file(args.out),
        "onnxSizeBytes": args.out.stat().st_size,
        "upstreamTarSha256": digest,
        "upstreamUrl": upstream["url"],
        "upstreamRevision": upstream["revision"],
        "toolchain": toolchain,
        "inferenceDir": str(infer_dir),
        "modelFiles": list(resolve_model_files(infer_dir)),
    }
    write_recipe(args.out.with_suffix(".recipe.json"), recipe)
    print(json.dumps(recipe, indent=2))


if __name__ == "__main__":
    main()
