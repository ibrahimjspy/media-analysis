"""Owned CPU export of the pinned official MODNet checkpoint, with ONNX parity.

Export-only dependencies are intentionally not installed in serving containers.
Upstream code/models: ZHKKKe/MODNet, Apache-2.0; see models/modnet-export.lock.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(upstream: Path, checkpoint: Path, lock: dict) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != lock["upstreamRevision"]:
        raise ValueError("MODNet source revision differs from export lock")
    subprocess.run(
        ["git", "-C", str(upstream), "diff", "--exit-code", "HEAD", "--", "src", "onnx"],
        check=True,
        capture_output=True,
    )
    if checkpoint.stat().st_size != lock["checkpointSizeBytes"]:
        raise ValueError("MODNet checkpoint size differs from export lock")
    if sha256_file(checkpoint) != lock["checkpointSha256"]:
        raise ValueError("MODNet checkpoint checksum differs from export lock")


def export(upstream: Path, checkpoint: Path, output: Path, lock: dict) -> dict:
    verify_inputs(upstream, checkpoint, lock)
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    if torch.__version__.split("+")[0] != lock["torch"] or onnx.__version__ != lock["onnx"]:
        raise ValueError("Use the pinned export toolchain")
    torch.set_num_threads(1)
    torch.manual_seed(lock["paritySeed"])
    sys.path.insert(0, str(upstream.resolve()))
    spec = importlib.util.spec_from_file_location(
        "modnet_export_network",
        upstream / "onnx" / "modnet_onnx.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MODNet(backbone_pretrained=False).cpu().eval()
    # Do not deserialize arbitrary Python objects from a downloaded checkpoint.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 512, 512),
        str(output),
        input_names=["input"],
        output_names=["output"],
        opset_version=lock["opset"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(output)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(output),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    rng = np.random.default_rng(lock["paritySeed"])
    cases = []
    fixtures = {}
    for index, (height, width) in enumerate([(512, 512), (384, 640), (640, 384)]):
        tensor = rng.uniform(-1, 1, (1, 3, height, width)).astype(np.float32)
        with torch.inference_mode():
            expected = model(torch.from_numpy(tensor)).numpy()
        actual = session.run(["output"], {"input": tensor})[0]
        error = np.abs(actual - expected)
        max_abs, mean_abs = float(error.max()), float(error.mean())
        if (
            not np.isfinite(actual).all()
            or max_abs > lock["parityMaxAbsTolerance"]
            or (mean_abs > lock["parityMeanAbsTolerance"])
        ):
            raise ValueError(f"MODNet parity failed: {height}x{width}, {max_abs}, {mean_abs}")
        cases.append({"height": height, "width": width, "maxAbs": max_abs, "meanAbs": mean_abs})
        fixtures[f"input_{index}"], fixtures[f"output_{index}"] = tensor, expected
    fixture_path = output.with_suffix(".parity.npz")
    np.savez_compressed(fixture_path, **fixtures)
    report = {
        **lock,
        "artifactSha256": sha256_file(output),
        "sizeBytes": output.stat().st_size,
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "parity": cases,
        "parityFixtureSha256": sha256_file(fixture_path),
        "productionApproved": False,
    }
    output.with_suffix(".recipe.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("models/modnet-export.lock.json"))
    args = parser.parse_args()
    print(
        json.dumps(
            export(args.upstream, args.checkpoint, args.out, json.loads(args.lock.read_text())),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
