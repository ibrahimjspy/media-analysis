"""Measure real pinned depth inference versus validated sample reuse.

Uses local PNGs and a temporary private cache. No HTTP jobs, uploads, shared
cache changes or model downloads. Model loading/warmup is measured separately;
this is a component benchmark, not whole-tour throughput or a p95 estimate.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from media_analysis.cache import LocalMediaCache
from media_analysis.features.depth import DepthAnalyzer, cached_payload, reusable_depth


def main():
    """Compare exact encoded outputs on each image with the same loaded model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    started = time.perf_counter()
    model = DepthAnalyzer.load(args.model)
    report = {
        "scope": "Real local depth component, not full pipeline",
        "rows": [],
        "modelLoadMs": round((time.perf_counter() - started) * 1000),
    }
    with TemporaryDirectory(prefix="depth-reuse-") as folder:
        cache = LocalMediaCache(Path(folder), max_bytes=32 * 1024 * 1024, ttl_sec=3600)
        for path in args.images:
            png = path.read_bytes()
            image = Image.open(path).convert("RGB")
            outputs = []
            for label in ("cold", "feature-set-change-reuse"):
                started = time.perf_counter()
                body, hit = reusable_depth(image, png, model, lambda: None, cache)
                elapsed = round((time.perf_counter() - started) * 1000)
                outputs.append(cached_payload(body, hashlib.sha256(png).hexdigest(), *image.size))
                report["rows"].append(
                    {
                        "imageSha256": hashlib.sha256(png).hexdigest(),
                        "mode": label,
                        "elapsedMs": elapsed,
                        "cacheHit": hit,
                    }
                )
            assert outputs[0] == outputs[1], "Cache changed depth bytes"
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
