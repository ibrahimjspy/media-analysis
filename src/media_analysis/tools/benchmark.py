"""Local latency/CPU/memory baseline harness.

This records a machine fingerprint and wall-time samples. It does not open
production gates. Commit calibrated numbers only after thresholds are approved.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from media_analysis.build_provenance import (
    analyze_build_provenance,
    collect_machine_fingerprint,
)
from media_analysis.decode import probe
from media_analysis.frame_access import BoundedFrameAccess, FrameAccessConfig


def benchmark_sample_cache(video: Path, *, samples: int, stride: int = 6) -> dict:
    """Compare repeated sampled-frame passes with and without disk spill.

    Each trial creates a fresh cache. This measures decoder/cache work only;
    it does not substitute for end-to-end inference or GPU benchmarks.
    """
    if samples < 1 or stride < 1:
        raise ValueError("samples and stride must be positive")
    media = probe(video)
    indices = sorted(set(range(0, media.frame_count, stride)) | {media.frame_count - 1})
    if media.frame_count < 1:
        raise ValueError("video must contain at least one frame")
    trials: dict[str, list[dict]] = {"memoryOnly": [], "diskSpill": []}
    for trial in range(samples):
        order = list(trials) if trial % 2 == 0 else list(reversed(trials))
        for name in order:
            config = FrameAccessConfig(max_spill_bytes=0) if name == "memoryOnly" else None
            with BoundedFrameAccess(video, config=config) as access:
                started = time.perf_counter()
                for index in indices:
                    access.read_bgr(index)
                second_started = time.perf_counter()
                for index in indices:
                    access.read_bgr(index)
                ended = time.perf_counter()
                trials[name].append({
                    "totalMs": (ended - started) * 1000,
                    "secondPassMs": (ended - second_started) * 1000,
                    "decodedSamples": access.unique_decodes,
                    "cacheHits": access.cache_hits,
                    "diskHits": access.disk_hits,
                    "spillBytes": access.spill_bytes,
                    "spillLimited": access.spill_limited,
                })
    return {
        "sampledFramesPerPass": len(indices),
        "passesPerTrial": 2,
        "trials": trials,
        "secondPassMs": {
            name: {
                "p50": _percentile([item["secondPassMs"] for item in rows], 0.50),
                "p95": _percentile([item["secondPassMs"] for item in rows], 0.95),
            }
            for name, rows in trials.items()
        },
    }


def _percentile(samples: list[float], fraction: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _sleep_probe() -> float:
    started = time.perf_counter()
    time.sleep(0)
    return (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a local analysis baseline.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/local-latest.json"),
        help="JSON output path",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--video", type=Path, help="Benchmark repeated sampled-frame passes")
    parser.add_argument("--stride", type=int, default=6)
    args = parser.parse_args()
    if args.video is not None:
        if args.samples < 1 or args.stride < 1:
            parser.error("--samples and --stride must be positive")
        payload = {
            "status": "sample-cache-benchmark",
            "note": "Decoder/cache microbenchmark; excludes models, network, and GPU work.",
            "machine": collect_machine_fingerprint(),
            "build": analyze_build_provenance(),
            **benchmark_sample_cache(args.video, samples=args.samples, stride=args.stride),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(args.out)
        return
    samples = [_sleep_probe() for _ in range(max(1, args.samples))]
    payload = {
        "status": "baseline-harness",
        "note": (
            "Timer sanity check only. Replace samples with /analyze wall times "
            "on a generated 60s 1080p reel before calibrating SLOs."
        ),
        "machine": collect_machine_fingerprint(),
        "build": analyze_build_provenance(),
        "timerMs": {
            "n": len(samples),
            "mean": statistics.fmean(samples),
            "p50": _percentile(samples, 0.50),
            "p95": _percentile(samples, 0.95),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
