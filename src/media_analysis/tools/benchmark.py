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
    args = parser.parse_args()
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
