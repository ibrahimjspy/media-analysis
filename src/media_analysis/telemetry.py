"""Per-stage timing and byte-count telemetry for one analysis job."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class StageRecord:
    name: str
    duration_ms: int = 0
    bytes_in: int = 0
    bytes_out: int = 0


class StageLatencyMetrics:
    """Thread-safe rolling stage distributions for p50/p95 autoscaling signals."""

    def __init__(self, *, max_samples_per_stage: int = 1024) -> None:
        if max_samples_per_stage < 1:
            raise ValueError("max_samples_per_stage must be positive")
        self._max_samples = max_samples_per_stage
        self._samples: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=self._max_samples)
        )
        self._lock = threading.Lock()

    def observe(self, stage: str, duration_ms: int) -> None:
        with self._lock:
            self._samples[stage].append(max(0, duration_ms))

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            copied = {name: sorted(values) for name, values in self._samples.items()}
        return {
            name: {
                "count": len(values),
                "p50Ms": _nearest_rank(values, 0.50),
                "p95Ms": _nearest_rank(values, 0.95),
            }
            for name, values in sorted(copied.items())
            if values
        }

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    rank = int(len(values) * quantile + 0.999999)
    return values[max(0, min(len(values) - 1, rank - 1))]


stage_latency_metrics = StageLatencyMetrics()


@dataclass(slots=True)
class JobTelemetry:
    stages: list[StageRecord] = field(default_factory=list)
    bytes_downloaded: int = 0
    frames_decoded: int = 0
    decoded_pixels: int = 0
    bytes_uploaded: int = 0
    source_cache_hit: bool = False
    canonical_cache_hit: bool = False
    result_cache_hit: bool = False
    unique_frames_decoded: int = 0
    frame_cache_hits: int = 0
    frame_disk_hits: int = 0
    frame_spill_bytes: int = 0
    frame_spill_limited: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @contextmanager
    def stage(self, name: str) -> Iterator[StageRecord]:
        record = StageRecord(name=name)
        started = time.perf_counter()
        try:
            yield record
        finally:
            record.duration_ms = max(0, int((time.perf_counter() - started) * 1000))
            with self._lock:
                self.stages.append(record)
            stage_latency_metrics.observe(name, record.duration_ms)

    def add_uploaded_bytes(self, byte_count: int) -> None:
        with self._lock:
            self.bytes_uploaded += byte_count

    def record_stage(self, name: str, duration_ms: int) -> None:
        record = StageRecord(name=name, duration_ms=max(0, duration_ms))
        with self._lock:
            self.stages.append(record)
        stage_latency_metrics.observe(name, record.duration_ms)

    def as_dict(self) -> dict[str, object]:
        with self._lock:
            stages = list(self.stages)
            bytes_uploaded = self.bytes_uploaded
        stage_timings: dict[str, int] = {}
        for item in stages:
            stage_timings[item.name] = stage_timings.get(item.name, 0) + item.duration_ms
        return {
            "stageTimingsMs": stage_timings,
            "stageLatencyPercentiles": stage_latency_metrics.snapshot(),
            "bytesDownloaded": self.bytes_downloaded,
            "framesDecoded": self.frames_decoded,
            "decodedPixels": self.decoded_pixels,
            "bytesUploaded": bytes_uploaded,
            "sourceCacheHit": self.source_cache_hit,
            "canonicalCacheHit": self.canonical_cache_hit,
            "resultCacheHit": self.result_cache_hit,
            "uniqueSampledFramesDecoded": self.unique_frames_decoded,
            "frameCacheHits": self.frame_cache_hits,
            "frameDiskCacheHits": self.frame_disk_hits,
            "frameSpillBytes": self.frame_spill_bytes,
            "frameSpillLimited": self.frame_spill_limited,
        }
