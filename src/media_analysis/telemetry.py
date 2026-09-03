"""Per-stage timing and byte-count telemetry for one analysis job."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class StageRecord:
    name: str
    duration_ms: int = 0
    bytes_in: int = 0
    bytes_out: int = 0


@dataclass(slots=True)
class JobTelemetry:
    stages: list[StageRecord] = field(default_factory=list)
    bytes_downloaded: int = 0
    frames_decoded: int = 0
    decoded_pixels: int = 0
    bytes_uploaded: int = 0

    @contextmanager
    def stage(self, name: str) -> Iterator[StageRecord]:
        record = StageRecord(name=name)
        started = time.perf_counter()
        try:
            yield record
        finally:
            record.duration_ms = max(0, int((time.perf_counter() - started) * 1000))
            self.stages.append(record)

    def as_dict(self) -> dict[str, object]:
        return {
            "stageTimingsMs": {item.name: item.duration_ms for item in self.stages},
            "bytesDownloaded": self.bytes_downloaded,
            "framesDecoded": self.frames_decoded,
            "decodedPixels": self.decoded_pixels,
            "bytesUploaded": self.bytes_uploaded,
        }
