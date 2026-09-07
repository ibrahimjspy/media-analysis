"""Aggregate matte component wall times per job, not per-frame percentiles."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class MatteTimings:
    def __init__(self, record_stage: Callable[[str, int], None] | None = None):
        self.record_stage = record_stage
        self.seconds: dict[str, float] = defaultdict(float)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += max(0.0, time.perf_counter() - started)

    def __enter__(self) -> MatteTimings:
        return self

    def __exit__(self, *_args: object) -> None:
        if self.record_stage:
            for name, seconds in self.seconds.items():
                self.record_stage(name, int(seconds * 1000))
