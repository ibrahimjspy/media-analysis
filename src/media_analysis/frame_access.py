from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from media_analysis.errors import CANCELLED, AnalyzeError


@dataclass(frozen=True, slots=True)
class FrameAccessConfig:
    max_cached_frames: int = 32

    def __post_init__(self) -> None:
        if self.max_cached_frames < 1:
            raise ValueError("max_cached_frames must be >= 1")


class BoundedFrameAccess(AbstractContextManager["BoundedFrameAccess"]):
    """Bounded random-access decode cache for visual feature passes."""

    def __init__(
        self,
        path: Path,
        *,
        config: FrameAccessConfig | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> None:
        self._path = path
        self._config = config or FrameAccessConfig()
        self._cancel_check = cancel_check
        self._capture: cv2.VideoCapture | None = None
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._closed = False

    def __enter__(self) -> BoundedFrameAccess:
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Could not open video for frame access")
        self._capture = capture
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cache.clear()
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _check_cancel(self) -> None:
        if self._cancel_check:
            self._cancel_check()

    def _ensure_open(self) -> cv2.VideoCapture:
        if self._closed or self._capture is None:
            raise RuntimeError("BoundedFrameAccess is not open")
        return self._capture

    def _store(self, frame_index: int, frame_bgr: np.ndarray) -> None:
        self._cache[frame_index] = frame_bgr.copy()
        while len(self._cache) > self._config.max_cached_frames:
            self._cache.popitem(last=False)

    def read_bgr(self, frame_index: int) -> np.ndarray:
        """Return an independent BGR frame copy safe for in-place mutation."""
        self._check_cancel()
        cached = self._cache.get(frame_index)
        if cached is not None:
            self._cache.move_to_end(frame_index)
            return cached.copy()

        capture = self._ensure_open()
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Could not decode frame {frame_index}")
        self._store(frame_index, frame_bgr)
        return self._cache[frame_index].copy()

    def prefetch(self, frame_indices: Iterable[int]) -> None:
        for frame_index in frame_indices:
            self.read_bgr(frame_index)

    def cached_frame_indices(self) -> tuple[int, ...]:
        return tuple(self._cache.keys())

    def clear_cache(self) -> None:
        self._cache.clear()


def iter_half_open_shot_frames(
    shots: list[dict[str, Any]],
    *,
    step: int = 1,
) -> Iterator[tuple[int, int]]:
    """Yield (shotIndex, sourceFrame) for half-open shot ranges."""
    if step <= 0:
        raise ValueError("step must be positive")
    for shot_index, shot in enumerate(shots):
        start = int(shot["startFrame"])
        end = int(shot["endFrameExclusive"])
        for frame_index in range(start, end, step):
            yield shot_index, frame_index


def cancel_check_from_flag(cancelled: Callable[[], bool]) -> Callable[[], None]:
    def check() -> None:
        if cancelled():
            raise AnalyzeError(CANCELLED, "Analysis cancelled")

    return check
