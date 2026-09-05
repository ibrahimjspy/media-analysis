from __future__ import annotations

import tempfile
import threading
import time
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
    max_cached_bytes: int = 256 * 1024 * 1024
    sequential_read_max_gap: int = 12
    max_spill_bytes: int = 2 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_cached_frames < 1:
            raise ValueError("max_cached_frames must be >= 1")
        if self.max_cached_bytes < 1:
            raise ValueError("max_cached_bytes must be >= 1")
        if self.sequential_read_max_gap < 0:
            raise ValueError("sequential_read_max_gap must be >= 0")
        if self.max_spill_bytes < 0:
            raise ValueError("max_spill_bytes must be >= 0")


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
        self._gray_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._gray_cache_bytes = 0
        self._lock = threading.RLock()
        self._next_frame_index: int | None = None
        self._unique_decodes = 0
        self._cache_hits = 0
        self._decode_duration_sec = 0.0
        self._closed = False
        self._spill_dir: tempfile.TemporaryDirectory | None = None
        self._spilled: dict[str, Path] = {}
        self._spill_bytes = 0
        self._spill_limited = False
        self._disk_hits = 0

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
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cache.clear()
            self._gray_cache.clear()
            self._cache_bytes = 0
            self._gray_cache_bytes = 0
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            if self._spill_dir is not None:
                self._spill_dir.cleanup()
                self._spill_dir = None
            self._spilled.clear()
            self._spill_bytes = 0

    def _check_cancel(self) -> None:
        if self._cancel_check:
            self._cancel_check()

    def _ensure_open(self) -> cv2.VideoCapture:
        if self._closed or self._capture is None:
            raise RuntimeError("BoundedFrameAccess is not open")
        return self._capture

    def _store(self, frame_index: int, frame_bgr: np.ndarray) -> None:
        if frame_bgr.nbytes > self._config.max_cached_bytes:
            self._spill(f"bgr-{frame_index}", frame_bgr)
            return
        stored = frame_bgr.copy()
        prior = self._cache.pop(frame_index, None)
        if prior is not None:
            self._cache_bytes -= prior.nbytes
        self._cache[frame_index] = stored
        self._cache_bytes += stored.nbytes
        while (
            len(self._cache) > self._config.max_cached_frames
            or self._cache_bytes + self._gray_cache_bytes > self._config.max_cached_bytes
        ) and (len(self._cache) > 1 or self._gray_cache):
            if len(self._cache) > 1:
                _old_index, old_frame = self._cache.popitem(last=False)
                self._cache_bytes -= old_frame.nbytes
                self._spill(f"bgr-{_old_index}", old_frame)
            else:
                _old_key, old_gray = self._gray_cache.popitem(last=False)
                self._gray_cache_bytes -= old_gray.nbytes
                self._spill(f"gray-{_old_key[0]}-{_old_key[1]}", old_gray)

    def _store_gray(
        self,
        frame_index: int,
        max_dimension: int,
        gray: np.ndarray,
    ) -> None:
        key = (frame_index, max_dimension)
        if gray.nbytes > self._config.max_cached_bytes:
            self._spill(f"gray-{frame_index}-{max_dimension}", gray)
            return
        stored = gray.copy()
        prior = self._gray_cache.pop(key, None)
        if prior is not None:
            self._gray_cache_bytes -= prior.nbytes
        self._gray_cache[key] = stored
        self._gray_cache_bytes += stored.nbytes
        while (self._cache_bytes + self._gray_cache_bytes > self._config.max_cached_bytes) and (
            self._cache or len(self._gray_cache) > 1
        ):
            if self._cache:
                _old_index, old_frame = self._cache.popitem(last=False)
                self._cache_bytes -= old_frame.nbytes
                self._spill(f"bgr-{_old_index}", old_frame)
            else:
                _old_key, old_gray = self._gray_cache.popitem(last=False)
                self._gray_cache_bytes -= old_gray.nbytes
                self._spill(f"gray-{_old_key[0]}-{_old_key[1]}", old_gray)

    def _spill(self, key: str, frame: np.ndarray) -> None:
        if key in self._spilled or not self._config.max_spill_bytes:
            return
        if self._spill_bytes + frame.nbytes + 256 > self._config.max_spill_bytes:
            self._spill_limited = True
            return
        try:
            if self._spill_dir is None:
                self._spill_dir = tempfile.TemporaryDirectory(prefix="media-frame-cache-")
            path = Path(self._spill_dir.name) / f"{key}.npy"
            np.save(path, frame, allow_pickle=False)
            self._spill_bytes += path.stat().st_size
            self._spilled[key] = path
        except OSError:
            self._spill_limited = True
            if self._spill_dir is not None:
                (Path(self._spill_dir.name) / f"{key}.npy").unlink(missing_ok=True)

    def _read_spill(self, key: str) -> np.ndarray | None:
        path = self._spilled.get(key)
        if path is None:
            return None
        frame = np.load(path, allow_pickle=False)
        self._cache_hits += 1
        self._disk_hits += 1
        return frame

    def _decode(self, capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
        started = time.perf_counter()
        next_index = self._next_frame_index
        if (
            next_index is None
            or frame_index < next_index
            or frame_index - next_index > self._config.sequential_read_max_gap
        ):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        elif frame_index > next_index:
            for _index in range(next_index, frame_index):
                if not capture.grab():
                    raise RuntimeError(f"Could not decode frame {frame_index}")
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Could not decode frame {frame_index}")
        self._next_frame_index = frame_index + 1
        self._unique_decodes += 1
        self._decode_duration_sec += max(0.0, time.perf_counter() - started)
        return frame_bgr

    def read_bgr(self, frame_index: int) -> np.ndarray:
        """Return an independent BGR frame copy safe for in-place mutation."""
        if frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        self._check_cancel()
        with self._lock:
            capture = self._ensure_open()
            cached = self._cache.get(frame_index)
            if cached is not None:
                self._cache_hits += 1
                self._cache.move_to_end(frame_index)
                return cached.copy()

            spilled = self._read_spill(f"bgr-{frame_index}")
            if spilled is not None:
                # Do not promote a replayed disk frame into the hot cache: a
                # sequential second pass would evict and rewrite every resident
                # frame before reaching it, turning cache reuse into disk churn.
                return spilled
            frame_bgr = self._decode(capture, frame_index)
            self._store(frame_index, frame_bgr)
            return frame_bgr

    def read_gray(self, frame_index: int, *, max_dimension: int) -> np.ndarray:
        """Return a downscaled grayscale view for motion/exposure-style measurements."""
        if frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        if max_dimension < 1:
            raise ValueError("max_dimension must be positive")
        self._check_cancel()
        with self._lock:
            self._ensure_open()
            cached = self._gray_cache.get((frame_index, max_dimension))
            if cached is not None:
                self._cache_hits += 1
                self._gray_cache.move_to_end((frame_index, max_dimension))
                return cached.copy()
            spilled = self._read_spill(f"gray-{frame_index}-{max_dimension}")
            if spilled is not None:
                return spilled
            frame = self.read_bgr(frame_index)
            gray = self._make_gray(frame, max_dimension=max_dimension)
            self._store_gray(frame_index, max_dimension, gray)
            return gray

    @staticmethod
    def _make_gray(frame: np.ndarray, *, max_dimension: int) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = min(1.0, max_dimension / max(width, height))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def prime_gray(
        self,
        frame_index: int,
        frame_bgr: np.ndarray,
        *,
        max_dimension: int,
    ) -> None:
        """Retain a compact view from an upstream sequential decode pass."""
        gray = self._make_gray(frame_bgr, max_dimension=max_dimension)
        with self._lock:
            self._store_gray(frame_index, max_dimension, gray)

    def read_bgr_downscaled(self, frame_index: int, *, max_pixels: int) -> np.ndarray:
        """Return a bounded color view without changing normalized image geometry."""
        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        frame = self.read_bgr(frame_index)
        height, width = frame.shape[:2]
        if height * width <= max_pixels:
            return frame
        scale = (max_pixels / (height * width)) ** 0.5
        return cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def resized_view(self, *, width: int, height: int) -> ResizedFrameView:
        return ResizedFrameView(self, width=width, height=height)

    def prefetch(self, frame_indices: Iterable[int]) -> None:
        for frame_index in frame_indices:
            self.read_bgr(frame_index)

    def cached_frame_indices(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._cache.keys())

    @property
    def unique_decodes(self) -> int:
        return self._unique_decodes

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def decode_duration_ms(self) -> int:
        return max(0, round(self._decode_duration_sec * 1000))

    @property
    def spill_bytes(self) -> int:
        return self._spill_bytes

    @property
    def disk_hits(self) -> int:
        return self._disk_hits

    @property
    def spill_limited(self) -> bool:
        return self._spill_limited

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._gray_cache.clear()
            self._cache_bytes = 0
            self._gray_cache_bytes = 0
            if self._spill_dir is not None:
                self._spill_dir.cleanup()
                self._spill_dir = None
            self._spilled.clear()
            self._spill_bytes = 0


@dataclass(frozen=True, slots=True)
class ResizedFrameView:
    """Analyzer-compatible reader backed by the shared canonical frame cache."""

    access: BoundedFrameAccess
    width: int
    height: int

    def read(self, frame_index: int) -> np.ndarray | None:
        frame = self.access.read_bgr(frame_index)
        if frame.shape[1] == self.width and frame.shape[0] == self.height:
            return frame
        return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)


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
