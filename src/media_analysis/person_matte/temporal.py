"""Versioned temporal stabilization with shot-boundary reset."""

from __future__ import annotations

import numpy as np

from media_analysis.person_matte.constants import MATTE_TEMPORAL_POLICY_VERSION, TEMPORAL_EMA_ALPHA
from media_analysis.person_matte.types import ShotBoundary


class TemporalMatteState:
    """Deterministic EMA state; reset at shot boundaries from typed priorFacts."""

    __slots__ = ("_prev", "_shot_starts")

    def __init__(self, shots: tuple[ShotBoundary, ...]) -> None:
        self._prev: np.ndarray | None = None
        self._shot_starts = frozenset(
            shot.start_frame for shot in shots[1:]  # first shot start does not reset cold state
        )

    def should_reset(self, source_frame: int) -> bool:
        return source_frame in self._shot_starts

    def apply(self, alpha: np.ndarray, *, source_frame: int) -> np.ndarray:
        if self.should_reset(source_frame):
            self._prev = None
        current = alpha.astype(np.float32)
        if self._prev is None or self._prev.shape != current.shape:
            stabilized = current
        else:
            stabilized = TEMPORAL_EMA_ALPHA * self._prev + (1.0 - TEMPORAL_EMA_ALPHA) * current
        self._prev = stabilized.copy()
        return np.clip(stabilized, 0.0, 255.0).astype(np.uint8)


def temporal_policy_version() -> str:
    return MATTE_TEMPORAL_POLICY_VERSION
