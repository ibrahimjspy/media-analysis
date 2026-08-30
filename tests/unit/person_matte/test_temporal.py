from __future__ import annotations

import numpy as np
import pytest

from media_analysis.person_matte.constants import TEMPORAL_EMA_ALPHA
from media_analysis.person_matte.temporal import TemporalMatteState
from media_analysis.person_matte.types import ShotBoundary


@pytest.mark.unit
def test_temporal_ema_smooths_between_frames() -> None:
    state = TemporalMatteState(())
    first = np.full((8, 8), 0, dtype=np.uint8)
    second = np.full((8, 8), 255, dtype=np.uint8)
    out1 = state.apply(first, source_frame=0)
    out2 = state.apply(second, source_frame=1)
    assert out1.max() == 0
    expected = int(round((1.0 - TEMPORAL_EMA_ALPHA) * 255.0))
    assert out2[0, 0] == expected


@pytest.mark.unit
def test_temporal_resets_at_shot_boundary_from_prior_facts() -> None:
    shots = (
        ShotBoundary(start_frame=0, end_frame_exclusive=10),
        ShotBoundary(start_frame=10, end_frame_exclusive=20),
    )
    state = TemporalMatteState(shots)
    warm = np.full((4, 4), 255, dtype=np.uint8)
    state.apply(warm, source_frame=9)
    cold = np.full((4, 4), 0, dtype=np.uint8)
    reset = state.apply(cold, source_frame=10)
    assert reset.max() == 0
