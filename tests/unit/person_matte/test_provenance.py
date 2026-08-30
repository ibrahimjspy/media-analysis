from __future__ import annotations

from pathlib import Path

import pytest

from media_analysis.person_matte.readiness import matte_worker_ready_for_production
from media_analysis.tools.model_lock import load_lock

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / "models" / "manifest.lock.json"


@pytest.mark.unit
def test_matte_production_gate_closed_for_matte_cpu_lock() -> None:
    lock = load_lock(LOCK_PATH, profile="matte-cpu")
    assert lock.gates is not None
    ready, _ = matte_worker_ready_for_production(
        modnet_entry=None,
        gates=lock.gates,
    )
    assert ready is False
