import hashlib
from pathlib import Path

import httpx
import numpy as np
import pytest

from media_analysis.config import Settings
from media_analysis.runtime import load_runtime
from media_analysis.tools.model_lock import load_lock
from media_analysis.tools.vendor_models import vendor


@pytest.mark.real_models
@pytest.mark.production
def test_owned_modnet_matches_pytorch_and_remains_evaluation_only(tmp_path, monkeypatch):
    path = Path("models/manifest.lock.json")
    lock = load_lock(path, profile="matte-cpu")
    entry = lock.models[0]
    vendor("matte-cpu", path, tmp_path, stub=False, force=True)
    monkeypatch.setenv("MEDIA_ANALYSIS_MATTE_REFERENCE_MODE", "1")
    state = load_runtime(
        Settings(
            media_analysis_image="matte-cpu",
            media_analysis_model_dir=tmp_path,
            media_analysis_matte_execution_provider="cpu",
        )
    )
    assert state.ready and state.warmup_complete
    assert state.reference_mode and not state.production_inference_ready
    assert not state.manifest.by_name("modnet").stub
    assert state.execution_providers == ("CPUExecutionProvider",)
    metadata = entry.provenance.raw
    response = httpx.get(metadata["parityFixtureUrl"], follow_redirects=True, timeout=120)
    response.raise_for_status()
    assert hashlib.sha256(response.content).hexdigest() == metadata["parityFixtureSha256"]
    fixture = tmp_path / "parity.npz"
    fixture.write_bytes(response.content)
    with np.load(fixture, allow_pickle=False) as data:
        for index in range(3):
            actual = state.modnet_session.run(["output"], {"input": data[f"input_{index}"]})[0]
            error = np.abs(actual - data[f"output_{index}"])
            assert np.isfinite(actual).all()
            assert error.max() < 0.0005
            assert error.mean() < 0.00005
