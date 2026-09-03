import json
from pathlib import Path

import pytest

from media_analysis.tools.benchmark import _percentile, main


@pytest.mark.unit
def test_percentile_is_stable_for_odd_samples() -> None:
    assert _percentile([1, 2, 3, 4, 5], 0.50) == 3
    assert _percentile([1, 2, 3, 4, 5], 0.95) == 5
    assert _percentile([], 0.50) == 0.0


@pytest.mark.unit
def test_benchmark_harness_writes_machine_and_build(tmp_path, monkeypatch) -> None:
    out = tmp_path / "local-latest.json"
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark", "--out", str(out), "--samples", "3"],
    )
    main()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "baseline-harness"
    assert payload["machine"]["cpuCount"]
    assert payload["build"]["runtimeBuild"]["onnxruntimeVersion"]
    assert payload["timerMs"]["n"] == 3
    assert Path(out).is_file()
