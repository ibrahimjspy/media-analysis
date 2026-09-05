import json
from pathlib import Path

import pytest
from tests.unit.test_frame_access import make_probe_mp4

from media_analysis.tools.benchmark import _percentile, benchmark_sample_cache, main


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


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_sample_benchmark_reports_decode_reuse_beyond_memory_capacity(tmp_path):
    video = make_probe_mp4(tmp_path / "clip.mp4", frames=210)
    result = benchmark_sample_cache(video, samples=1, stride=6)
    count = result["sampledFramesPerPass"]
    assert count > 32
    memory = result["trials"]["memoryOnly"][0]
    spill = result["trials"]["diskSpill"][0]
    assert memory["decodedSamples"] == count * 2
    assert spill["decodedSamples"] == count
    assert spill["cacheHits"] == count
