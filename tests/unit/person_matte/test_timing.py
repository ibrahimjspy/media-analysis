import pytest

from media_analysis.person_matte.timing import MatteTimings


def test_component_samples_are_aggregated_once_per_job_including_failures(monkeypatch):
    clock = iter([0, 0.010, 1, 1.020, 2, 2.050])
    monkeypatch.setattr("media_analysis.person_matte.timing.time.perf_counter", lambda: next(clock))
    recorded = []
    with pytest.raises(ValueError), MatteTimings(lambda *item: recorded.append(item)) as timings:
        with timings.measure("matte_inference"):
            pass
        with timings.measure("matte_inference"):
            pass
        with timings.measure("matte_refinement"):
            raise ValueError("test error")
    assert [name for name, _ in recorded] == ["matte_inference", "matte_refinement"]
    assert recorded[0][1] == pytest.approx(30, abs=1)
    assert recorded[1][1] == pytest.approx(50, abs=1)
