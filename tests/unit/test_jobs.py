import pytest

from media_analysis.jobs import JobRegistry


@pytest.mark.unit
def test_same_key_and_hash_returns_cached_result() -> None:
    registry = JobRegistry()
    job, cached = registry.begin("k1", "hash-a")
    assert cached is None
    registry.complete(job, {"overallStatus": "completed"})
    _, again = registry.begin("k1", "hash-a")
    assert again == {"overallStatus": "completed"}


@pytest.mark.unit
def test_same_key_different_hash_is_rejected() -> None:
    registry = JobRegistry()
    registry.begin("k1", "hash-a")
    with pytest.raises(ValueError, match="different request"):
        registry.begin("k1", "hash-b")


@pytest.mark.unit
def test_cancel_sets_flag_and_unknown_key_is_false() -> None:
    registry = JobRegistry()
    job, _ = registry.begin("k1", "hash-a")
    assert registry.cancel("k1") is True
    assert job.cancelled()
    assert registry.cancel("missing") is False


@pytest.mark.unit
def test_completed_job_cannot_be_cancelled() -> None:
    registry = JobRegistry()
    job, _ = registry.begin("k1", "hash-a")
    registry.complete(job, {"overallStatus": "completed"})
    assert registry.cancel("k1") is False


@pytest.mark.unit
def test_completed_job_cache_is_bounded() -> None:
    registry = JobRegistry(max_jobs=2)
    for index in range(3):
        job, _ = registry.begin(f"k{index}", f"hash-{index}")
        registry.complete(job, {"index": index})

    assert registry.get("k0") is None
    assert registry.get("k1") is not None
    assert registry.get("k2") is not None
