import time

import pytest

from media_analysis.analyze import _feature_stage
from media_analysis.errors import TIMEOUT, AnalyzeError
from media_analysis.jobs import Job
from media_analysis.telemetry import JobTelemetry


@pytest.mark.unit
def test_zero_stage_budget_times_out_immediately() -> None:
    job = Job(idempotency_key="stage", request_hash="abc")
    telemetry = JobTelemetry()
    with pytest.raises(AnalyzeError) as exc:
        with _feature_stage("ocr", telemetry, job, time.monotonic() + 30, 0):
            pass
    assert exc.value.code == TIMEOUT
    assert "ocr" in exc.value.message
    assert telemetry.stages == []


@pytest.mark.unit
def test_stage_deadline_is_enforced_during_check() -> None:
    job = Job(idempotency_key="stage", request_hash="abc")
    telemetry = JobTelemetry()
    with pytest.raises(AnalyzeError) as exc:
        with _feature_stage("shots", telemetry, job, time.monotonic() + 30, 0.01) as check:
            time.sleep(0.03)
            check()
    assert exc.value.code == TIMEOUT
    assert "shots" in exc.value.message
    assert telemetry.as_dict()["stageTimingsMs"]["shots"] >= 20


@pytest.mark.unit
def test_stage_deadline_is_enforced_after_work() -> None:
    job = Job(idempotency_key="stage", request_hash="abc")
    telemetry = JobTelemetry()
    with pytest.raises(AnalyzeError) as exc:
        with _feature_stage("shots_internal", telemetry, job, time.monotonic() + 30, 0.01):
            time.sleep(0.03)
    assert exc.value.code == TIMEOUT
    assert "shots_internal" in exc.value.message
    assert telemetry.as_dict()["stageTimingsMs"]["shots_internal"] >= 20


@pytest.mark.unit
def test_stage_remaining_is_capped_by_stage_budget() -> None:
    job = Job(idempotency_key="stage", request_hash="abc")
    telemetry = JobTelemetry()
    job_deadline = time.monotonic() + 30
    with _feature_stage("probe", telemetry, job, job_deadline, 0.5) as stage:
        remaining = stage.remaining()
        assert 0 < remaining <= 0.5
        assert remaining < job_deadline - time.monotonic()
