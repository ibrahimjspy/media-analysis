import time

import pytest

from media_analysis.telemetry import JobTelemetry, StageLatencyMetrics


@pytest.mark.unit
def test_stage_timings_and_byte_counts_are_monotonic() -> None:
    telemetry = JobTelemetry()
    with telemetry.stage("download") as stage:
        time.sleep(0.01)
        stage.bytes_in = 128
    telemetry.bytes_downloaded = 128
    telemetry.frames_decoded = 30
    telemetry.decoded_pixels = 320 * 240 * 30
    with telemetry.stage("shots"):
        time.sleep(0.005)

    payload = telemetry.as_dict()
    assert payload["bytesDownloaded"] == 128
    assert payload["framesDecoded"] == 30
    assert payload["decodedPixels"] == 320 * 240 * 30
    assert payload["bytesUploaded"] == 0
    assert payload["stageTimingsMs"]["download"] >= payload["stageTimingsMs"]["shots"]
    assert list(payload["stageTimingsMs"]) == ["download", "shots"]


@pytest.mark.unit
def test_stage_latency_metrics_report_rolling_p50_and_p95() -> None:
    metrics = StageLatencyMetrics(max_samples_per_stage=4)
    for duration in (10, 20, 30, 40, 50):
        metrics.observe("ocr", duration)
    assert metrics.snapshot()["ocr"] == {"count": 4, "p50Ms": 30, "p95Ms": 50}


@pytest.mark.unit
def test_duplicate_stage_names_are_summed_per_job() -> None:
    telemetry = JobTelemetry()
    telemetry.record_stage("upload_thumbnail", 4)
    telemetry.record_stage("upload_thumbnail", 7)
    assert telemetry.as_dict()["stageTimingsMs"]["upload_thumbnail"] == 11
