import time

import pytest

from media_analysis.telemetry import JobTelemetry


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
