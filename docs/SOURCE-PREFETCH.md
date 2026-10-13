# Bounded source prefetch

`/analyze` admits at most two job-owned source contexts. Network/cache source reads
happen before the existing analysis gate, so the next image, audio or video can
be downloaded while the current request is analyzed. Decoding, model inference
and output delivery stay serialized. This assumes one Uvicorn process per worker;
multiple processes each own separate gates and require independent resource sizing.

Both gates check cancellation while waiting. A third request waits before creating
its temporary source directory. Closing the source context cleans files on success
or failure. Existing source expiry, allowlists, byte limits, SHA checks and bounded
disk-cache behavior remain intact. Model readiness is checked before source I/O.

Admission/gate wait has a bounded deadline. Once inference starts, the original
job processing budget is available minus time already spent preparing the source;
waiting behind another job does not steal its analysis time. API transports must
allow the queue wait: with two in-flight requests and default 240-second job budgets,
the matching backend defaults to 495 seconds rather than 255 seconds. Do not raise
model feature concurrency or remove the analysis gate to enable this overlap.

`telemetry.stageTimingsMs` adds `admission_wait` and `inference_wait`. JSON events
`media_analysis.pipeline.completed` and `.failed` expose the idempotency key,
elapsed time and queue/stage information through the Uvicorn logger. Cached
idempotent responses remain old receipts; they are not fresh inference measurements.
No new model identities, decoding recipes or response evidence semantics are added.

API rollout: deploy this worker first, then set
`MEDIA_ANALYSIS_DELIVERY_CONCURRENCY=2` on the API and restart it. With one delivery,
the new worker is compatible but there is no next source to prefetch. Preserve the
current dev memory limit and verify complete media requests, not health alone.

Tests: `pytest tests/e2e/test_concurrency_cancel.py tests/e2e/test_analyze_pipeline.py`.
The overlap/gate tests control I/O/model boundaries to prove scheduling and cleanup;
they do not benchmark real detector/depth models or establish production latency.
