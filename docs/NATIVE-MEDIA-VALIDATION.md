# Native media implementation verification

Verified locally on 2026-09-16, Python 3.12.3, macOS/Apple Silicon.
Baseline commit: `0363403`. No commit or deployment was made.

## Results

- Baseline: `.venv/bin/pytest tests/unit -m unit` — 378 passed.
- Focused native unit/HTTP tests — 72 added behavioral cases, included in the final suite.
- `.venv/bin/pytest tests/e2e -m e2e` passed during implementation; the final full suite also includes the later HTTP cases.
- Final: `.venv/bin/pytest --cov --cov-report=term:skip-covered -rs` — **548 passed, 1 skipped, 11 deselected**, in 47.22 seconds.
- Total statement/branch coverage: **81.68%**, above the required 70%.
- `.venv/bin/ruff check src tests` — passed.
- `git diff --check` — passed.

The default suite includes local FFmpeg integration tests. An unrestricted
`pytest -m ffmpeg` was not used because it also selects opt-in real-model download
and inference cases. No real-model execution or Linux production-parity claim is made.

## Behavioral evidence

- Recorded request/result branch fixtures and an exact pre-change legacy video digest;
  omitted and explicit video kinds preserve identity.
- All eight EXIF orientations checked against independently located image pixels;
  PNG/JPEG/WebP/SDR HEIF, alpha compositing, ICC handling, invalid profiles, animation,
  pixel bounds and unsupported high-bit-depth rejection.
- Native image quality/contrast/focus evidence, no fabricated center fallback, detector
  coordinates, completed empty detections, unavailable detectors and failed detectors.
- Independent click tracks at multiple tempos/phases, leading silence, missing beats,
  tempo changes and irregular/silent/absent evidence states; timestamp/count assertions.
- Real decoding for every advertised audio codec family, with timing checks after
  lossy encoding/container delay handling; stereo canonical WAV preservation.
- Rhythm-only video requests with/without audio and canonicalization disabled.
- Signed GET/PUT through local HTTP servers, checksums, MIME types, renewed grant
  delivery, expired grants, failed-upload retries without repeated measurements,
  worker-local job-state loss and renewed video canonical delivery.
- Current limits and source checksums rechecked on disk-cache hits, wrong decoded kinds,
  malformed metadata, non-finite PCM, timeout/cancellation and redelivery exclusion.
- Restricted runtime initialization loads only requested models; an audio DSP worker
  starts without visual models. Loader mocks are orchestration evidence, not real inference.
- Existing video, audio, VFR/rotation, cancellation, cache and reference-matte regressions
  remain covered by the full repository suite.

## Open gates

The skip is the existing owned PP-OCR export parity gate. Eleven `real_models` tests
are excluded by the repository's default marker. The test run also emits the existing
Starlette/httpx TestClient deprecation warning.

Real-image saliency/focus usefulness, real-music beat accuracy, speech/model accuracy,
matte hair/hands/occlusion/temporal quality, Linux x86_64 validation and caller
preview/export/parser compatibility remain unverified. Caller fixtures and a reviewed,
rights-cleared evaluation corpus were not provided. Local worker contract fixtures are
not a substitute for those checks. Existing OCR/matte production gates remain open.

See [the native contract and caller handoff](NATIVE-MEDIA.md) for exact supported
formats, heuristic evidence semantics, output schemas, configuration and delivery policy.
