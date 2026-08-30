# Testing

If a behavior is not tested, it is not done. The HTTP contract, frame math, and
error codes are specified by this suite — not by comments in the service.

There are three layers. Use all three.

## Unit tests (`tests/unit`)

Fast and network-free; a small marked subset exercises local ffmpeg/ffprobe. They lock
rules that are easy to get wrong later:

| File | What it proves |
|---|---|
| `test_frames.py` | Half-open ranges, `durationSec`, uniform analysis-scale boxes |
| `test_request_hash.py` | Idempotency ignores signed URLs and `expiresAt` |
| `test_source.py` | Expiry, exact/wildcard host allowlist, cancellation, no URL leakage |
| `test_manifest.py` | SHA-256 boot, CPU image must not ship matte weights |
| `test_jobs.py` | One key → one hash, bounded replay cache, cancel flag |
| `test_analyze_rules.py` | Feature/image rules, duplicate features, resolution validation |
| `test_errors.py` | Only the locked error codes exist |
| `test_subject_yolox.py`, `test_subject_pipeline.py`, `test_subject_bytetrack.py` | Official YOLOX preprocessing/decode and shot-local subject association |
| `test_yunet_face_detector.py`, `test_face_*.py` | YuNet boxes, face tracking, association, cleanup, cancellation |
| `test_ppocr_preprocessing.py`, `test_ocr_*.py` | DB postprocess/unclip, ROI sampling, per-shot merge, cancellation |
| `test_runtime.py` | Manifest verification, runtime construction, warmup failure |
| `test_motion.py`, `test_quality.py` | Static/estimated motion, masking, per-shot quality, duplicates |
| `test_audio_*.py`, `test_silero_vad.py`, `test_waveform_analyzer.py` | PCM, subprocess cancellation, VAD state, loudness, BPM, waveform |
| `test_exposure.py`, `test_thumbnails.py`, `test_measure_common.py` | Color science, bounded sampling, fill-only shots, upload grants |
| `person_matte/*` | MODNet preprocessing, guided refinement, temporal resets, streaming encode |

Run just these:

```bash
pytest tests/unit -m unit
```

## End-to-end tests (`tests/e2e`)

These start the real FastAPI app with `TestClient`, write stub model files,
serve a generated H.264 clip from `127.0.0.1`, and call `/analyze` the way a
caller would.

| File | What it proves |
|---|---|
| `test_health_ready_auth.py` | `/health`, `/ready`, 401, unknown feature, no HTML errors |
| `test_analyze_pipeline.py` | Full route, CFR, limits, timeout, SSRF, checksums, idempotency, output semantics |
| `test_concurrency_cancel.py` | One inference at a time and active-job cancellation |
| `test_v11_v12_integration.py` | Combined features, internal/prior shots, uploads, audio absence, matte reference mode |

Stub ONNX files are tiny byte strings with a matching `models/manifest.json`.
They exist so `/ready` and provenance can run without downloading weights.
They are **not** a claim that YOLOX/YuNet/PP-OCR/Silero/MODNet ran. Stub visual models
return completed empty outputs and stub Silero/MODNet sessions are only available when
explicit reference/test settings are enabled. Shots use real ffprobe + PySceneDetect on a
real MP4.

Tests that need ffmpeg are marked `@pytest.mark.ffmpeg`.

```bash
pytest tests/e2e -m e2e
pytest -m ffmpeg
```

## Commands

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest                  # unit + e2e + offline production contracts
pytest --cov            # coverage (fail under 70%)
pytest tests/unit -q    # quick loop while changing helpers
```

CI should run the full suite on Linux x86_64. Apple Silicon may run the same
commands as a smoke test; do not treat Mac CFR hashes as production parity.

## How to add a test

1. If the rule is pure (math, hashing, allowlist), add a unit test that would
   fail if someone “simplifies” the code.
2. If the rule is an HTTP status, JSON field, or ffmpeg behavior, add an e2e
   test. Hit the real route. Do not mock the app.
3. Name the test after the contract sentence, not the function name.
   Good: `test_empty_person_video_completes_subjects_as_empty_list`
   Bad: `test_analyze_1`
4. Never commit a developer `Downloads` fixture. Generate media with ffmpeg in
   `tmp_path`, or skip until a pinned object exists.
5. Assertions must use stable error `code` values and half-open frames.
6. If you add a feature, add both:
   - requested-and-empty (`[]` + `completed`)
   - not-requested (field omitted)
   - failed-independently (other features still complete)

## What this suite does not prove yet

These are explicit gaps. Do not pretend they are covered:

- Real YOLOX / YuNet / PP-OCR recall on people, faces, or burned-in text
- ByteTrack ID-switch rate or cut-reset on a multi-person fixture
- `Explain things simply.mp4` (needs a rights-cleared, SHA-pinned object)
- VFR iPhone / OBS / screen-record CFR identity across encoder builds
- Benchmark calibration for provisional motion, quality, BPM, exposure, and thumbnail policies
- Real speech recall/precision for Silero on a rights-cleared fixture corpus
- MODNet foreground quality and temporal ghosting on hair, hands, occlusion, and fast motion
- Person-matte cross-consumer/browser decoded-luma parity
- Load / p95 on a 60s 1080p reel

When those land, add fixtures and tests **before** calling the feature done.

## Production setup tests (`tests/production`)

These lock deployment contracts without touching feature algorithms:

| File | What it proves |
|---|---|
| `test_prod_manifest_lock_schema.py` | Lock pins URLs, SHA-256, no matte on `analysis-cpu` |
| `test_prod_vendor_stub_flow.py` | Vendor `--stub` writes manifest and verifies |
| `test_prod_dockerfile_contract.py` | amd64 slim image, ffmpeg, build-time vendoring, one worker |
| `test_prod_opencv_dependency_policy.py` | scenedetect `<0.7` avoids `opencv-python` conflict |
| `test_prod_vendor_real_models.py` | Checksum/size + PP-OCR community provenance + YuNet commit pin |
| `test_prod_real_model_runtime_compat.py` | Opt-in YOLOX/YuNet/OCR/Silero load + dummy inference compat |
| `test_prod_real_http_pipeline.py` | Opt-in real model boot + all-v1 `/analyze` smoke |
| `test_prod_matte_*` | Separate image, closed gates, lock profile, and reference-mode contracts |

```bash
pytest tests/production -m production
pytest tests/production -m real_models   # downloads ~28 MB, opt-in only
```

## Markers

```text
unit          isolated contract tests
e2e           HTTP + disk + (usually) ffmpeg
ffmpeg        requires ffmpeg and ffprobe on PATH
production    deployment, lock, and Docker contract tests
real_models   downloads real ONNX weights (opt-in)
```

`--strict-markers` is on. A typo in a marker fails the run.
