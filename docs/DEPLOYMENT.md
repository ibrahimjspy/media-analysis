# Model acquisition and deployment

Production baseline is **linux/amd64**. Weights are vendored at **build or explicit setup time**, never on the first `/analyze` request. The worker fails to boot when checksums, model loading, or warmup fail.

**Checksum match is necessary but not sufficient for production.** Community OCR exports and runtime-compat tests do not substitute for owned export parity or fixture recall.

## Lock and manifest

| File | Role |
|---|---|
| `models/manifest.lock.json` | Pinned URLs, revisions, SHA-256, sizes, provenance, and production gates |
| `models/manifest.json` | Runtime manifest written by the vendor tool |

Lock entries include `provenance.artifactKind`:

| Kind | Meaning |
|---|---|
| `official-release` | Upstream published artifact (YOLOX) |
| `upstream-vendor` | Third-party vendor bundle at pinned revision (YuNet @ opencv_zoo commit) |
| `upstream-commit` | Upstream-owned artifact pinned to an immutable source commit (Silero VAD) |
| `community-export` | Interim community ONNX export — **not** owned/official (current PP-OCR) |
| `owned-export` | media-analysis recorded export after parity (target state for PP-OCR) |

Profile gate `gates.ownedPpOcrExportRecorded` must be `true` before treating OCR as production-ready.
Silero VAD v6.2 is pinned byte-for-byte to the official
`snakers4/silero-vad` repository at commit
`bfdc0193023f121ea5b3cc7b176dbed570a68a59`; runtime compatibility is checked
by loading and warming its stateful ONNX interface.

## PP-OCR provenance (interim)

The current lock points at a **GreatV/oar-ocr community export**, not an official PaddleOCR artifact and not a media-analysis owned export. See [ppocr-owned-export.md](ppocr-owned-export.md) and `models/ppocr-export.lock.json` for the reproducible owned export recipe. The export GitHub Action produces an ONNX artifact; it does not flip the production gate.

Docker sets `MEDIA_ANALYSIS_PP_OCR_EXPORT=community-interim` until an owned export is recorded.

## Vendor models (explicit command)

Real `analysis-cpu` weights (~28 MB):

```bash
./scripts/vendor-models.sh \
  --profile analysis-cpu \
  --lock models/manifest.lock.json \
  --dest ./models
```

Stub weights (no download):

```bash
./scripts/vendor-models.sh --stub --dest ./models
```

Opt-in checksum verification:

```bash
pytest tests/production/test_prod_vendor_real_models.py -m real_models
```

Opt-in **runtime compatibility** (I/O names, rank, dtype, dummy inference — **not recall/parity**):

```bash
pytest tests/production/test_prod_real_model_runtime_compat.py -m real_models
```

Opt-in real-model HTTP pipeline (boot, warmup, decode, all v1 runners, and response):

```bash
pytest tests/production/test_prod_real_http_pipeline.py -m real_models -o addopts="-q --strict-markers"
```

Linux CI runs the full `real_models` marker, including generated OCR, speech,
VFR, and rotation goldens. That job proves load and pipeline completion. It
does not flip the owned PP-OCR parity gate.

## Docker

```bash
docker build --platform linux/amd64 -f docker/analysis-cpu.Dockerfile -t media-analysis:analysis-cpu .
```

Production image properties:

- Non-editable `pip install .` (not editable)
- Runs as `mediaanalysis` (uid 10001)
- Writable temp: `/var/tmp/media-analysis` via `TMPDIR` / `MEDIA_ANALYSIS_TMPDIR`
- Models at `/models` (read-only to runtime user, world-readable)
- One uvicorn worker
- `/ready` distinguishes mechanical serving readiness from production parity

The checked-in `matte-cpu.Dockerfile` is explicitly a Stage 1 reference image. It enables
`MEDIA_ANALYSIS_MATTE_REFERENCE_MODE=1`, uses a stub MODNet artifact, and must report
`productionInferenceReady: false`. Do not deploy it as a production matte worker. A
production profile requires an owned immutable MODNet export plus benchmark and decoded-luma
cross-consumer parity gates.

## OpenCV / PySceneDetect

Pin `scenedetect>=0.6.7.1,<0.7` to avoid `opencv-python` conflicting with headless OpenCV. `docker/constraints.txt` pins headless OpenCV.

Full-tree `ruff check src tests` runs in CI with no feature-file exemptions.

## Updating the lock

1. Download or export artifact; compute SHA-256.
2. Update `models/manifest.lock.json` (URL, revision, provenance, gates).
3. `./scripts/vendor-models.sh --profile analysis-cpu --dest ./models --force`
4. `pytest tests/production -m real_models -o addopts="-q --strict-markers"`

For `upstream-commit` artifacts, use the immutable raw commit URL and verify both SHA-256
and byte size before opening the corresponding production gate.

## Licenses

Third-party notices: [NOTICE](../NOTICE).
