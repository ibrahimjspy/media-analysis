# Owned PP-OCRv5 mobile detection export (not yet recorded in lock)

The vendored **community export** from [GreatV/oar-ocr v0.3.0](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_mobile_det.onnx) is an **interim** artifact. It is **not** the media-analysis owned/official export. Checksum match and runtime-compat tests prove **DB probability-map wiring only** — not Paddle parity, not production validation.

Production readiness for OCR requires `ownedPpOcrExportRecorded: true` in `models/manifest.lock.json` after an owned export is pinned with its own URL, SHA-256, and parity notes.

## Upstream official model

| Field | Value |
|---|---|
| Project | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| Model | PP-OCRv5 mobile **det** (DB detector head) |
| License | Apache-2.0 |

## Owned export recipe (reproducible)

These steps produce an ONNX file suitable for `media_analysis.features.ppocr` (ImageNet-normalized NCHW input, single-channel DB probability map output). Run on **linux/amd64** to match production.

### 1. Pin PaddleOCR revision

```bash
export PADDLEOCR_REF="<git-sha-or-tag>"   # record in lock provenance.upstreamOfficialRevision
git clone --depth 1 --branch "${PADDLEOCR_REF}" https://github.com/PaddlePaddle/PaddleOCR.git
```

### 2. Export inference model

Follow PaddleOCR docs to obtain the mobile det inference weights (`.pdmodel` / `.pdiparams` or `.json` depending on release). Record the exact checkpoint path and revision in the lock entry.

### 3. Convert to ONNX (paddle2onnx)

```bash
pip install paddlepaddle==<pinned> paddle2onnx==<pinned>
paddle2onnx \
  --model_dir ./inference/PP-OCRv5_mobile_det \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file PP-OCRv5_mobile_det.onnx \
  --opset_version 11
```

Adjust filenames to match the pinned PaddleOCR export layout.

### 4. Verify runtime compatibility (not recall)

```bash
pytest tests/production/test_prod_real_model_runtime_compat.py -m real_models -k ppocr
```

### 5. Record parity and update lock

1. `shasum -a 256 PP-OCRv5_mobile_det.onnx`
2. Upload/store the artifact at a pinned URL (release asset or internal mirror).
3. Update `models/manifest.lock.json`:
   - Set `provenance.artifactKind` to `owned-export`
   - Set `provenance.ownedExportRecorded` to `true`
   - Set profile `gates.ownedPpOcrExportRecorded` to `true`
   - Add parity notes (export date, paddle2onnx version, diff summary vs community export)
4. Re-run `./scripts/vendor-models.sh --profile analysis-cpu --dest ./models --force`

Until step 5 completes, **do not describe OCR weights as production-validated**.

## Helper script

`scripts/export-ppocr-det.sh` documents the same flow and exits unless `PADDLEOCR_REF` and inference paths are set — it does not run automatically at build or startup.
