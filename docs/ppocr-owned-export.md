# Owned PP-OCRv5 mobile detection export

The vendored detector is still the **community export** from
[GreatV/oar-ocr v0.3.0](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_mobile_det.onnx)
until this recipe records an owned artifact. Checksum and runtime-compat tests
are not Paddle parity.

`productionInferenceReady` stays false while
`gates.ownedPpOcrExportRecorded` is false.

## Pins

Toolchain and official inference URL live in
[`models/ppocr-export.lock.json`](../models/ppocr-export.lock.json):

| Field | Pin |
|---|---|
| Official inference tar | `paddle3.0.0/PP-OCRv5_mobile_det_infer` SHA `50446e5d…ffc58` |
| Platform | linux/amd64 |
| Python | 3.12 |
| PaddlePaddle | 3.1.1 |
| paddle2onnx | 2.1.0 |
| ONNX opset | 11 |
| Probability-map max abs | `1e-3` |
| Synthetic recall | `0.80` at IoU `0.50` |

## Export

On linux/amd64:

```bash
pip install paddlepaddle==3.1.1 paddle2onnx==2.1.0 onnx==1.17.0 packaging
python -m media_analysis.tools.export_ppocr \
  --lock models/ppocr-export.lock.json \
  --out ./models/PP-OCRv5_mobile_det.onnx
```

Or, on a native linux/amd64 host:

```bash
./scripts/export-ppocr-det.sh --docker
```

Do not treat `docker --platform linux/amd64` on Apple Silicon as the export
environment. Paddle's CPU wheel needs real x86_64 and hangs under qemu.

GitHub Actions workflow `.github/workflows/export-ppocr.yml` runs the same
command on `ubuntu-latest` and uploads the ONNX plus `*.recipe.json`.

The exporter downloads the official PaddleX inference tar, verifies SHA-256
when the lock records one, converts `inference.json` or `inference.pdmodel`,
and writes the recipe sidecar. It does not run in the worker image.

## After a successful export

1. Host the ONNX at an immutable HTTPS URL (GitHub release asset).
2. Record `sha256`, `sizeBytes`, URL, and recipe fields in
   `models/manifest.lock.json`.
3. Set `provenance.artifactKind` to `owned-export` and
   `gates.ownedPpOcrExportRecorded` to `true`.
4. Commit `tests/fixtures/ppocr_parity/reference.npz` from the same export
   (Paddle probability map + matching input tensor).
5. Re-vendor and run:

```bash
pytest tests/production -m real_models -o addopts="-q --strict-markers"
```

That suite includes Paddle/ONNX probability-map parity (when the reference
exists) and synthetic Latin-banner recall.

Until those artifacts are recorded, do not describe OCR weights as
production-validated.
