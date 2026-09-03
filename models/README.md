# Models

Weights are **vendored at setup or image build**, never on the first `/analyze` request.

## Files

| File | Purpose |
|---|---|
| `manifest.lock.json` | Pinned upstream URLs, revisions, SHA-256, and sizes (committed) |
| `ppocr-export.lock.json` | Official PP-OCR inference URL and owned-export toolchain pins |
| `manifest.json` | Runtime manifest written by the vendor tool (generated locally / in Docker) |
| `*.onnx` | Vendored weights (gitignored unless a release explicitly ships them) |

## Acquire weights

Real weights (~28 MB for the `analysis-cpu` profile):

```bash
./scripts/vendor-models.sh --profile analysis-cpu --dest ./models
```

Stub weights for tests (no download):

```bash
./scripts/vendor-models.sh --stub --dest ./models
```

See [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md) for Docker and lock-update procedures.

The process fails `/ready` if checksums in `manifest.json` do not match files on disk. The `analysis-cpu` image must not include matte weights (`modnet`).

Do not commit raw ONNX blobs unless a release explicitly vendors them. Do not add Ultralytics or unapproved GPL-3.0 weights (RVM) to this directory.
