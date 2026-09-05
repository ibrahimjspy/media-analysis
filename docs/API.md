# API contract

media-analysis exposes a small HTTP API for trusted service-to-service use.
FastAPI serves the generated OpenAPI document at `/openapi.json` and interactive
documentation at `/docs` while the worker is running. Keep those endpoints on a
private network.

## Authentication

`GET /health`, `GET /ready`, and `GET /metrics` are unauthenticated for private
container probes and metrics scraping.
Every analysis and cancellation request requires:

```http
X-Media-Analysis-Key: <MEDIA_ANALYSIS_KEY>
```

## Analyze

`POST /analyze` accepts JSON:

```json
{
  "idempotencyKey": "job-018f",
  "canonicalize": false,
  "features": ["shots", "subjects", "faces", "ocr"],
  "source": {
    "signedGetUrl": "https://media.example/video.mp4?signature=...",
    "expiresAt": "2026-09-01T12:00:00Z",
    "sha256": "optional-64-character-hex-digest"
  },
  "analysisResolution": {
    "width": 960,
    "height": 540
  }
}
```

Supported `analysis-cpu` features are `subjects`, `faces`, `ocr`, `shots`,
`motion`, `quality`, `audio`, `exposure`, `waveform`, and `thumbnails`.
`person_matte` is accepted only by the separate `matte-cpu` worker.

Optional `outputGrants` contains scoped signed PUT grants for canonical media,
thumbnail indices, or matte labels. Optional `priorFacts` can provide canonical
shots and subject tracks for fill-only analysis. Consult `/openapi.json` for
the complete generated input schema.

When `canonicalize` is `true` and `outputGrants.canonicalMp4` is present, the
worker uploads the measured CFR MP4 before returning. The returned
`canonicalMedia` then includes the uploaded artifact's `sha256` and
`byteCount`. The orchestrator must verify those values before publishing a
derived asset ID. If no canonical grant is supplied, canonicalization remains
process-local for measurement compatibility.

The same `idempotencyKey` and logical request return the cached result within a
worker. Reusing the key with different logical content returns
`INVALID_REQUEST`. Signed URLs and expiration timestamps are excluded from the
logical request hash.

## Output conventions

- Frame ranges are half-open: `[startFrame, endFrameExclusive)`.
- Boxes are normalized to `0..1` against canonical frame dimensions.
- A missing feature field means it was not requested.
- `[]` or `null` means the feature ran and produced an empty or absent result.
- Tracks are local to one analysis and reset at shot boundaries.
- Subject tracks are not identities, and OCR returns regions rather than text.
- Generated artifacts include byte counts and SHA-256 digests.
- Provisional measurements include versioned policy identifiers.
- `telemetry` reports per-stage wall time, rolling stage p50/p95, downloaded and
  uploaded bytes, frame-cache activity, decoded frames, and decoded pixels. It
  is additive diagnostics and is not part of request identity.
- `provenance.ffmpegBuild` records the worker's `ffmpeg`/`ffprobe` version and
  configure line. `provenance.runtimeBuild` records Python, ONNX Runtime,
  OpenCV, and NumPy versions.

## Cancellation

`POST /analyze/{idempotencyKey}/cancel` requires authentication and returns:

```json
{
  "cancelled": true,
  "idempotencyKey": "job-018f"
}
```

Cancellation is cooperative across download, decode, inference, encoding, and
upload boundaries. Partial artifacts are not published as successful results.

## Health and readiness

`GET /health` reports process liveness:

```json
{"status": "ok", "version": "1.2.0"}
```

`GET /ready` reports model loading and gate state. `ready: true` means the
configured mode can serve requests; it does not mean all production parity
gates are open. Deploy production inference only when
`productionInferenceReady` is also `true`.

`GET /metrics` returns rolling `count`, `p50Ms`, and `p95Ms` per stage (including
download, decode, shots, faces, OCR, audio, matte, and upload stages). Use these
worker-local distributions as autoscaling inputs after aggregation by the
metrics collector.

## Errors

Errors are JSON and contain stable `code` and `error` fields:

```json
{
  "code": "SOURCE_EXPIRED",
  "error": "Source URL has expired"
}
```

Stable codes are `UNAUTHORIZED`, `INVALID_REQUEST`, `SOURCE_EXPIRED`,
`SOURCE_FETCH_FAILED`, `DECODE_FAILED`, `FEATURE_UNAVAILABLE`,
`MODEL_NOT_READY`, `TIMEOUT`, `CANCELLED`, `UPLOAD_FAILED`,
`CHECKSUM_MISMATCH`, `LIMIT_EXCEEDED`, and `INTERNAL_ERROR`.

Messages may become more specific; clients should branch on `code`.
