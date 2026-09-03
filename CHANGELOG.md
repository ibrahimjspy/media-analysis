# Changelog

All notable changes are documented here. This project follows
[Semantic Versioning](https://semver.org/) and uses
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Canonical CFR MP4 delivery through `outputGrants.canonicalMp4`, including
  response `sha256` and `byteCount` for orchestrator verification.
- Per-stage timings, download/upload byte counts, decoded-pixel limits, and
  per-stage timeouts on `/analyze`.
- FFmpeg configure/version and runtime library provenance on every result.
- Generated people, face, OCR, speech, VFR, and rotation golden fixtures, plus
  a Linux CI job that runs real-model tests.
- Reproducible PP-OCRv5 owned-export toolchain, Paddle/ONNX parity helpers,
  and synthetic OCR recall floors. The production OCR gate stays closed until
  an owned artifact is hosted and the reference tensor is committed.

### Fixed

- Filter weak YuNet false positives against requested subject evidence and
  consolidate face fragments associated with the same subject track.
- Parse the final FFmpeg `ebur128` summary instead of its silent startup values.
- Prepend Silero VAD's required rolling 64-sample context to each inference
  window.

### Planned

- Owned PP-OCR export and parity validation.
- Owned MODNet export, quality benchmarks, and decoded-luma parity validation.
- v2 Stage 2 subject-targeted and segmented person mattes.

## [1.2.0] - 2026-08-31

### Added

- v1 subjects, faces, OCR-region detection, and shot detection.
- v1.1 motion, quality, audio, and Silero VAD measurements.
- v1.2 exposure, waveform, and thumbnail measurements.
- v2 Stage 1 full-duration `all_people` person-matte reference worker.
- Signed source downloads, scoped artifact uploads, cancellation, idempotency,
  model manifests, checksums, and explicit production-readiness gates.
- Unit, end-to-end, production-contract, real-model, coverage, and Docker tests.

### Security

- Exact and wildcard signed-URL host allowlists.
- Redirect validation, constant-time shared-key comparison, bounded downloads,
  and URL-safe errors.

[Unreleased]: https://github.com/ibrahimjspy/media-analysis/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/ibrahimjspy/media-analysis/releases/tag/v1.2.0
