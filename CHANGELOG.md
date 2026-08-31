# Changelog

All notable changes are documented here. This project follows
[Semantic Versioning](https://semver.org/) and uses
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Canonical CFR MP4 delivery through `outputGrants.canonicalMp4`, including
  response `sha256` and `byteCount` for orchestrator verification.

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
