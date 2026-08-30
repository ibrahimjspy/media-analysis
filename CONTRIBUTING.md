# Contributing to media-analysis

Thank you for helping improve media-analysis. Contributions of code, tests,
documentation, reproducible benchmarks, and rights-cleared fixtures are welcome.

## Before you start

- Search existing issues and pull requests.
- Open an issue before a large feature, model change, API change, or new dependency.
- Keep pull requests focused on one behavior.
- Never include secrets, signed URLs, private media, or data you cannot redistribute.
- Read [docs/TESTING.md](docs/TESTING.md) and [SECURITY.md](SECURITY.md).

## Development setup

Requirements are Python 3.12 plus `ffmpeg` and `ffprobe`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --constraint docker/constraints.txt -e ".[dev]"
ruff check src tests
pytest -m "not real_models" --cov
```

Tests create checksum-matched stub model files, so the default suite does not
download model weights. Real-model checks are opt-in:

```bash
pytest tests/production -m real_models
```

## Making a change

1. Fork the repository and create a branch from `main`.
2. Add or update a focused test before considering the behavior complete.
3. Preserve half-open frame ranges: `[startFrame, endFrameExclusive)`.
4. Keep feature failures isolated when the API contract allows partial results.
5. Run lint and the relevant unit, end-to-end, and production contract tests.
6. Update public documentation when behavior, configuration, or output changes.

Use Conventional Commit messages, for example:

- `feat: add shot-level camera motion metrics`
- `fix: reset face tracks at shot boundaries`
- `test: cover expired thumbnail upload grants`
- `docs: explain model provenance gates`

## Model and dependency policy

Model changes require an immutable revision, SHA-256, byte size, license,
provenance, runtime compatibility test, and an explicit production-readiness
decision. A checksum proves identity, not accuracy or parity.

Do not add Ultralytics code or weights, AGPL dependencies, or unapproved GPL
model code without a public issue that records the licensing decision. Do not
replace `opencv-python-headless` with GUI OpenCV.

## Pull requests

A reviewable pull request:

- explains the problem and the chosen approach;
- links its issue when one exists;
- includes tests for success, empty, and failure behavior where applicable;
- reports the exact validation commands run;
- calls out compatibility, performance, security, and licensing effects;
- avoids unrelated formatting or refactoring.

Maintainers may ask to split a large pull request. All contributions are
licensed under Apache-2.0 as described in the repository [LICENSE](LICENSE).

## Reporting problems

Use a public issue for reproducible bugs and proposals. Use the private process
in [SECURITY.md](SECURITY.md) for vulnerabilities or accidental secret exposure.
Questions and respectful disagreement are welcome under the
[Code of Conduct](CODE_OF_CONDUCT.md).
