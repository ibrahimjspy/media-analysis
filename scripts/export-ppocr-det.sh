#!/usr/bin/env bash
# Owned PP-OCRv5 mobile det export. Run on linux/amd64 with the pinned toolchain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--docker" ]]; then
  docker build --platform linux/amd64 -f docker/ppocr-export.Dockerfile -t media-analysis:ppocr-export .
  docker run --rm --platform linux/amd64 \
    -v "$ROOT:/work" \
    -w /work \
    media-analysis:ppocr-export \
    python -m media_analysis.tools.export_ppocr \
      --lock models/ppocr-export.lock.json \
      --out "${2:-./models/PP-OCRv5_mobile_det.onnx}"
  exit 0
fi

python -m media_analysis.tools.export_ppocr \
  --lock models/ppocr-export.lock.json \
  --out "${1:-./models/PP-OCRv5_mobile_det.onnx}"
