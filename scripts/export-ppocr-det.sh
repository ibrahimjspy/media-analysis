#!/usr/bin/env bash
# Reproducible owned PP-OCRv5 mobile det export — explicit operator command only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PADDLEOCR_REF:-}" ]]; then
  echo "Set PADDLEOCR_REF to a pinned PaddleOCR git tag or SHA." >&2
  echo "See docs/ppocr-owned-export.md" >&2
  exit 1
fi

if [[ -z "${PADDLE_DET_INFER_DIR:-}" ]]; then
  echo "Set PADDLE_DET_INFER_DIR to the mobile det inference directory." >&2
  echo "See docs/ppocr-owned-export.md" >&2
  exit 1
fi

OUT="${1:-./models/PP-OCRv5_mobile_det.onnx}"
mkdir -p "$(dirname "$OUT")"

if ! command -v paddle2onnx >/dev/null 2>&1; then
  echo "Install pinned paddlepaddle and paddle2onnx before running this script." >&2
  exit 1
fi

paddle2onnx \
  --model_dir "$PADDLE_DET_INFER_DIR" \
  --model_filename "${PADDLE_DET_MODEL_FILE:-inference.pdmodel}" \
  --params_filename "${PADDLE_DET_PARAMS_FILE:-inference.pdiparams}" \
  --save_file "$OUT" \
  --opset_version "${PADDLE2ONNX_OPSET:-11}"

echo "Wrote $OUT"
echo "Next: shasum -a 256 $OUT"
echo "Then update models/manifest.lock.json per docs/ppocr-owned-export.md"
