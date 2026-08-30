#!/usr/bin/env bash
# Explicit model vendoring for local setup or CI smoke — not invoked at container startup.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON=python3
fi
exec "$PYTHON" -m media_analysis.tools.vendor_models "$@"
