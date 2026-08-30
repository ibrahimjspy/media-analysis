#!/bin/sh
set -eu

# One worker until matte job concurrency is measured and documented.
exec uvicorn media_analysis.app:app \
  --host 0.0.0.0 \
  --port 5001 \
  --workers 1
