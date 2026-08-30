# syntax=docker/dockerfile:1
# Production source of truth: linux/amd64, Python 3.12 slim, vendored models at build time.
FROM python:3.12-slim-bookworm@sha256:0f5b26b9518d002b6173fd61daad821fa340635ebfec5bba471013f9ca114579

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version >/dev/null \
    && ffprobe -version >/dev/null

WORKDIR /app

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY models/manifest.lock.json ./models/manifest.lock.json
COPY docker/constraints.txt ./docker/constraints.txt

RUN python -m pip install --upgrade pip \
    && pip install --constraint docker/constraints.txt .

# Explicit build-time vendoring — never download weights on container start.
RUN python -m media_analysis.tools.vendor_models \
    --profile analysis-cpu \
    --lock models/manifest.lock.json \
    --dest /models

RUN groupadd --gid 10001 mediaanalysis \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin mediaanalysis \
    && mkdir -p /var/tmp/media-analysis \
    && chown -R mediaanalysis:mediaanalysis /app /models /var/tmp/media-analysis \
    && chmod -R a+rX /app /models

ENV MEDIA_ANALYSIS_MODEL_DIR=/models \
    MEDIA_ANALYSIS_IMAGE=analysis-cpu \
    MEDIA_ANALYSIS_TMPDIR=/var/tmp/media-analysis \
    TMPDIR=/var/tmp/media-analysis \
    MEDIA_ANALYSIS_PP_OCR_EXPORT=community-interim

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5001

USER mediaanalysis

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://127.0.0.1:5001/health', timeout=3); r.raise_for_status()"

ENTRYPOINT ["/entrypoint.sh"]
