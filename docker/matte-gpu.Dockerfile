# syntax=docker/dockerfile:1
# GPU production target: ORT 1.22 + TensorRT 10.9 + CUDA 12.8 on Python 3.12.
# The checked-in MODNet lock remains gated; replace it with the approved owned
# export before this image can become productionInferenceReady.
FROM nvcr.io/nvidia/tensorrt:25.03-py3

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
COPY docker/matte-constraints.txt ./docker/matte-constraints.txt

# Only one ORT distribution may be installed. ORT 1.22 matches TensorRT 10.9.
RUN python -m pip install --upgrade pip \
    && pip install --constraint docker/matte-constraints.txt . \
    && pip uninstall -y onnxruntime \
    && pip install onnxruntime-gpu==1.22.0

RUN python -m media_analysis.tools.vendor_models \
    --profile matte-cpu \
    --lock models/manifest.lock.json \
    --dest /models

RUN groupadd --gid 10001 mediaanalysis \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin mediaanalysis \
    && mkdir -p /var/tmp/media-analysis/tensorrt \
    && chown -R mediaanalysis:mediaanalysis /app /models /var/tmp/media-analysis \
    && chmod -R a+rX /app /models

ENV MEDIA_ANALYSIS_MODEL_DIR=/models \
    MEDIA_ANALYSIS_IMAGE=matte-cpu \
    MEDIA_ANALYSIS_MATTE_EXECUTION_PROVIDER=tensorrt \
    MEDIA_ANALYSIS_TENSORRT_CACHE_DIR=/var/tmp/media-analysis/tensorrt \
    MEDIA_ANALYSIS_TMPDIR=/var/tmp/media-analysis \
    TMPDIR=/var/tmp/media-analysis

COPY docker/matte-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5001

USER mediaanalysis

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://127.0.0.1:5001/health', timeout=3); r.raise_for_status()"

ENTRYPOINT ["/entrypoint.sh"]
