# linux/amd64 export toolchain. Not a worker image.
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    paddlepaddle==3.1.1 \
    paddle2onnx==2.1.0 \
    onnx==1.17.0 \
    packaging

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY src /work/src
COPY models/ppocr-export.lock.json /work/models/ppocr-export.lock.json
ENV PYTHONPATH=/work/src
