# Build the base from this checkout first. A base with the other optional features
# may be supplied to preserve their vendored weights and runtime configuration.
ARG BASE_IMAGE=media-analysis:analysis-cpu
FROM ${BASE_IMAGE}
USER root
RUN pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir transformers==4.51.3 'safetensors>=0.4.3,<1' \
    && python -m media_analysis.tools.vendor_depth /models/depth-v1
# Vendoring does not activate the feature. Enable only via deployment config.
USER mediaanalysis
