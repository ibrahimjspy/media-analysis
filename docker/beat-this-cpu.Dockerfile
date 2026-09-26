# Build analysis-cpu from this checkout first, then layer the optional pilot.
ARG BASE_IMAGE=media-analysis:analysis-cpu
FROM ${BASE_IMAGE}
USER root
RUN pip install --no-cache-dir torch==2.7.0 torchaudio==2.7.0 \
      --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir beat-this==1.1.0 \
    && python -m media_analysis.tools.vendor_beat_this /models
# Activation is explicit at deployment; vendoring alone never enables inference.
USER mediaanalysis
