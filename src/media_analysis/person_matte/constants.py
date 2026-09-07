"""Versioned matte policy identifiers (store identity inputs)."""

from __future__ import annotations

MODNET_PREPROCESSING_VERSION = "modnet-onnx-ref512-norm127-aligned-1.1.0"
MATTE_TEMPORAL_POLICY_VERSION = "keyframe-flow256-backward-aligned-ema-shot-reset-2.2.0"
MATTE_FLOW_MAX_DIMENSION = 256
RGB_GUIDED_REFINEMENT_VERSION = "rgb-guided-filter-r8-eps1e-3-1.0.0"
MATTE_ENCODING_RECIPE = "matte-h264-yuv420p-fullrange-bt709-gop15-b0-1.0.1"

MODNET_REF_SIZE = 512
MODNET_MIN_ALIGNED_DIM = 32
MODNET_MAX_ALIGNED_DIM = 2048
MODNET_MAX_ASPECT_RATIO = 32.0

MODNET_INPUT_NAME = "input"
MODNET_OUTPUT_NAME = "output"

TEMPORAL_EMA_ALPHA = 0.65

GUIDED_FILTER_RADIUS = 8
GUIDED_FILTER_EPS = 1e-3

MAX_GOP_FRAMES = 15

# Decoded-luma semantics: ffprobe may report yuvj420p for full-range gray H.264.
DECODED_LUMA_TOLERANCE = 2
