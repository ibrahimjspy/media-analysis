"""Deterministic RGB-guided filter refinement (versioned TRD policy)."""

from __future__ import annotations

import cv2
import numpy as np

from media_analysis.person_matte.constants import (
    GUIDED_FILTER_EPS,
    GUIDED_FILTER_RADIUS,
    RGB_GUIDED_REFINEMENT_VERSION,
)


def _box_filter(channel: np.ndarray, radius: int) -> np.ndarray:
    ksize = 2 * radius + 1
    return cv2.boxFilter(channel, ddepth=-1, ksize=(ksize, ksize), normalize=True)


def refine_alpha_rgb_guided(alpha: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """RGB-guided covariance filter: guide from luminance, bounded radius/epsilon."""
    if alpha.shape[:2] != image_bgr.shape[:2]:
        raise ValueError("alpha and RGB must share spatial dimensions")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    guide = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    matte = alpha.astype(np.float32) / 255.0
    radius = GUIDED_FILTER_RADIUS
    eps = GUIDED_FILTER_EPS

    mean_i = _box_filter(guide, radius)
    mean_p = _box_filter(matte, radius)
    mean_ip = _box_filter(guide * matte, radius)
    mean_ii = _box_filter(guide * guide, radius)

    var_i = mean_ii - mean_i * mean_i
    cov_ip = mean_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)
    refined = mean_a * guide + mean_b
    return np.clip(refined * 255.0, 0.0, 255.0).astype(np.uint8)


def refinement_version() -> str:
    return RGB_GUIDED_REFINEMENT_VERSION
