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


def upsample_alpha_image_guided(alpha: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """Recover source-resolution edges from local foreground/background RGB.

    Estimate local colors from confident low-resolution alpha regions, then
    project each original-resolution RGB pixel onto that local color mixture.
    Only refine uncertain alpha where both anchors and color agreement exist.
    Low-contrast/ambiguous regions retain model alpha; no global gain/threshold.
    """
    if alpha.ndim != 2 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected 2D alpha and HxWx3 source image")
    height, width = image_bgr.shape[:2]
    low_h, low_w = alpha.shape
    if min(height, width, low_h, low_w) < 1:
        raise ValueError("empty matte or source image")
    prior = alpha.astype(np.float32) / 255.0
    fg_weight = np.clip((prior - 0.8) / 0.2, 0, 1)
    bg_weight = np.clip((0.2 - prior) / 0.2, 0, 1)
    if not np.any(fg_weight) or not np.any(bg_weight):
        return np.rint(
            cv2.resize(prior, (width, height), interpolation=cv2.INTER_LINEAR) * 255
        ).astype(np.uint8)
    source = image_bgr.astype(np.float32) / 255.0
    low = cv2.resize(source, (low_w, low_h), interpolation=cv2.INTER_AREA)

    def anchors(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        support = _box_filter(weight, 16)
        color = _box_filter(low * weight[:, :, None], 16) / np.maximum(
            support[:, :, None],
            1e-6,
        )
        return (
            cv2.resize(color, (width, height), interpolation=cv2.INTER_LINEAR),
            cv2.resize(support, (width, height), interpolation=cv2.INTER_LINEAR),
        )

    foreground, fg_support = anchors(fg_weight)
    background, bg_support = anchors(bg_weight)
    upsampled = cv2.resize(prior, (width, height), interpolation=cv2.INTER_LINEAR)
    # Opaque/background regions cannot change. Avoid full-frame color algebra.
    rows, columns = np.nonzero(
        (upsampled > 0.02) & (upsampled < 0.98) & (fg_support > 0.01) & (bg_support > 0.01)
    )
    foreground = foreground[rows, columns]
    background = background[rows, columns]
    pixels = source[rows, columns]
    delta = foreground - background
    contrast = np.sum(delta * delta, axis=1)
    projected = np.clip(
        np.sum((pixels - background) * delta, axis=1) / np.maximum(contrast, 1e-6),
        0,
        1,
    )
    residual = np.sum(
        (pixels - background - projected[:, None] * delta) ** 2,
        axis=1,
    )
    reliable = (contrast > 0.0004) & (residual < np.maximum(0.001, contrast * 0.1))
    strength = np.clip((contrast - 0.0004) / 0.0021, 0, 1) * reliable
    prior_band = upsampled[rows, columns]
    upsampled[rows, columns] = prior_band + strength * (projected - prior_band)
    return np.rint(np.clip(upsampled, 0, 1) * 255).astype(np.uint8)
