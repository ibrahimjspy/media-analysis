from __future__ import annotations

import numpy as np
import pytest

from media_analysis.person_matte.refinement import refine_alpha_rgb_guided


@pytest.mark.unit
def test_refinement_preserves_flat_guidance() -> None:
    alpha = np.full((32, 32), 200, dtype=np.uint8)
    rgb = np.full((32, 32, 3), 128, dtype=np.uint8)
    out = refine_alpha_rgb_guided(alpha, rgb)
    assert out.shape == alpha.shape
    assert np.allclose(out, alpha, atol=1)


@pytest.mark.unit
def test_refinement_is_deterministic() -> None:
    alpha = np.full((32, 32), 128, dtype=np.uint8)
    alpha[:, 16:] = 220
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, 16:, :] = 255
    first = refine_alpha_rgb_guided(alpha, rgb)
    second = refine_alpha_rgb_guided(alpha, rgb)
    assert np.array_equal(first, second)


@pytest.mark.unit
def test_refinement_aligns_soft_edge_to_rgb_boundary() -> None:
    alpha = np.full((32, 32), 64, dtype=np.uint8)
    alpha[:, 16:] = 192
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, 16:, :] = 255
    out = refine_alpha_rgb_guided(alpha, rgb)
    left = int(out[16, 8])
    right = int(out[16, 24])
    assert right > left
    assert not np.array_equal(out, alpha)
