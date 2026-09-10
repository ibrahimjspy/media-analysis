from __future__ import annotations

import subprocess

import numpy as np
import pytest

from media_analysis.frames import Rational
from media_analysis.person_matte.encoding import encode_matte_mp4
from media_analysis.person_matte.refinement import (
    refine_alpha_rgb_guided,
    upsample_alpha_image_guided,
)


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


@pytest.mark.parametrize("background,foreground", [(0, 255), (100, 110)])
def test_source_guidance_recovers_focused_edge_instead_of_just_resizing(background, foreground):
    source = np.full((64, 128, 3), background, np.uint8)
    source[:, 64:] = foreground
    ramp = np.clip((np.arange(64) - 24) / 16, 0, 1)
    alpha = np.tile(np.rint(ramp * 255).astype(np.uint8), (32, 1))
    output = upsample_alpha_image_guided(alpha, source)
    assert output.shape == (64, 128)
    assert np.count_nonzero((output[32] > 25) & (output[32] < 230)) <= 4
    assert output[32, 62] < 25 and output[32, 65] > 230


def test_source_guidance_preserves_real_motion_blur():
    native = np.clip((np.arange(128) - 59) / 10, 0, 1)
    source = np.repeat(
        np.tile(np.rint(native * 255).astype(np.uint8), (64, 1))[:, :, None], 3, axis=2
    )
    alpha = np.tile(
        np.rint(np.clip((np.arange(64) - 24) / 16, 0, 1) * 255).astype(np.uint8), (32, 1)
    )
    output = upsample_alpha_image_guided(alpha, source)
    expected_band = np.count_nonzero((native > 0.1) & (native < 0.9))
    actual_band = np.count_nonzero((output[32] > 25) & (output[32] < 230))
    assert abs(actual_band - expected_band) <= 2
    assert actual_band > 4


@pytest.mark.parametrize("value", [0, 128, 255])
def test_ambiguous_guidance_does_not_invent_opacity_or_normalize(value):
    alpha = np.full((16, 16), value, np.uint8)
    source = np.full((32, 32, 3), 128, np.uint8)
    output = upsample_alpha_image_guided(alpha, source)
    np.testing.assert_array_equal(output, np.full((32, 32), value, np.uint8))


@pytest.mark.ffmpeg
def test_source_edge_survives_the_unchanged_h264_encoding(tmp_path):
    source = np.zeros((64, 128, 3), np.uint8)
    source[:, 63:] = 255
    prior = np.tile(
        np.rint(np.clip((np.arange(64) - 24) / 16, 0, 1) * 255).astype(np.uint8), (32, 1)
    )
    alpha = upsample_alpha_image_guided(prior, source)
    output = tmp_path / "edge.mp4"
    encode_matte_mp4([alpha.tobytes()], width=128, height=64, fps=Rational(30, 1), dest=output)
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(output),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
    row = np.frombuffer(decoded, np.uint8).reshape(64, 128)[32]
    assert np.count_nonzero((row > 25) & (row < 230)) <= 4
    assert row.max() == 255 and row.min() == 0
