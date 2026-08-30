from __future__ import annotations

import numpy as np
import pytest
from tests.unit.person_matte.fake_ort import FakeModnetSession

from media_analysis.errors import INVALID_REQUEST, AnalyzeError
from media_analysis.person_matte.modnet import (
    compute_modnet_input_size,
    infer_modnet_alpha,
    postprocess_modnet,
    preprocess_modnet,
)


@pytest.mark.unit
def test_modnet_preprocess_normalizes_to_minus1_plus1() -> None:
    frame = np.full((64, 48, 3), 128, dtype=np.uint8)
    tensor, h, w = preprocess_modnet(frame)
    assert h == 64 and w == 48
    assert tensor.shape[0] == 1 and tensor.shape[1] == 3
    assert abs(float(tensor.mean())) < 0.01


@pytest.mark.unit
def test_modnet_postprocess_resizes_in_float_space_before_quantize() -> None:
    output = np.full((1, 1, 32, 32), 0.5, dtype=np.float32)
    alpha = postprocess_modnet(output, orig_height=40, orig_width=60)
    assert alpha.shape == (40, 60)
    assert abs(int(alpha[0, 0]) - 128) <= 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fill", "expected"),
    [(0, 0), (128, 128), (255, 255)],
)
def test_fake_modnet_session_maps_synthetic_patches(fill: int, expected: int) -> None:
    normalized = (fill - 127.5) / 127.5
    tensor = np.full((1, 3, 32, 32), normalized, dtype=np.float32)
    session = FakeModnetSession()
    alpha = postprocess_modnet(
        session.run(["output"], {"input": tensor})[0],
        orig_height=32,
        orig_width=32,
    )
    assert int(alpha[0, 0]) == expected


@pytest.mark.unit
def test_modnet_rejects_extreme_aspect_ratio() -> None:
    with pytest.raises(AnalyzeError) as exc:
        compute_modnet_input_size(2000, 70000)
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_modnet_small_dimension_is_aligned_safely() -> None:
    rw, rh = compute_modnet_input_size(8, 64)
    assert rw >= 32 and rh >= 32
    assert rw <= 2048 and rh <= 2048


@pytest.mark.unit
def test_infer_modnet_alpha_end_to_end() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    alpha = infer_modnet_alpha(FakeModnetSession(), frame)
    assert alpha.shape == (32, 32)
    assert alpha.max() == 0
