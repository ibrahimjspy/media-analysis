import numpy as np
import pytest

from media_analysis.features.ppocr import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DetPreprocessMeta,
    clip_normalized_box,
    normalize_det_input,
    polygon_to_box,
    postprocess_db_map,
    preprocess_det_image,
    resize_for_det,
    unclip_quad,
)


@pytest.mark.unit
def test_resize_rounds_to_multiple_of_32_and_preserves_aspect() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    resized, meta = resize_for_det(image, limit_side_len=960)
    assert resized.shape[0] % 32 == 0
    assert resized.shape[1] % 32 == 0
    assert meta.src_height == 720
    assert meta.src_width == 1280
    assert meta.resized_height == resized.shape[0]
    assert meta.resized_width == resized.shape[1]


@pytest.mark.unit
def test_preprocess_output_is_nchw_imagenet_normalized() -> None:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    tensor, meta = preprocess_det_image(image)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == np.float32
    rgb = image.astype(np.float32) / 255.0
    expected = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_allclose(tensor[0, 0, 0, 0], expected[0, 0, 0], rtol=1e-5)
    assert meta.src_height == 64


@pytest.mark.unit
def test_normalize_rejects_non_bgr() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        normalize_det_input(np.zeros((8, 8), dtype=np.uint8))


@pytest.mark.unit
def test_postprocess_extracts_quad_and_normalizes_polygon() -> None:
    meta = DetPreprocessMeta(
        src_height=100,
        src_width=200,
        ratio=1.0,
        resized_height=100,
        resized_width=200,
    )
    prob = np.zeros((100, 200), dtype=np.float32)
    prob[20:40, 50:150] = 0.95
    polys = postprocess_db_map(prob, meta, thresh=0.5, box_thresh=0.5, min_size=2)
    assert len(polys) == 1
    poly = polys[0]
    assert len(poly.points) == 4
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in poly.points)
    assert poly.score >= 0.5


@pytest.mark.unit
def test_unclip_increases_area_and_clips_to_frame() -> None:
    pyclipper = pytest.importorskip("pyclipper")
    assert pyclipper is not None

    box = np.array(
        [[50.0, 20.0], [150.0, 20.0], [150.0, 40.0], [50.0, 40.0]],
        dtype=np.float32,
    )
    original_area = cv2_contour_area(box)
    expanded = unclip_quad(box, 1.5)
    assert len(expanded) == 4
    assert cv2_contour_area(expanded) > original_area

    meta = DetPreprocessMeta(
        src_height=100,
        src_width=200,
        ratio=1.0,
        resized_height=100,
        resized_width=200,
    )
    prob = np.zeros((100, 200), dtype=np.float32)
    prob[22:38, 52:148] = 0.95
    polys = postprocess_db_map(prob, meta, thresh=0.5, box_thresh=0.5, min_size=2)
    assert polys
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in polys[0].points)


def cv2_contour_area(points: np.ndarray) -> float:
    import cv2

    return float(abs(cv2.contourArea(points.reshape(-1, 1, 2).astype(np.float32))))


@pytest.mark.unit
def test_clip_normalized_box_clamps_to_unit_square() -> None:
    clipped = clip_normalized_box({"x": -0.1, "y": 0.2, "width": 1.5, "height": 0.9})
    assert clipped == {"x": 0.0, "y": 0.2, "width": 1.0, "height": 0.8}


@pytest.mark.unit
def test_polygon_to_box_is_axis_aligned_envelope() -> None:
    box = polygon_to_box(((0.1, 0.2), (0.5, 0.2), (0.5, 0.4), (0.1, 0.4)))
    assert box == pytest.approx({"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.2})


@pytest.mark.unit
def test_malformed_probability_map_shape_raises() -> None:
    meta = DetPreprocessMeta(10, 10, 1.0, 10, 10)
    with pytest.raises(ValueError, match="unexpected probability map shape"):
        postprocess_db_map(np.zeros((5,), dtype=np.float32), meta)
