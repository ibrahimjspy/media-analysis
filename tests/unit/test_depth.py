"""Numeric depth artifact checks independent of optional model dependencies."""

import base64
import hashlib
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from media_analysis.features.depth import (
    DepthAnalyzer,
    cached_payload,
    encode_depth,
    validate_depth_image,
    verify_files,
)


def test_depth_polarity_endianness_shape_and_exact_digest():
    raw = np.tile(np.linspace(0, 10, 20, dtype=np.float32), (10, 1))
    data, meta = encode_depth(raw, 1200, 600)
    values = np.frombuffer(data, dtype="<u2").reshape(256, 512)
    assert values[0, 0] == 0 and values[0, -1] == 65535
    assert np.all(np.diff(values[0].astype(np.int32)) >= 0)
    assert len(data) == 256 * 512 * 2 == meta["byteCount"]
    assert meta["sha256"] == hashlib.sha256(data).hexdigest()
    assert meta["canonicalWidth"] == 1200
    assert meta["productionQualified"] is False


@pytest.mark.parametrize(
    "raw", [np.ones((3, 3)), np.full((3, 3), np.nan), np.full((3, 3), np.inf), np.zeros(4)]
)
def test_invalid_or_flat_depth_is_never_published(raw):
    with pytest.raises(ValueError):
        encode_depth(raw, 100, 100)


def test_missing_or_tampered_model_files_fail_before_loading(tmp_path):
    with pytest.raises(ValueError, match="DEPTH_MODEL_FILE_INVALID"):
        verify_files(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"not weights")
    with pytest.raises(ValueError):
        verify_files(tmp_path)


@pytest.mark.parametrize("color", [(0, 0, 0), (255, 255, 255), (128, 128, 128), (20, 70, 120)])
def test_uniform_images_are_declined_before_model_execution(color):
    """A full-range model output cannot make a blank RGB input scene evidence."""
    processor, model = Mock(), Mock()
    analyzer = DepthAnalyzer(processor, model)
    with pytest.raises(ValueError, match="^DEPTH_NO_IMAGE_VARIATION$"):
        analyzer.analyze(Image.new("RGB", (640, 480), color), lambda: None)
    processor.assert_not_called()
    model.assert_not_called()


def test_uniform_image_guard_does_not_introduce_a_contrast_threshold():
    """Any real spatial variation passes this narrow guard, even one RGB level."""
    image = Image.new("RGB", (640, 480), (128, 128, 128))
    image.putpixel((200, 200), (129, 128, 128))
    validate_depth_image(image)


def cache_entry():
    """Prepare a valid private entry without upload-generated mime metadata."""
    data, metadata = encode_depth(np.arange(32, dtype=np.float32).reshape(4, 8), 64, 32)
    metadata["canonicalSha256"] = "a" * 64
    metadata["_encoded"] = base64.b64encode(data).decode("ascii")
    return data, metadata


def test_cache_accepts_private_metadata_before_and_after_valid_mime_annotation():
    data, metadata = cache_entry()
    assert cached_payload(metadata, "a" * 64, 64, 32) == data
    metadata["mimeType"] = "application/octet-stream"
    assert cached_payload(metadata, "a" * 64, 64, 32) == data


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("model", "revision"), "wrong-revision"),
        (("model", "weightsSha256"), "b" * 64),
        (("model", "extra"), "not-in-recipe"),
        (("canonicalHeight",), 64),
        (("coordinateSpace",), "original_image"),
        (("width",), True),
        (("byteCount",), True),
        (("nearValue",), 0),
        (("farValue",), True),
        (("normalization", "low"), float("nan")),
        (("normalization", "high"), float("inf")),
        (("normalization", "high"), 10**400),
        (("normalization", "policy"), "min-max"),
        (("normalization", "low"), True),
        (("diagnostics", "edgeFraction"), float("nan")),
        (("evidenceKind",), "verified-measurement"),
        (("productionQualified",), 0),
        (("mimeType",), None),
        (("unexpected",), "would-fail-backend-strict-validation"),
        (("_encoded",), None),
    ],
)
def test_cache_rejects_corrupt_metadata_even_when_binary_samples_are_valid(path, replacement):
    """Preserving bytes and their digest must not authorize changed semantics."""
    _, original = cache_entry()
    metadata = deepcopy(original)
    target = metadata
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(ValueError, match="^DEPTH_CACHE_INVALID$"):
        cached_payload(metadata, "a" * 64, 64, 32)


def test_cache_binds_grid_geometry_to_freshly_decoded_canonical_dimensions():
    _, metadata = cache_entry()
    with pytest.raises(ValueError, match="^DEPTH_CACHE_INVALID$"):
        cached_payload(metadata, "a" * 64, 32, 64)
