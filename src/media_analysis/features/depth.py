"""Pinned, optional still-image depth evidence; no animation or metric claims.

Weights load and warm only at startup. Inference is serialized and bounded to a
518px tensor; cancellation/deadlines are checked around native kernels. There
are no request-time downloads. Relative maps describe visible surfaces only.
"""

from __future__ import annotations

import base64
import hashlib
import math
import threading
from pathlib import Path

import cv2
import numpy as np

POLICY = "home-tour-depth-1"
MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
WEIGHTS_SHA = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
FILES = {
    "model.safetensors": (99173660, WEIGHTS_SHA),
    "config.json": (950, "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75"),
    "preprocessor_config.json": (
        775,
        "d41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84",
    ),
}
MAX_DIMENSION = 512
RECIPE = f"{POLICY}:{REVISION}:{WEIGHTS_SHA}:bounded518:map512:p1-p99:uint16le"
ANALYSIS_WARNING_CODES = frozenset(
    {
        "DEPTH_NO_IMAGE_VARIATION",
        "DEPTH_ASPECT_UNSUPPORTED",
        "DEPTH_INVALID_VALUES",
        "DEPTH_DEGENERATE",
    }
)


def verify_files(folder: Path) -> None:
    """Reject missing/changed weights AND preprocessing files before model loading."""
    for name, (size, digest) in FILES.items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError("DEPTH_MODEL_FILE_INVALID")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("DEPTH_MODEL_CHECKSUM_MISMATCH")


def encode_depth(raw: np.ndarray, width: int, height: int) -> tuple[bytes, dict]:
    """Produce bounded little-endian samples aligned to the full canonical image.

    Percentiles normalize affine-invariant inverse depth: 0 far, 65535 near.
    Constant/nonfinite maps are rejected. Gradient fraction is diagnostic, not
    confidence or a guarantee that a camera trajectory avoids disocclusion.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all() or min(width, height) < 1:
        raise ValueError("DEPTH_INVALID_VALUES")
    scale = min(1.0, MAX_DIMENSION / max(width, height))
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    grid = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
    low, high = (float(x) for x in np.percentile(grid, [1, 99]))
    if high - low <= max(1e-6, abs(high) * 1e-6):
        raise ValueError("DEPTH_DEGENERATE")
    normalized = np.clip((grid - low) / (high - low), 0, 1)
    samples = np.rint(normalized * 65535).astype("<u2")
    data = samples.tobytes(order="C")
    edge_count = np.count_nonzero(np.abs(np.diff(normalized, axis=0)) > 0.15)
    edge_count += np.count_nonzero(np.abs(np.diff(normalized, axis=1)) > 0.15)
    return data, {
        "policyVersion": POLICY,
        "coordinateSpace": "normalized_canonical_image",
        "encoding": "uint16le_inverse_relative",
        "width": w,
        "height": h,
        "canonicalWidth": width,
        "canonicalHeight": height,
        "nearValue": 65535,
        "farValue": 0,
        "normalization": {"policy": "percentile-1-99", "low": low, "high": high},
        "model": {"id": MODEL, "revision": REVISION, "weightsSha256": WEIGHTS_SHA},
        "sha256": hashlib.sha256(data).hexdigest(),
        "byteCount": len(data),
        "diagnostics": {"edgeFraction": float(edge_count / max(1, (w - 1) * h + (h - 1) * w))},
        "evidenceKind": "model-estimate",
        "productionQualified": False,
    }


def validate_depth_image(image) -> None:
    """Decline unsupported geometry or absent spatial evidence before reuse.

    Canonical inputs are opaque RGB. A solid color has no scene evidence even
    when a learned predictor returns nonconstant depth. This exact check also
    protects reads of maps cached before the input guard was introduced.
    """
    width, height = image.size
    if max(width, height) / min(width, height) > 8:
        raise ValueError("DEPTH_ASPECT_UNSUPPORTED")
    if all(low == high for low, high in image.getextrema()):
        raise ValueError("DEPTH_NO_IMAGE_VARIATION")


class DepthAnalyzer:
    """One warm CPU model behind a cancellable, process-local inference lock."""

    def __init__(self, processor, model):
        """Retain warm collaborators and a lock; construction performs no inference."""
        self.processor = processor
        self.model = model
        self._lock = threading.Lock()

    @classmethod
    def load(cls, folder: Path):
        """Verify local safetensors and warm the exact preprocessing/model pair."""
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        verify_files(folder)
        processor = AutoImageProcessor.from_pretrained(
            folder, local_files_only=True, use_fast=False
        )
        model = (
            AutoModelForDepthEstimation.from_pretrained(
                folder,
                local_files_only=True,
                use_safetensors=True,
            )
            .eval()
            .to("cpu")
        )
        with torch.inference_mode():
            model(
                **processor(
                    images=Image.new("RGB", (56, 56)),
                    size={"width": 56, "height": 56},
                    keep_aspect_ratio=False,
                    return_tensors="pt",
                )
            )
        return cls(processor, model)

    def analyze(self, image, cancel_check) -> tuple[bytes, dict]:
        """Infer once on canonical sRGB pixels; never interpret results as metres.

        Exactly uniform images contain no spatial evidence. Decline them before
        model execution rather than normalizing invented depth across a blank
        frame. This exact check is not a general image-suitability threshold.

        Native kernels cannot be interrupted; cancellation after the kernel
        prevents publication, and callers retain their shared job timeout policy.
        """
        validate_depth_image(image)
        width, height = image.size

        import torch

        while not self._lock.acquire(timeout=0.1):
            cancel_check()
        try:
            cancel_check()
            scale = 518 / max(width, height)
            size = {
                "width": max(14, round(width * scale / 14) * 14),
                "height": max(14, round(height * scale / 14) * 14),
            }
            inputs = self.processor(
                images=image, size=size, keep_aspect_ratio=False, return_tensors="pt"
            )
            with torch.inference_mode():
                raw = self.model(**inputs).predicted_depth[0].cpu().numpy()
            cancel_check()
            return encode_depth(raw, width, height)
        finally:
            self._lock.release()


def cached_payload(
    metadata: dict, canonical_sha: str, canonical_width: int, canonical_height: int
) -> bytes:
    """Validate the complete pinned recipe, image identity and cached samples.

    Cache corruption must cause recomputation, not repeated publication failures.
    The model itself is not rerun merely because a signed output grant changes.
    Canonical dimensions come from the freshly decoded image, not cached claims.
    The private entry precedes upload, so mimeType is optional here; when present,
    it must agree with the numeric artifact type. No field establishes suitability.
    """
    if not isinstance(metadata, dict) or min(canonical_width, canonical_height) < 1:
        raise ValueError("DEPTH_CACHE_INVALID")

    # Recipe-bound grid geometry and provenance must agree before any byte reuse.
    scale = min(1.0, MAX_DIMENSION / max(canonical_width, canonical_height))
    w = max(1, round(canonical_width * scale))
    h = max(1, round(canonical_height * scale))
    expected = {
        "policyVersion": POLICY,
        "coordinateSpace": "normalized_canonical_image",
        "encoding": "uint16le_inverse_relative",
        "width": w,
        "height": h,
        "canonicalWidth": canonical_width,
        "canonicalHeight": canonical_height,
        "canonicalSha256": canonical_sha,
        "nearValue": 65535,
        "farValue": 0,
        "model": {"id": MODEL, "revision": REVISION, "weightsSha256": WEIGHTS_SHA},
        "byteCount": w * h * 2,
        "evidenceKind": "model-estimate",
        "productionQualified": False,
    }
    required = {*expected, "normalization", "diagnostics", "sha256", "_encoded"}
    if (
        not required <= metadata.keys()
        or metadata.keys() - required - {"mimeType"}
        or any(metadata[key] != value for key, value in expected.items())
        or metadata["productionQualified"] is not False
        or ("mimeType" in metadata and metadata["mimeType"] != "application/octet-stream")
    ):
        raise ValueError("DEPTH_CACHE_INVALID")
    for key in (
        "width",
        "height",
        "canonicalWidth",
        "canonicalHeight",
        "byteCount",
        "nearValue",
        "farValue",
    ):
        if type(metadata[key]) is not int:
            raise ValueError("DEPTH_CACHE_INVALID")

    # Range and diagnostics are validated as data, never interpreted as confidence.
    normalization = metadata["normalization"]
    diagnostics = metadata["diagnostics"]
    if (
        not isinstance(normalization, dict)
        or set(normalization) != {"policy", "low", "high"}
        or normalization["policy"] != "percentile-1-99"
        or not isinstance(diagnostics, dict)
        or set(diagnostics) != {"edgeFraction"}
    ):
        raise ValueError("DEPTH_CACHE_INVALID")
    low, high = normalization["low"], normalization["high"]
    edge_fraction = diagnostics["edgeFraction"]
    if any(type(value) not in {int, float} for value in (low, high, edge_fraction)):
        raise ValueError("DEPTH_CACHE_INVALID")
    try:
        low, high, edge_fraction = (float(value) for value in (low, high, edge_fraction))
    except OverflowError as error:
        raise ValueError("DEPTH_CACHE_INVALID") from error
    if (
        not all(math.isfinite(value) for value in (low, high, edge_fraction, high - low))
        or high - low <= max(1e-6, abs(high) * 1e-6)
        or not 0 <= edge_fraction <= 1
    ):
        raise ValueError("DEPTH_CACHE_INVALID")

    encoded = metadata["_encoded"]
    if not isinstance(encoded, str) or len(encoded) > 700000:
        raise ValueError("DEPTH_CACHE_INVALID")
    data = base64.b64decode(encoded, validate=True)
    if (
        len(data) != w * h * 2
        or metadata["byteCount"] != len(data)
        or hashlib.sha256(data).hexdigest() != metadata["sha256"]
    ):
        raise ValueError("DEPTH_CACHE_INVALID")
    values = np.frombuffer(data, dtype="<u2")
    if values.min() != 0 or values.max() != 65535:
        raise ValueError("DEPTH_CACHE_INVALID")
    return data
