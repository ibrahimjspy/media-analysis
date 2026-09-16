"""Bounded still-image normalization. Canonical pixels are oriented opaque sRGB."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from media_analysis.config import Settings
from media_analysis.errors import DECODE_FAILED, INVALID_REQUEST, LIMIT_EXCEEDED, AnalyzeError
from media_analysis.schemas import AnalyzeRequest

register_heif_opener()
IMAGE_DECODE_VERSION = "image-srgb-exif-alpha-v1"
IMAGE_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "HEIF": "image/heif",
}
# Homogeneous transforms of normalized pixel-edge coordinates, before EXIF transpose.
ORIENTATION_TRANSFORMS = {
    1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    2: [[-1, 0, 1], [0, 1, 0], [0, 0, 1]],
    3: [[-1, 0, 1], [0, -1, 1], [0, 0, 1]],
    4: [[1, 0, 0], [0, -1, 1], [0, 0, 1]],
    5: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
    6: [[0, -1, 1], [1, 0, 0], [0, 0, 1]],
    7: [[0, -1, 1], [-1, 0, 1], [0, 0, 1]],
    8: [[0, 1, 0], [-1, 0, 1], [0, 0, 1]],
}


@dataclass
class DecodedImage:
    canonical: Image.Image
    bgr: np.ndarray
    metadata: dict[str, Any]

    def png(self) -> bytes:
        target = io.BytesIO()
        self.canonical.save(target, format="PNG")
        return target.getvalue()


def decode_image(path: Path, request: AnalyzeRequest, settings: Settings) -> DecodedImage:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                fmt = source.format
                if fmt not in IMAGE_FORMATS:
                    raise AnalyzeError(DECODE_FAILED, "Unsupported still-image format")
                if source.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK"}:
                    raise AnalyzeError(
                        DECODE_FAILED, "Only 8-bit still-image channels are supported"
                    )
                width, height = source.size
                if width * height > settings.media_analysis_max_image_pixels:
                    raise AnalyzeError(LIMIT_EXCEEDED, "Image exceeds decoded pixel limit")
                if getattr(source, "n_frames", 1) != 1:
                    raise AnalyzeError(
                        INVALID_REQUEST, "Animated/multi-image sources are unsupported"
                    )
                orientation = int(source.getexif().get(274, 1))
                if orientation not in ORIENTATION_TRANSFORMS:
                    raise AnalyzeError(DECODE_FAILED, "Invalid EXIF orientation")
                profile = source.info.get("icc_profile")
                decoder_orientation = source.info.get("original_orientation")
                if fmt == "HEIF":
                    nclx = source.info.get("nclx_profile") or {}
                    if source.info.get("bit_depth", 8) != 8 or (
                        not profile
                        and (
                            nclx.get("color_primaries", 1) != 1
                            or nclx.get("transfer_characteristics", 13) != 13
                        )
                    ):
                        raise AnalyzeError(DECODE_FAILED, "Only SDR sRGB/ICC HEIF is supported")
                oriented = ImageOps.exif_transpose(source)
                alpha = oriented.convert("RGBA").getchannel("A")
                if profile:
                    color = ImageCms.profileToProfile(
                        oriented,
                        ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                        ImageCms.createProfile("sRGB"),
                        outputMode="RGB",
                    )
                else:
                    color = oriented.convert("RGB")
                background = (
                    request.imageOptions.alphaBackground if request.imageOptions else "white"
                )
                canonical = Image.new("RGB", color.size, background)
                canonical.paste(color, mask=alpha)
                # No original EXIF/ICC metadata is copied to the canonical derivative.
                canonical.info.clear()
    except AnalyzeError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise AnalyzeError(LIMIT_EXCEEDED, "Image exceeds decoded pixel limit") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, ImageCms.PyCMSError) as exc:
        raise AnalyzeError(DECODE_FAILED, "Could not decode image/color profile") from exc
    cw, ch = canonical.size
    limit = min(
        settings.media_analysis_image_max_dimension,
        request.imageOptions.maxDimension if request.imageOptions else 1280,
    )
    scale = min(1.0, limit / max(cw, ch))
    aw, ah = max(1, round(cw * scale)), max(1, round(ch * scale))
    if request.analysisResolution:
        aw, ah = request.analysisResolution.width, request.analysisResolution.height
        if aw > cw or ah > ch or max(aw, ah) > limit or abs(aw / cw - ah / ch) > 1e-6:
            raise AnalyzeError(
                INVALID_REQUEST, "Image analysisResolution must fit bounds and aspect"
            )
    working = canonical.resize((aw, ah), Image.Resampling.LANCZOS)
    bgr = cv2.cvtColor(np.asarray(working), cv2.COLOR_RGB2BGR)
    return DecodedImage(
        canonical,
        bgr,
        {
            "decoderOrientationPolicy": "libheif-normalized-raster"
            if fmt == "HEIF"
            else "EXIF-transpose",
            "decoderReportedOrientation": decoder_orientation,
            "kind": "image",
            "width": cw,
            "height": ch,
            "aspectRatio": cw / ch,
            "mimeType": "image/png",
            "sourceFormat": fmt,
            "sourceMimeType": IMAGE_FORMATS[fmt],
            "originalWidth": width,
            "originalHeight": height,
            "exifOrientation": orientation,
            "originalToCanonicalNormalized": ORIENTATION_TRANSFORMS[orientation],
            "colorSpace": "sRGB",
            "colorPolicy": "ICC-to-sRGB; untagged-assumed-sRGB",
            "alphaPolicy": f"composite-{background}-in-sRGB",
            "canonicalizationVersion": IMAGE_DECODE_VERSION,
            "analysisWidth": aw,
            "analysisHeight": ah,
            "analysisToCanonicalScale": {"x": cw / aw, "y": ch / ah},
        },
    )
