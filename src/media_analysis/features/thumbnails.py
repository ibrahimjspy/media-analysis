"""Per-shot thumbnail candidate selection and JPEG upload (v1.2)."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from media_analysis.decode import ProbedMedia
from media_analysis.errors import INVALID_REQUEST, UPLOAD_FAILED, AnalyzeError
from media_analysis.features.measure_common import (
    FrameProvider,
    PriorFactsInput,
    ShotsInput,
    middle_third_sample_frames,
    resolve_shots,
    shot_ranges,
)

# Provisional until benchmark; Linux x86_64 OpenCV is source of truth for JPEG 4:2:0.
THUMBNAIL_ANALYZER_VERSION = "middle-third-laplacian-cap7-provisional-1.0.0"
THUMBNAIL_JPEG_ENCODING_VERSION = "opencv-jpeg-q85-420-provisional-1.0.0"
THUMBNAIL_SHORT_SHOT_WARNING = "THUMBNAIL_SHORT_SHOT"
THUMBNAIL_MIME_TYPE = "image/jpeg"

# Provisional until benchmark.
SHORT_SHOT_FRAME_THRESHOLD = 3
THUMBNAIL_MIDDLE_THIRD_MAX_FRAMES = 7
JPEG_QUALITY = 85
SOURCE_OF_TRUTH_PLATFORM = "linux-x86_64"


@dataclass(frozen=True, slots=True)
class ThumbnailGrant:
    index: int
    signed_put_url: str
    expires_at: str


UploadCallback = Callable[[ThumbnailGrant, bytes], None]


@dataclass(frozen=True, slots=True)
class ThumbnailCandidate:
    shot_index: int
    source_frame: int
    width: int
    height: int
    mime_type: Literal["image/jpeg"]
    sha256: str
    byte_count: int
    grant_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "shotIndex": self.shot_index,
            "sourceFrame": self.source_frame,
            "width": self.width,
            "height": self.height,
            "mimeType": self.mime_type,
            "sha256": self.sha256,
            "byteCount": self.byte_count,
            "grantIndex": self.grant_index,
        }


@dataclass(frozen=True, slots=True)
class ThumbnailAnalyzeResult:
    candidates: list[dict[str, Any]]
    status: Literal["completed", "failed"] = "completed"
    version: str = THUMBNAIL_ANALYZER_VERSION
    warning_codes: tuple[str, ...] = ()


def _jpeg_420_supported() -> bool:
    return hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR") and hasattr(
        cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_420"
    )


def jpeg_encode_params() -> list[int]:
    """Build deterministic OpenCV JPEG params; require explicit 4:2:0 when supported."""
    params = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        JPEG_QUALITY,
        int(cv2.IMWRITE_JPEG_OPTIMIZE),
        0,
    ]
    if not _jpeg_420_supported():
        raise RuntimeError(
            "OpenCV JPEG 4:2:0 sampling factor is unavailable; "
            f"{SOURCE_OF_TRUTH_PLATFORM} build must provide IMWRITE_JPEG_SAMPLING_FACTOR_420"
        )
    params.extend(
        [
            int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
            int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420),
        ]
    )
    return params


def parse_thumbnail_grants(raw_grants: list[dict[str, Any]] | None) -> list[ThumbnailGrant]:
    if not raw_grants:
        raise AnalyzeError(INVALID_REQUEST, "outputGrants.thumbnails is required for thumbnails")
    grants: list[ThumbnailGrant] = []
    for item in raw_grants:
        grants.append(
            ThumbnailGrant(
                index=int(item["index"]),
                signed_put_url=str(item["signedPutUrl"]),
                expires_at=str(item["expiresAt"]),
            )
        )
    return sorted(grants, key=lambda grant: grant.index)


def validate_one_grant_per_shot(grants: list[ThumbnailGrant], shot_count: int) -> None:
    if shot_count <= 0:
        raise AnalyzeError(INVALID_REQUEST, "thumbnails requires at least one shot")
    if len(grants) != shot_count:
        raise AnalyzeError(
            INVALID_REQUEST,
            "outputGrants.thumbnails must contain exactly one grant per shot",
        )
    expected = list(range(shot_count))
    actual = [grant.index for grant in grants]
    if actual != expected:
        raise AnalyzeError(
            INVALID_REQUEST,
            "outputGrants.thumbnails grant index must align 0..shotCount-1",
        )


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_sharpest_frame(
    candidate_frames: list[int],
    read_frame: Callable[[int], np.ndarray | None],
    *,
    cancel_check: Callable[[], None] | None = None,
) -> tuple[int, np.ndarray]:
    best_frame = -1
    best_image: np.ndarray | None = None
    best_score = -1.0
    decoded_any = False
    for frame_index in candidate_frames:
        if cancel_check:
            cancel_check()
        frame = read_frame(frame_index)
        if frame is None:
            continue
        decoded_any = True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = laplacian_sharpness(gray)
        if score > best_score or (score == best_score and frame_index < best_frame):
            best_score = score
            best_frame = frame_index
            best_image = frame
    if best_image is None or best_frame < 0:
        if decoded_any:
            raise RuntimeError("could not score thumbnail candidates")
        raise RuntimeError(f"could not decode thumbnail candidates for frames {candidate_frames}")
    return best_frame, best_image


def encode_thumbnail_jpeg(frame_bgr: np.ndarray) -> bytes:
    """Deterministic JPEG bytes with explicit 4:2:0 when OpenCV supports it."""
    ok, encoded = cv2.imencode(".jpg", frame_bgr, jpeg_encode_params())
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return encoded.tobytes()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def analyze_thumbnails(
    path: Path,
    media: ProbedMedia,
    *,
    shots: ShotsInput = None,
    prior_facts: PriorFactsInput = None,
    fill_only: bool = False,
    thumbnail_grants: list[dict[str, Any]] | None = None,
    upload_callback: UploadCallback | None = None,
    frame_provider: FrameProvider | None = None,
    cancel_check: Callable[[], None] | None = None,
    middle_third_max_frames: int = THUMBNAIL_MIDDLE_THIRD_MAX_FRAMES,
) -> ThumbnailAnalyzeResult:
    """Pick, encode, and upload one sharp middle-third frame per shot."""
    resolved = resolve_shots(
        prior_facts=prior_facts,
        supplied_shots=shots,
        frame_count=media.frame_count,
        fill_only=fill_only,
    )
    ranges = shot_ranges(resolved, media.frame_count)
    grants = parse_thumbnail_grants(thumbnail_grants)
    validate_one_grant_per_shot(grants, len(ranges))

    reader = _OpenCvFrameProvider(path) if frame_provider is None else frame_provider
    owns_reader = frame_provider is None
    read_frame = reader if callable(reader) else reader.read

    warnings: list[str] = []
    candidates: list[dict[str, Any]] = []
    pending_uploads: list[tuple[ThumbnailGrant, bytes]] = []

    try:
        for shot_index, (start, end) in enumerate(ranges):
            if cancel_check:
                cancel_check()
            length = end - start
            if length <= SHORT_SHOT_FRAME_THRESHOLD:
                warnings.append(THUMBNAIL_SHORT_SHOT_WARNING)
            candidate_frames = middle_third_sample_frames(
                start,
                end,
                max_frames=middle_third_max_frames,
            )
            if not candidate_frames:
                raise RuntimeError(f"shot {shot_index} has no thumbnail candidate frames")
            source_frame, frame = select_sharpest_frame(
                candidate_frames,
                read_frame,
                cancel_check=cancel_check,
            )
            grant = grants[shot_index]
            jpeg = encode_thumbnail_jpeg(frame)
            candidate = ThumbnailCandidate(
                shot_index=shot_index,
                source_frame=source_frame,
                width=frame.shape[1],
                height=frame.shape[0],
                mime_type=THUMBNAIL_MIME_TYPE,
                sha256=sha256_hex(jpeg),
                byte_count=len(jpeg),
                grant_index=grant.index,
            )
            pending_uploads.append((grant, jpeg))
            candidates.append(candidate.as_dict())
    finally:
        if owns_reader and isinstance(reader, _OpenCvFrameProvider):
            reader.close()

    if upload_callback is not None:
        for grant, payload in pending_uploads:
            if cancel_check:
                cancel_check()
            try:
                upload_callback(grant, payload)
            except AnalyzeError:
                raise
            except Exception as exc:
                raise AnalyzeError(UPLOAD_FAILED, "Thumbnail upload failed") from exc

    deduped_warnings = tuple(dict.fromkeys(warnings))
    return ThumbnailAnalyzeResult(
        candidates=candidates,
        status="completed",
        warning_codes=deduped_warnings,
    )


@dataclass(slots=True)
class _OpenCvFrameProvider:
    path: Path
    _capture: cv2.VideoCapture | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Could not open video for thumbnail analysis")
        self._capture = capture

    def __call__(self, frame_index: int) -> np.ndarray | None:
        if self._capture is None:
            return None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def read(self, frame_index: int) -> np.ndarray | None:
        return self(frame_index)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def thumbnail_policy_metadata() -> dict[str, str | int | bool]:
    """Versioned knobs for parent provenance / cache identity wiring."""
    return {
        "analyzerVersion": THUMBNAIL_ANALYZER_VERSION,
        "jpegEncodingVersion": THUMBNAIL_JPEG_ENCODING_VERSION,
        "shortShotFrameThreshold": SHORT_SHOT_FRAME_THRESHOLD,
        "shortShotWarningCode": THUMBNAIL_SHORT_SHOT_WARNING,
        "jpegQuality": JPEG_QUALITY,
        "middleThirdMaxFrames": THUMBNAIL_MIDDLE_THIRD_MAX_FRAMES,
        "jpeg420Explicit": _jpeg_420_supported(),
        "sourceOfTruthPlatform": SOURCE_OF_TRUTH_PLATFORM,
        "runtimePlatform": sys.platform,
        "provisional": True,
    }
