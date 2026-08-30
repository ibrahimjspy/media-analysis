from __future__ import annotations

import hashlib

import cv2
import numpy as np
import pytest

from media_analysis.decode import ProbedMedia
from media_analysis.errors import CANCELLED, INVALID_REQUEST, AnalyzeError
from media_analysis.features.measure_common import middle_third_sample_frames
from media_analysis.features.thumbnails import (
    THUMBNAIL_JPEG_ENCODING_VERSION,
    THUMBNAIL_MIDDLE_THIRD_MAX_FRAMES,
    THUMBNAIL_MIME_TYPE,
    THUMBNAIL_SHORT_SHOT_WARNING,
    ThumbnailGrant,
    analyze_thumbnails,
    encode_thumbnail_jpeg,
    jpeg_encode_params,
    laplacian_sharpness,
    parse_thumbnail_grants,
    select_sharpest_frame,
    sha256_hex,
    thumbnail_policy_metadata,
    validate_one_grant_per_shot,
)
from media_analysis.frames import Rational


def _media(frame_count: int = 30) -> ProbedMedia:
    return ProbedMedia(
        width=80,
        height=60,
        fps=Rational(30, 1),
        frame_count=frame_count,
        duration=frame_count / 30.0,
        codec="h264",
        has_audio=False,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channel_count=None,
    )


def _frame_with_pattern(seed: int, *, blur: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
    if blur > 0:
        frame = cv2.GaussianBlur(frame, (blur | 1, blur | 1), 0)
    return frame


class DictFrameProvider:
    def __init__(self, frames: dict[int, np.ndarray]) -> None:
        self.frames = frames

    def __call__(self, frame_index: int) -> np.ndarray | None:
        return self.frames.get(frame_index)


def _grants(count: int) -> list[dict[str, str | int]]:
    return [
        {
            "index": index,
            "signedPutUrl": f"https://upload.example/{index}",
            "expiresAt": "2026-12-31T00:00:00Z",
        }
        for index in range(count)
    ]


@pytest.mark.unit
def test_middle_third_sample_frames_are_capped_and_evenly_spaced() -> None:
    capped = middle_third_sample_frames(0, 300, max_frames=7)
    assert len(capped) == 7
    assert capped[0] == 100
    assert capped[-1] == 199
    assert middle_third_sample_frames(0, 2, max_frames=7) == [0, 1]
    assert middle_third_sample_frames(5, 6, max_frames=7) == [5]


@pytest.mark.unit
def test_select_sharpest_frame_prefers_lower_index_on_tie() -> None:
    sharp = _frame_with_pattern(1, blur=0)
    frames = {10: sharp.copy(), 11: sharp.copy()}
    provider = DictFrameProvider(frames)
    chosen_index, _ = select_sharpest_frame([10, 11], provider)
    assert chosen_index == 10


@pytest.mark.unit
def test_select_sharpest_frame_from_middle_third() -> None:
    sharp = _frame_with_pattern(1, blur=0)
    blurry = _frame_with_pattern(1, blur=9)
    frames = {10: blurry, 11: sharp, 12: blurry}
    provider = DictFrameProvider(frames)
    chosen_index, chosen = select_sharpest_frame([10, 11, 12], provider)
    assert chosen_index == 11
    assert chosen is sharp
    assert laplacian_sharpness(cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)) > laplacian_sharpness(
        cv2.cvtColor(blurry, cv2.COLOR_BGR2GRAY)
    )


@pytest.mark.unit
def test_grant_alignment_validation() -> None:
    grants = parse_thumbnail_grants(_grants(2))
    validate_one_grant_per_shot(grants, 2)
    with pytest.raises(AnalyzeError) as exc:
        validate_one_grant_per_shot(grants, 3)
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_grant_indices_must_be_contiguous_from_zero() -> None:
    grants = parse_thumbnail_grants(
        [
            {"index": 1, "signedPutUrl": "https://upload.example/1", "expiresAt": "t"},
            {"index": 2, "signedPutUrl": "https://upload.example/2", "expiresAt": "t"},
        ]
    )
    with pytest.raises(AnalyzeError) as exc:
        validate_one_grant_per_shot(grants, 2)
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_jpeg_encode_requests_explicit_420_and_decodes() -> None:
    frame = _frame_with_pattern(42)
    params = jpeg_encode_params()
    assert cv2.IMWRITE_JPEG_SAMPLING_FACTOR in params
    assert cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420 in params
    payload = encode_thumbnail_jpeg(frame)
    assert payload.startswith(b"\xff\xd8")
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == frame.shape
    assert sha256_hex(payload) == hashlib.sha256(payload).hexdigest()
    assert "420" in THUMBNAIL_JPEG_ENCODING_VERSION


@pytest.mark.unit
def test_jpeg_encoding_version_constant_is_declared() -> None:
    meta = thumbnail_policy_metadata()
    assert meta["jpegEncodingVersion"] == THUMBNAIL_JPEG_ENCODING_VERSION
    assert meta["shortShotWarningCode"] == THUMBNAIL_SHORT_SHOT_WARNING
    assert meta["middleThirdMaxFrames"] == THUMBNAIL_MIDDLE_THIRD_MAX_FRAMES
    assert meta["provisional"] is True


@pytest.mark.unit
def test_analyze_thumbnails_upload_callback_once_per_shot_with_expiry() -> None:
    media = _media(9)
    shots = [
        {"startFrame": 0, "endFrameExclusive": 3},
        {"startFrame": 3, "endFrameExclusive": 9},
    ]
    frames = {index: _frame_with_pattern(index) for index in range(9)}
    uploads: list[tuple[ThumbnailGrant, bytes]] = []

    def upload(grant: ThumbnailGrant, payload: bytes) -> None:
        uploads.append((grant, payload))

    result = analyze_thumbnails(
        path=__file__,
        media=media,
        shots=shots,
        thumbnail_grants=_grants(2),
        frame_provider=DictFrameProvider(frames),
        upload_callback=upload,
    )
    assert result.status == "completed"
    assert len(result.candidates) == 2
    assert len(uploads) == 2
    assert THUMBNAIL_SHORT_SHOT_WARNING in result.warning_codes
    for candidate, (grant, payload) in zip(result.candidates, uploads, strict=True):
        assert grant.index == candidate["grantIndex"]
        assert grant.expires_at == "2026-12-31T00:00:00Z"
        assert grant.signed_put_url.endswith(f"/{grant.index}")
        assert candidate["mimeType"] == THUMBNAIL_MIME_TYPE
        assert candidate["byteCount"] == len(payload)
        assert candidate["sha256"] == sha256_hex(payload)
        assert "signedPutUrl" not in candidate
        assert "derivedAssetId" not in candidate
        assert "url" not in candidate


@pytest.mark.unit
def test_analyze_thumbnails_caps_decode_count_on_long_shot() -> None:
    media = _media(300)
    decode_calls: list[int] = []

    def provider(frame_index: int) -> np.ndarray:
        decode_calls.append(frame_index)
        return _frame_with_pattern(frame_index)

    result = analyze_thumbnails(
        path=__file__,
        media=media,
        shots=[{"startFrame": 0, "endFrameExclusive": 300}],
        thumbnail_grants=_grants(1),
        frame_provider=provider,
        middle_third_max_frames=7,
    )
    assert len(result.candidates) == 1
    assert len(decode_calls) == 7


@pytest.mark.unit
def test_analyze_thumbnails_uses_prior_facts_and_middle_third() -> None:
    media = _media(12)
    frames = {index: _frame_with_pattern(index) for index in range(12)}
    frames[7] = _frame_with_pattern(99, blur=0)
    for index in range(4, 9):
        if index != 7:
            frames[index] = _frame_with_pattern(index, blur=11)
    prior = {"shots": [{"startFrame": 0, "endFrameExclusive": 12}]}
    result = analyze_thumbnails(
        path=__file__,
        media=media,
        prior_facts=prior,
        fill_only=True,
        thumbnail_grants=_grants(1),
        frame_provider=DictFrameProvider(frames),
    )
    assert result.candidates[0]["sourceFrame"] == 7


@pytest.mark.unit
def test_analyze_thumbnails_cancellation_before_upload() -> None:
    media = _media(6)
    shots = [{"startFrame": 0, "endFrameExclusive": 6}]
    frames = {index: _frame_with_pattern(index) for index in range(6)}
    uploads: list[tuple[ThumbnailGrant, bytes]] = []

    def cancel_check() -> None:
        raise AnalyzeError(CANCELLED, "Analysis cancelled")

    with pytest.raises(AnalyzeError) as exc:
        analyze_thumbnails(
            path=__file__,
            media=media,
            shots=shots,
            thumbnail_grants=_grants(1),
            frame_provider=DictFrameProvider(frames),
            upload_callback=lambda grant, payload: uploads.append((grant, payload)),
            cancel_check=cancel_check,
        )
    assert exc.value.code == CANCELLED
    assert uploads == []


@pytest.mark.unit
def test_analyze_thumbnails_does_not_skip_failed_shot_decode() -> None:
    media = _media(6)
    shots = [
        {"startFrame": 0, "endFrameExclusive": 3},
        {"startFrame": 3, "endFrameExclusive": 6},
    ]
    frames = {0: _frame_with_pattern(0), 1: _frame_with_pattern(1), 2: _frame_with_pattern(2)}
    with pytest.raises(RuntimeError, match="could not decode thumbnail candidates"):
        analyze_thumbnails(
            path=__file__,
            media=media,
            shots=shots,
            thumbnail_grants=_grants(2),
            frame_provider=DictFrameProvider(frames),
        )


@pytest.mark.unit
def test_deterministic_jpeg_bytes_for_same_frame() -> None:
    frame = _frame_with_pattern(7)
    first = encode_thumbnail_jpeg(frame)
    second = encode_thumbnail_jpeg(frame)
    assert first == second
