import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from media_analysis.errors import CANCELLED, AnalyzeError
from media_analysis.frame_access import (
    BoundedFrameAccess,
    FrameAccessConfig,
    cancel_check_from_flag,
    iter_half_open_shot_frames,
)


def make_probe_mp4(path: Path, *, frames: int = 6) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    seconds = frames / 30
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s=160x120:d={seconds}:r=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.mark.unit
def test_iter_half_open_shot_frames() -> None:
    shots = [
        {"startFrame": 0, "endFrameExclusive": 3},
        {"startFrame": 3, "endFrameExclusive": 6},
    ]
    assert list(iter_half_open_shot_frames(shots, step=1)) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (1, 5),
    ]


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_bounded_frame_access_reads_and_caches(tmp_path: Path) -> None:
    video = make_probe_mp4(tmp_path / "clip.mp4", frames=6)
    with BoundedFrameAccess(video, config=FrameAccessConfig(max_cached_frames=2)) as access:
        first = access.read_bgr(0)
        second = access.read_bgr(0)
        assert not np.shares_memory(first, second)
        original = int(second[0, 0, 0])
        first[0, 0, 0] = 42 if original != 42 else 43
        third = access.read_bgr(0)
        assert int(third[0, 0, 0]) == original
        access.read_bgr(1)
        access.read_bgr(2)
        assert len(access.cached_frame_indices()) <= 2


@pytest.mark.unit
def test_frame_access_config_rejects_nonpositive_cache() -> None:
    with pytest.raises(ValueError, match="max_cached_frames"):
        FrameAccessConfig(max_cached_frames=0)


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_bounded_frame_access_cleans_up_on_exit(tmp_path: Path) -> None:
    video = make_probe_mp4(tmp_path / "clip.mp4", frames=3)
    access = BoundedFrameAccess(video)
    with access:
        access.read_bgr(0)
        assert access.cached_frame_indices()
    assert not access.cached_frame_indices()
    assert access._capture is None  # noqa: SLF001


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_bounded_frame_access_honors_cancellation(tmp_path: Path) -> None:
    video = make_probe_mp4(tmp_path / "clip.mp4", frames=3)
    cancelled = False

    def is_cancelled() -> bool:
        return cancelled

    with BoundedFrameAccess(video, cancel_check=cancel_check_from_flag(is_cancelled)) as access:
        access.read_bgr(0)
        cancelled = True
        with pytest.raises(AnalyzeError) as exc:
            access.read_bgr(1)
        assert exc.value.code == CANCELLED


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_frame_access_coalesces_concurrent_reads_and_builds_gray_view(tmp_path: Path) -> None:
    video = make_probe_mp4(tmp_path / "clip.mp4", frames=6)
    with BoundedFrameAccess(video) as access:
        with ThreadPoolExecutor(max_workers=4) as pool:
            frames = list(pool.map(access.read_bgr, [2, 2, 2, 2]))
        gray = access.read_gray(2, max_dimension=80)
        assert all(frame.shape == frames[0].shape for frame in frames)
        assert gray.ndim == 2
        assert max(gray.shape) <= 80
        assert access.unique_decodes == 1
        assert access.cache_hits == 4


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_repeated_pass_reuses_samples_after_memory_eviction(tmp_path: Path) -> None:
    video = make_probe_mp4(tmp_path / "clip.mp4", frames=60)
    config = FrameAccessConfig(max_cached_frames=2, max_cached_bytes=120000)
    access = BoundedFrameAccess(video, config=config)
    with access:
        indices = list(range(0, 60, 6)) + [59]
        first = [access.read_bgr(index) for index in indices]
        assert access.unique_decodes == len(indices)
        spill_bytes_before_replay = access.spill_bytes
        for index, expected in zip(indices, first, strict=True):
            actual = access.read_bgr(index)
            np.testing.assert_array_equal(actual, expected)
        assert access.unique_decodes == len(indices)
        assert access.disk_hits > 0
        assert access.spill_bytes == spill_bytes_before_replay
        assert access._cache_bytes + access._gray_cache_bytes <= config.max_cached_bytes
        assert 0 < access.spill_bytes <= config.max_spill_bytes
        spill_root = Path(access._spill_dir.name)
    assert not spill_root.exists()


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_gray_eviction_reuses_tapped_samples_without_redecode(tmp_path: Path) -> None:
    video = make_probe_mp4(tmp_path / "clip.mp4", frames=6)
    config = FrameAccessConfig(max_cached_bytes=4000)
    with BoundedFrameAccess(video, config=config) as access:
        for index in range(6):
            access.prime_gray(index, np.full((60, 80, 3), index, np.uint8), max_dimension=80)
        for index in range(6):
            gray = access.read_gray(index, max_dimension=80)
            assert np.all(gray == index)
        assert access.unique_decodes == 0
        assert access._cache_bytes + access._gray_cache_bytes <= config.max_cached_bytes


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_disk_budget_exhaustion_keeps_decode_available(tmp_path: Path) -> None:
    video = make_probe_mp4(tmp_path / "clip.mp4", frames=3)
    config = FrameAccessConfig(max_cached_frames=1, max_spill_bytes=10)
    with BoundedFrameAccess(video, config=config) as access:
        expected = access.read_bgr(0)
        access.read_bgr(1)
        np.testing.assert_array_equal(access.read_bgr(0), expected)
        assert access.spill_bytes <= 10
        assert access.spill_limited
