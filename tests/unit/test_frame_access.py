import subprocess
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
