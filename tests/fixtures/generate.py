"""Generate redistribution-safe golden clips with ffmpeg.

These fixtures are created at test time. They are not third-party media and
may be regenerated on any machine with ffmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
)


def _run(args: list[str]) -> None:
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"ffmpeg failed ({completed.returncode}): {detail}")


def _font_file() -> Path | None:
    return next((path for path in _FONT_CANDIDATES if path.is_file()), None)


def write_people_shapes_mp4(path: Path, *, seconds: float = 1.0) -> Path:
    """Moving upright boxes that stand in for people until a rights-cleared reel exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x240:d={seconds}:r=30",
            "-vf",
            "drawbox=x='80+20*sin(2*PI*t)':y=80:w=48:h=96:color=white:t=fill,"
            "drawbox=x='200+16*cos(2*PI*t)':y=90:w=40:h=80:color=gray:t=fill",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return path


def write_faces_circles_mp4(path: Path, *, seconds: float = 1.0) -> Path:
    """Circular blobs that exercise face-sized geometry without using real faces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x335577:s=320x240:d={seconds}:r=30",
            "-vf",
            "drawbox=x=88:y=48:w=64:h=64:color=0xffcc99:t=fill,"
            "drawbox=x=176:y=56:w=56:h=56:color=0xffcc99:t=fill",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return path


def write_image_loop_mp4(
    path: Path,
    image: np.ndarray,
    *,
    seconds: float = 1.0,
    fps: int = 30,
) -> Path:
    """Encode a BGR image as a short CFR H.264 clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    import cv2

    if not cv2.imwrite(str(png), image):
        raise RuntimeError(f"could not write {png}")
    _run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(png),
            "-t",
            str(seconds),
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return path


def write_ocr_text_mp4(path: Path, *, text: str = "HELLO WORLD", seconds: float = 1.0) -> Path:
    """Burned-in Latin text, or testsrc2 timestamps when no system font is present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    font = _font_file()
    if font is None:
        _run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size=320x240:rate=30:duration={seconds}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(path),
            ]
        )
        return path
    escaped = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x240:d={seconds}:r=30",
            "-vf",
            f"drawtext=fontfile={font}:text='{escaped}':fontsize=36:"
            "fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return path


def write_speech_tone_mp4(path: Path, *, seconds: float = 1.0) -> Path:
    """Spoken-audio stand-in: a 440 Hz tone over a silent-looking frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x240:d={seconds}:r=30",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )
    return path


def write_vfr_mp4(path: Path) -> Path:
    """Concat 15 fps and 30 fps CFR segments without re-timing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    first = path.with_name(f"{path.stem}-15fps.mp4")
    second = path.with_name(f"{path.stem}-30fps.mp4")
    listing = path.with_name(f"{path.stem}-concat.txt")
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:d=1:r=15",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(first),
        ]
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=1:r=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(second),
        ]
    )
    listing.write_text(
        f"file '{first.resolve()}'\nfile '{second.resolve()}'\n",
        encoding="utf-8",
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(path),
        ]
    )
    return path


def write_rotated_mp4(path: Path, *, seconds: float = 1.0) -> Path:
    """Landscape pixels tagged with 90-degree stream rotation metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_name(f"{path.stem}-raw.mp4")
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:s=320x240:d={seconds}:r=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(raw),
        ]
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw),
            "-c",
            "copy",
            "-metadata:s:v:0",
            "rotate=90",
            str(path),
        ]
    )
    return path
