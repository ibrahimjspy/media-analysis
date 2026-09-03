"""Generate redistribution-safe golden clips with ffmpeg.

These fixtures are created at test time. They are not third-party media and
may be regenerated on any machine with ffmpeg.
"""

from __future__ import annotations

import subprocess
import wave
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


def _synthetic_speech(seconds: float, *, sample_rate: int = 16_000) -> np.ndarray:
    """Generate a deterministic voiced formant sequence without external speech assets."""
    rng = np.random.default_rng(7)
    phonemes: tuple[tuple[float, tuple[int, int, int] | None], ...] = (
        (0.22, (730, 1090, 2440)),
        (0.18, (270, 2290, 3010)),
        (0.24, (530, 1840, 2480)),
        (0.16, None),
        (0.25, (570, 840, 2410)),
        (0.22, (300, 870, 2240)),
        (0.25, (660, 1720, 2410)),
        (0.18, None),
        (0.28, (400, 2000, 2550)),
    )
    duration_scale = seconds / sum(duration for duration, _ in phonemes)
    chunks: list[np.ndarray] = []
    phase = 0.0
    for index, (raw_duration, formants) in enumerate(phonemes):
        duration = raw_duration * duration_scale
        sample_count = max(1, round(sample_rate * duration))
        time = np.arange(sample_count, dtype=np.float64) / sample_rate
        attack = max(0.005, min(0.025, duration * 0.12))
        release = max(0.005, min(0.035, duration * 0.18))
        envelope = np.minimum(1.0, time / attack) * np.minimum(
            1.0, np.maximum(0.0, duration - time) / release
        )
        if formants is None:
            chunk = rng.normal(0.0, 0.12, sample_count) * envelope
        else:
            fundamental = 115.0 + 18.0 * np.sin(2.0 * np.pi * 0.7 * time + index)
            phase_values = phase + 2.0 * np.pi * np.cumsum(fundamental) / sample_rate
            phase = float(phase_values[-1] % (2.0 * np.pi))
            chunk = np.zeros(sample_count, dtype=np.float64)
            for harmonic in range(1, 45):
                frequency = harmonic * 130.0
                gain = sum(
                    np.exp(-0.5 * ((frequency - formant) / (110.0 + 0.05 * formant)) ** 2)
                    for formant in formants
                )
                chunk += gain / harmonic**0.8 * np.sin(harmonic * phase_values)
            chunk += rng.normal(0.0, 0.015, sample_count)
            chunk *= envelope
            chunk *= 0.35 / max(float(np.max(np.abs(chunk))), 1e-8)
        chunks.append(chunk.astype(np.float32))

    target_count = max(1, round(sample_rate * seconds))
    samples = np.concatenate(chunks)
    return np.pad(samples[:target_count], (0, max(0, target_count - samples.size)))


def write_speech_tone_mp4(path: Path, *, seconds: float = 1.0) -> Path:
    """Mux deterministic speech-like formants over a silent-looking frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    samples = _synthetic_speech(seconds, sample_rate=sample_rate)
    wav_path = path.with_suffix(".speech.wav")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x240:d={seconds}:r=30",
            "-i",
            str(wav_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
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
