"""FFmpeg, runtime, and machine identity for analyze provenance."""

from __future__ import annotations

import os
import platform
import subprocess
from functools import lru_cache
from typing import Any


def _first_token_after(text: str, prefix: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            rest = stripped[len(prefix) :].strip()
            return rest.split(" ", 1)[0] if rest else None
    return None


def _configuration_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("configuration:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def _tool_version_text(binary: str) -> str:
    try:
        completed = subprocess.run(
            [binary, "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout or completed.stderr


@lru_cache
def collect_ffmpeg_build() -> dict[str, str]:
    ffmpeg_text = _tool_version_text("ffmpeg")
    ffprobe_text = _tool_version_text("ffprobe")
    return {
        "ffmpegVersion": _first_token_after(ffmpeg_text, "ffmpeg version") or "unavailable",
        "ffmpegConfiguration": _configuration_line(ffmpeg_text),
        "ffprobeVersion": _first_token_after(ffprobe_text, "ffprobe version") or "unavailable",
    }


@lru_cache
def collect_runtime_build() -> dict[str, str]:
    import cv2
    import numpy
    import onnxruntime

    return {
        "pythonVersion": platform.python_version(),
        "onnxruntimeVersion": onnxruntime.__version__,
        "opencvVersion": cv2.__version__,
        "numpyVersion": numpy.__version__,
    }


def collect_machine_fingerprint() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "pythonVersion": platform.python_version(),
        "cpuCount": os.cpu_count(),
    }


def analyze_build_provenance() -> dict[str, dict[str, str]]:
    return {
        "ffmpegBuild": collect_ffmpeg_build(),
        "runtimeBuild": collect_runtime_build(),
    }
