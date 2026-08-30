"""Synthetic frame helpers for motion and quality unit tests."""

from __future__ import annotations

import cv2
import numpy as np

from media_analysis.features.measure_common import FrameProvider


def checkerboard(width: int = 320, height: int = 240, cell: int = 16) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if ((x // cell) + (y // cell)) % 2 == 0:
                frame[y, x] = (220, 220, 220)
            else:
                frame[y, x] = (40, 40, 40)
    return frame


def solid_color(
    width: int = 320,
    height: int = 240,
    color: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def static_sequence(frame_count: int, width: int = 320, height: int = 240) -> dict[int, np.ndarray]:
    base = checkerboard(width, height)
    return {index: base.copy() for index in range(frame_count)}


def solid_sequence(
    frame_count: int,
    *,
    width: int = 320,
    height: int = 240,
    color: tuple[int, int, int] = (128, 128, 128),
) -> dict[int, np.ndarray]:
    base = solid_color(width, height, color)
    return {index: base.copy() for index in range(frame_count)}


def pan_sequence(
    frame_count: int,
    *,
    width: int = 320,
    height: int = 240,
    shift_px: int = 4,
) -> dict[int, np.ndarray]:
    base = checkerboard(width, height)
    frames: dict[int, np.ndarray] = {}
    for index in range(frame_count):
        matrix = np.float32([[1, 0, -index * shift_px], [0, 1, 0]])
        frames[index] = cv2.warpAffine(base, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    return frames


def tilt_sequence(
    frame_count: int,
    *,
    width: int = 320,
    height: int = 240,
    shift_px: int = 4,
) -> dict[int, np.ndarray]:
    base = checkerboard(width, height)
    frames: dict[int, np.ndarray] = {}
    for index in range(frame_count):
        matrix = np.float32([[1, 0, 0], [0, 1, -index * shift_px]])
        frames[index] = cv2.warpAffine(base, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    return frames


def scale_sequence(
    frame_count: int,
    *,
    width: int = 320,
    height: int = 240,
    scale_step: float = 1.012,
) -> dict[int, np.ndarray]:
    base = checkerboard(width, height)
    frames: dict[int, np.ndarray] = {}
    for index in range(frame_count):
        scale = scale_step**index
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale)
        frames[index] = cv2.warpAffine(base, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    return frames


def noisy_sequence(frames: dict[int, np.ndarray], sigma: float = 18.0) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        index: np.clip(
            frame.astype(np.float32) + rng.normal(0, sigma, frame.shape), 0, 255
        ).astype(np.uint8)
        for index, frame in frames.items()
    }


def blurred_frame(frame: np.ndarray, ksize: int = 21) -> np.ndarray:
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def clipped_highlights(frame: np.ndarray, fraction: float = 0.25) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    region_h = max(1, int(height * fraction))
    out[:region_h, :] = 255
    return out


def clipped_shadows(frame: np.ndarray, fraction: float = 0.25) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    region_h = max(1, int(height * fraction))
    out[-region_h:, :] = 0
    return out


def dict_frame_provider(frames: dict[int, np.ndarray | None]) -> FrameProvider:
    def provider(index: int) -> np.ndarray | None:
        if index not in frames:
            return None
        value = frames[index]
        if value is None:
            return None
        return value.copy()

    return provider


class RecordingFrameProvider:
    """Frame provider that records decode indices for reuse assertions."""

    def __init__(self, frames: dict[int, np.ndarray | None]) -> None:
        self._frames = frames
        self.read_indices: list[int] = []

    def read(self, index: int) -> np.ndarray | None:
        self.read_indices.append(index)
        return dict_frame_provider(self._frames)(index)

    def __call__(self, index: int) -> np.ndarray | None:
        return self.read(index)
