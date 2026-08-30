"""Fake MODNet ONNX sessions for matte unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FakeModnetSession:
    """Returns deterministic alpha from input tensor mean (0/128/255 patches)."""

    input_name: str = "input"
    output_name: str = "output"
    scale: float = 1.0

    def run(self, output_names: list[str], input_feed: dict[str, Any]) -> list[np.ndarray]:
        tensor = input_feed[self.input_name]
        mean = float(tensor.mean())
        if mean < -0.5:
            value = 0.0
        elif mean > 0.5:
            value = 1.0
        else:
            value = 128.0 / 255.0
        _, _, height, width = tensor.shape
        alpha = np.full((1, 1, height, width), value * self.scale, dtype=np.float32)
        return [alpha]

    def get_inputs(self) -> list[Any]:
        return [type("IO", (), {"name": self.input_name})()]

    def get_outputs(self) -> list[Any]:
        return [type("IO", (), {"name": self.output_name})()]


@dataclass
class SequenceModnetSession:
    """Returns a per-call alpha scalar sequence (for temporal tests)."""

    values: list[float]
    input_name: str = "input"
    output_name: str = "output"
    _index: int = 0

    def run(self, output_names: list[str], input_feed: dict[str, Any]) -> list[np.ndarray]:
        tensor = input_feed[self.input_name]
        _, _, height, width = tensor.shape
        if self._index >= len(self.values):
            value = self.values[-1]
        else:
            value = self.values[self._index]
        self._index += 1
        alpha = np.full((1, 1, height, width), value, dtype=np.float32)
        return [alpha]
