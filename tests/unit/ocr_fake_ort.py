"""Fake ONNX Runtime sessions for OCR unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FakeIoBinding:
    name: str


class FakeOrtSession:
    """Minimal ORT session stand-in for dependency injection in tests."""

    def __init__(
        self,
        output: Any,
        *,
        input_name: str = "x",
        output_name: str = "sigmoid_0.tmp_0",
        fail_on_run: bool = False,
    ) -> None:
        self._output = output
        self._input_name = input_name
        self._output_name = output_name
        self._fail_on_run = fail_on_run
        self.closed = False

    def run(self, output_names: list[str], input_feed: dict[str, Any]) -> list[Any]:
        if self._fail_on_run:
            raise RuntimeError("inference failed")
        if self._input_name not in input_feed:
            raise KeyError(self._input_name)
        return [self._output]

    def get_inputs(self) -> list[FakeIoBinding]:
        return [FakeIoBinding(self._input_name)]

    def get_outputs(self) -> list[FakeIoBinding]:
        return [FakeIoBinding(self._output_name)]

    def close(self) -> None:
        self.closed = True
