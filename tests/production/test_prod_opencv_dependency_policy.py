from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution, metadata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


@pytest.mark.production
@pytest.mark.unit
def test_prod_pyproject_pins_scenedetect_below_07() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "scenedetect>=" in text
    assert ",<0.7" in text or "<0.7" in text


def _runtime_requires_dist(package: str) -> list[str]:
    return [
        requirement
        for requirement in metadata(package).get_all("Requires-Dist") or []
        if "extra ==" not in requirement
    ]


@pytest.mark.production
@pytest.mark.unit
def test_prod_installed_scenedetect_does_not_require_opencv_python() -> None:
    try:
        requirements = _runtime_requires_dist("scenedetect")
    except PackageNotFoundError:
        pytest.skip("scenedetect not installed in this environment")
    assert not any(req.startswith("opencv-python") for req in requirements)


@pytest.mark.production
@pytest.mark.unit
def test_prod_opencv_python_not_installed_when_headless_is() -> None:
    try:
        distribution("opencv-python-headless")
    except PackageNotFoundError:
        pytest.skip("opencv-python-headless not installed")
    with pytest.raises(PackageNotFoundError):
        distribution("opencv-python")
