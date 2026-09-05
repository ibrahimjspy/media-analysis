from __future__ import annotations

import pytest

from media_analysis.analyze import validate_features
from media_analysis.config import Settings
from media_analysis.errors import FEATURE_UNAVAILABLE, AnalyzeError


@pytest.mark.unit
def test_general_worker_rejects_ocr() -> None:
    settings = Settings(media_analysis_worker_role="general")
    with pytest.raises(AnalyzeError) as exc:
        validate_features(["ocr"], settings)
    assert exc.value.code == FEATURE_UNAVAILABLE


@pytest.mark.unit
def test_ocr_worker_rejects_general_features() -> None:
    settings = Settings(media_analysis_worker_role="ocr")
    with pytest.raises(AnalyzeError) as exc:
        validate_features(["motion"], settings)
    assert exc.value.code == FEATURE_UNAVAILABLE
    validate_features(["ocr", "shots"], settings)
