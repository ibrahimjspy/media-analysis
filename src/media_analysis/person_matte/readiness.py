"""Matte worker production vs reference readiness gates."""

from __future__ import annotations

import os
from typing import Any

from media_analysis.models_manifest import ModelEntry
from media_analysis.person_matte.provenance import matte_production_gate_open
from media_analysis.tools.model_lock import ProfileGates


def matte_reference_mode_enabled(raw: str | None = None) -> bool:
    value = raw if raw is not None else os.environ.get("MEDIA_ANALYSIS_MATTE_REFERENCE_MODE", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def matte_worker_ready_for_production(
    *,
    modnet_entry: ModelEntry | None,
    gates: ProfileGates | None,
) -> tuple[bool, tuple[str, ...]]:
    """Default production worker must not report inference-ready on stub/closed gates."""
    errors: list[str] = []
    if modnet_entry is None:
        errors.append("missing modnet manifest entry")
    elif modnet_entry.stub:
        errors.append("modnet artifact is stub/reference-only")
    if not matte_production_gate_open(gates):
        errors.append("matte production gate is closed")
    return (not errors, tuple(errors))


def matte_worker_ready_for_serving(
    *,
    modnet_entry: ModelEntry | None,
    gates: ProfileGates | None,
    reference_mode: bool | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Reference/test mode may serve stub pipeline; production mode never does."""
    if reference_mode is None:
        reference_mode = matte_reference_mode_enabled()
    production_ready, production_errors = matte_worker_ready_for_production(
        modnet_entry=modnet_entry,
        gates=gates,
    )
    if production_ready:
        return True, ()
    if reference_mode:
        return True, ("reference mode enabled; not production-ready",)
    return False, production_errors


def readiness_report(
    *,
    modnet_entry: ModelEntry | None,
    gates: ProfileGates | None,
    reference_mode: bool | None = None,
) -> dict[str, Any]:
    if reference_mode is None:
        reference_mode = matte_reference_mode_enabled()
    production_ready, production_errors = matte_worker_ready_for_production(
        modnet_entry=modnet_entry,
        gates=gates,
    )
    serving_ready, serving_errors = matte_worker_ready_for_serving(
        modnet_entry=modnet_entry,
        gates=gates,
        reference_mode=reference_mode,
    )
    return {
        "productionInferenceReady": production_ready,
        "servingReady": serving_ready,
        "referenceMode": reference_mode,
        "productionBlockers": list(production_errors),
        "servingNotes": list(serving_errors),
    }
