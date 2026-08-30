"""MODNet model provenance and production gate reporting."""

from __future__ import annotations

from typing import Any

from media_analysis.models_manifest import ModelEntry
from media_analysis.person_matte.modnet import preprocessing_version
from media_analysis.person_matte.temporal import temporal_policy_version
from media_analysis.tools.model_lock import ModelLock, ProfileGates


def matte_production_gate_open(gates: ProfileGates | None) -> bool:
    if gates is None:
        return False
    return bool(gates.matte_production_enabled and gates.benchmark_gate_passed)


def modnet_model_provenance(
    entry: ModelEntry | None,
    *,
    runtime: str = "onnxruntime",
) -> dict[str, Any] | None:
    if entry is None:
        return None
    return {
        "name": entry.name,
        "version": entry.version,
        "artifactSha256": entry.sha256,
        "runtimeProvider": runtime,
        "preprocessingVersion": preprocessing_version(),
        "license": entry.license,
    }


def matte_provenance_block(
    *,
    modnet_entry: ModelEntry | None,
    lock: ModelLock | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "matteTemporalPolicyVersion": temporal_policy_version(),
    }
    model = modnet_model_provenance(modnet_entry)
    if model is not None:
        block["matteModel"] = model
    if lock is not None and lock.gates is not None:
        block["matteProductionGateOpen"] = matte_production_gate_open(lock.gates)
        if lock.gates.note:
            block["matteProductionGateNote"] = lock.gates.note
    return block
