from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from tests.unit.person_matte.fake_ort import FakeModnetSession, SequenceModnetSession

from media_analysis.errors import CANCELLED, UPLOAD_FAILED, AnalyzeError
from media_analysis.frames import Rational
from media_analysis.models_manifest import ModelEntry
from media_analysis.person_matte.encoding import probe_matte_mp4
from media_analysis.person_matte.pipeline import _iter_refined_gray_frames, run_person_matte_stage1
from media_analysis.person_matte.readiness import (
    matte_worker_ready_for_production,
    matte_worker_ready_for_serving,
)
from media_analysis.person_matte.temporal import TemporalMatteState
from media_analysis.person_matte.types import CanonicalMediaSpec, MatteOutputGrant, ShotBoundary
from media_analysis.tools.model_lock import ProfileGates


def _shots(frame_count: int) -> dict:
    return {"shots": [{"startFrame": 0, "endFrameExclusive": frame_count}]}


def _grants() -> dict:
    return {
        "matteAssets": [
            {
                "label": "full",
                "signedPutUrl": "https://example.test/matte",
                "expiresAt": "2099-01-01T00:00:00Z",
            }
        ]
    }


def _canonical(*, frame_count: int = 6, width: int = 64, height: int = 64) -> CanonicalMediaSpec:
    return CanonicalMediaSpec(
        fps=Rational(30, 1),
        time_base=Rational(1, 30),
        frame_count=frame_count,
        width=width,
        height=height,
    )


def _frames(frame_count: int, width: int, height: int, fill: int) -> list[tuple[int, np.ndarray]]:
    frame = np.full((height, width, 3), fill, dtype=np.uint8)
    return [(index, frame.copy()) for index in range(frame_count)]


@pytest.mark.unit
@pytest.mark.ffmpeg
@pytest.mark.parametrize("fill", [0, 128, 255])
def test_pipeline_full_duration_output_range(tmp_path: Path, fill: int) -> None:
    uploads: list[tuple[bytes, str, MatteOutputGrant]] = []

    def put(body: bytes, content_type: str, grant: MatteOutputGrant) -> None:
        uploads.append((body, content_type, grant))

    canonical = _canonical(frame_count=4)
    records = {}
    candidate = run_person_matte_stage1(
        frames=_frames(4, 64, 64, fill),
        canonical=canonical,
        matte_target={"mode": "all_people"},
        output_grants=_grants(),
        prior_facts=_shots(4),
        session=FakeModnetSession(),
        upload=put,
        work_dir=tmp_path,
        record_stage=records.__setitem__,
    )
    assert {
        "matte_inference",
        "matte_refinement",
        "matte_temporal",
        "matte_resize",
        "matte_encode_write",
        "matte_encode_finalize",
    } <= records.keys()
    assert all(value >= 0 for value in records.values())
    assert "matte_flow" not in records  # Cold-start frames use fresh inference.
    assert candidate.matte_resolution == {"width": 64, "height": 64}
    assert len(uploads) == 1
    body, mime, grant = uploads[0]
    assert mime == "video/mp4"
    assert grant.label == "full"
    assert grant.expires_at.startswith("2099")
    assert candidate.delivery["frameCount"] == 4
    assert candidate.delivery["sha256"] == hashlib.sha256(body).hexdigest()
    mp4_path = tmp_path / "out.mp4"
    mp4_path.write_bytes(body)
    probe = probe_matte_mp4(mp4_path)
    assert len(probe["streams"]) == 1
    assert (probe["streams"][0]["width"], probe["streams"][0]["height"]) == (64, 64)


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_pipeline_temporal_reset_at_shot_boundary(tmp_path: Path) -> None:
    uploads: list[tuple[bytes, str, MatteOutputGrant]] = []

    def put(body: bytes, content_type: str, grant: MatteOutputGrant) -> None:
        uploads.append((body, content_type, grant))

    session = SequenceModnetSession([1.0, 1.0, 0.0, 0.0])
    candidate = run_person_matte_stage1(
        frames=_frames(4, 32, 32, 128),
        canonical=_canonical(frame_count=4, width=32, height=32),
        matte_target={"mode": "all_people"},
        output_grants=_grants(),
        prior_facts={
            "shots": [
                {"startFrame": 0, "endFrameExclusive": 2},
                {"startFrame": 2, "endFrameExclusive": 4},
            ]
        },
        session=session,
        upload=put,
        work_dir=tmp_path,
    )
    assert candidate.delivery["frameCount"] == 4
    assert len(uploads) == 1


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_pipeline_upload_receives_full_grant_and_preserves_analyze_error(tmp_path: Path) -> None:
    def failing_put(body: bytes, content_type: str, grant: MatteOutputGrant) -> None:
        raise AnalyzeError(UPLOAD_FAILED, "grant expired")

    with pytest.raises(AnalyzeError) as exc:
        run_person_matte_stage1(
            frames=_frames(2, 32, 32, 128),
            canonical=_canonical(frame_count=2, width=32, height=32),
            matte_target={"mode": "all_people"},
            output_grants=_grants(),
            prior_facts=_shots(2),
            session=FakeModnetSession(),
            upload=failing_put,
            work_dir=tmp_path,
        )
    assert exc.value.code == UPLOAD_FAILED


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_pipeline_checks_cancel_before_upload(tmp_path: Path) -> None:
    uploads: list[MatteOutputGrant] = []

    def put(body: bytes, content_type: str, grant: MatteOutputGrant) -> None:
        uploads.append(grant)

    def cancel_check() -> None:
        if uploads:
            return
        raise AnalyzeError(CANCELLED, "Analysis cancelled")

    with pytest.raises(AnalyzeError) as exc:
        run_person_matte_stage1(
            frames=_frames(2, 32, 32, 128),
            canonical=_canonical(frame_count=2, width=32, height=32),
            matte_target={"mode": "all_people"},
            output_grants=_grants(),
            prior_facts=_shots(2),
            session=FakeModnetSession(),
            upload=put,
            work_dir=tmp_path,
            cancel_check=cancel_check,
        )
    assert exc.value.code == CANCELLED
    assert not uploads


@pytest.mark.unit
def test_pipeline_rejects_out_of_order_frames(tmp_path: Path) -> None:
    zero = np.zeros((32, 32, 3), dtype=np.uint8)
    frames = [(1, zero.copy()), (0, zero.copy())]

    def put(body: bytes, content_type: str, grant: MatteOutputGrant) -> None:
        return

    with pytest.raises(AnalyzeError):
        run_person_matte_stage1(
            frames=frames,
            canonical=_canonical(frame_count=2, width=32, height=32),
            matte_target={"mode": "all_people"},
            output_grants=_grants(),
            prior_facts=_shots(2),
            session=FakeModnetSession(),
            upload=put,
            work_dir=tmp_path,
        )


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_pipeline_infers_keyframes_and_propagates_between_them(tmp_path: Path) -> None:
    session = SequenceModnetSession([0.5, 0.5])

    run_person_matte_stage1(
        frames=_frames(12, 32, 32, 128),
        canonical=_canonical(frame_count=12, width=32, height=32),
        matte_target={"mode": "all_people"},
        output_grants=_grants(),
        prior_facts=_shots(12),
        session=session,
        upload=lambda _body, _content_type, _grant: None,
        work_dir=tmp_path,
        keyframe_interval=3,
    )

    assert session._index == 8  # 0..5 cold, then 6 and 9.


def _raw_frames(frames, session, shots=()):
    return [
        np.frombuffer(frame, np.uint8).reshape(32, 32)
        for frame in _iter_refined_gray_frames(
            frames,
            canonical=_canonical(frame_count=len(frames), width=32, height=32),
            matte_w=32,
            matte_h=32,
            session=session,
            temporal=TemporalMatteState(shots),
            keyframe_interval=3,
            cancel_check=None,
            deadline=None,
        )
    ]


def test_declared_cut_has_immediate_opacity_and_no_six_frame_ramp():
    frames = _frames(16, 32, 32, 128)
    session = SequenceModnetSession([0.0] * 8 + [1.0] * 6)
    output = _raw_frames(frames, session, (ShotBoundary(0, 10), ShotBoundary(10, 16)))
    assert output[9].max() == 0
    for frame in output[10:]:
        assert frame.min() == 255


def test_missing_shot_boundary_is_reset_from_appearance():
    frames = _frames(10, 32, 32, 0) + [
        (i, np.full((32, 32, 3), 255, np.uint8)) for i in range(10, 16)
    ]
    output = _raw_frames(frames, FakeModnetSession())
    assert output[9].max() == 0
    assert output[10].min() == 255
    np.testing.assert_array_equal(output[10], output[15])


def test_fresh_inference_correction_is_not_damped_by_empty_history():
    output = _raw_frames(_frames(12, 32, 32, 128), SequenceModnetSession([0.0] * 7 + [1.0]))
    assert output[8].max() == 0
    assert output[9].min() == 255


def test_odd_canonical_dimensions_fail_instead_of_silent_geometry_change(tmp_path):
    with pytest.raises(AnalyzeError, match="even"):
        run_person_matte_stage1(
            frames=_frames(2, 31, 33, 128),
            canonical=_canonical(frame_count=2, width=31, height=33),
            matte_target={"mode": "all_people"},
            output_grants=_grants(),
            prior_facts=_shots(2),
            session=FakeModnetSession(),
            upload=lambda *args: None,
            work_dir=tmp_path,
        )


@pytest.mark.unit
def test_stub_model_not_production_ready() -> None:
    entry = ModelEntry(
        name="modnet",
        file="modnet.onnx",
        sha256="a" * 64,
        license="Apache-2.0",
        stub=True,
    )
    gates = ProfileGates(
        owned_ppocr_export_recorded=False,
        matte_production_enabled=False,
        benchmark_gate_passed=False,
    )
    ready, errors = matte_worker_ready_for_production(modnet_entry=entry, gates=gates)
    assert ready is False
    assert any("stub" in item for item in errors)


@pytest.mark.unit
def test_stub_serving_requires_reference_mode() -> None:
    entry = ModelEntry(
        name="modnet",
        file="modnet.onnx",
        sha256="a" * 64,
        license="Apache-2.0",
        stub=True,
    )
    gates = ProfileGates(
        owned_ppocr_export_recorded=False,
        matte_production_enabled=False,
        benchmark_gate_passed=False,
    )
    ready, _ = matte_worker_ready_for_serving(
        modnet_entry=entry,
        gates=gates,
        reference_mode=False,
    )
    assert ready is False
    ready_ref, notes = matte_worker_ready_for_serving(
        modnet_entry=entry,
        gates=gates,
        reference_mode=True,
    )
    assert ready_ref is True
    assert notes
