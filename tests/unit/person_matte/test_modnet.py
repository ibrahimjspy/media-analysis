from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from tests.unit.person_matte.fake_ort import FakeModnetSession

from media_analysis.errors import INVALID_REQUEST, AnalyzeError
from media_analysis.person_matte.modnet import (
    compute_modnet_input_size,
    infer_modnet_alpha,
    load_modnet_session,
    postprocess_modnet,
    preprocess_modnet,
)


@pytest.mark.unit
def test_modnet_preprocess_normalizes_to_minus1_plus1() -> None:
    frame = np.full((64, 48, 3), 128, dtype=np.uint8)
    tensor, h, w = preprocess_modnet(frame)
    assert h == 64 and w == 48
    assert tensor.shape[0] == 1 and tensor.shape[1] == 3
    assert abs(float(tensor.mean())) < 0.01


@pytest.mark.unit
def test_modnet_postprocess_resizes_in_float_space_before_quantize() -> None:
    output = np.full((1, 1, 32, 32), 0.5, dtype=np.float32)
    alpha = postprocess_modnet(output, orig_height=40, orig_width=60)
    assert alpha.shape == (40, 60)
    assert abs(int(alpha[0, 0]) - 128) <= 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fill", "expected"),
    [(0, 0), (128, 128), (255, 255)],
)
def test_fake_modnet_session_maps_synthetic_patches(fill: int, expected: int) -> None:
    normalized = (fill - 127.5) / 127.5
    tensor = np.full((1, 3, 32, 32), normalized, dtype=np.float32)
    session = FakeModnetSession()
    alpha = postprocess_modnet(
        session.run(["output"], {"input": tensor})[0],
        orig_height=32,
        orig_width=32,
    )
    assert int(alpha[0, 0]) == expected


@pytest.mark.unit
def test_modnet_rejects_extreme_aspect_ratio() -> None:
    with pytest.raises(AnalyzeError) as exc:
        compute_modnet_input_size(2000, 70000)
    assert exc.value.code == INVALID_REQUEST


@pytest.mark.unit
def test_modnet_small_dimension_is_aligned_safely() -> None:
    rw, rh = compute_modnet_input_size(8, 64)
    assert rw >= 32 and rh >= 32
    assert rw <= 2048 and rh <= 2048


@pytest.mark.unit
def test_infer_modnet_alpha_end_to_end() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    alpha = infer_modnet_alpha(FakeModnetSession(), frame)
    assert alpha.shape == (32, 32)
    assert alpha.max() == 0


@pytest.mark.unit
def test_modnet_prefers_tensorrt_with_fp16(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class SessionOptions:
        intra_op_num_threads = 0

    def inference_session(path, *, sess_options, providers):
        captured.update(path=path, options=sess_options, providers=providers)
        return SimpleNamespace(
            get_providers=lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
            disable_fallback=lambda: captured.update(fallback_disabled=True),
        )

    fake_ort = SimpleNamespace(
        SessionOptions=SessionOptions,
        get_available_providers=lambda: [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        InferenceSession=inference_session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    model = tmp_path / "modnet.onnx"
    model.write_bytes(b"model")

    load_modnet_session(
        model,
        execution_provider="tensorrt",
        engine_cache_dir=tmp_path / "trt-cache",
    )

    provider, options = captured["providers"][0]
    assert provider == "TensorrtExecutionProvider"
    assert options["trt_fp16_enable"] is True
    assert options["trt_engine_cache_path"] == str(tmp_path / "trt-cache")
    assert captured["fallback_disabled"] is True


@pytest.mark.unit
@pytest.mark.parametrize("advertised", [False, True])
def test_explicit_tensorrt_rejects_cuda_fallback(tmp_path, monkeypatch, advertised) -> None:
    available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if advertised:
        available.insert(0, "TensorrtExecutionProvider")
    fake_ort = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(),
        get_available_providers=lambda: available,
        InferenceSession=lambda *args, **kwargs: SimpleNamespace(
            get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    model = tmp_path / "modnet.onnx"
    model.write_bytes(b"model")
    with pytest.raises(RuntimeError, match="requested tensorrt"):
        load_modnet_session(
            model,
            execution_provider="tensorrt",
            engine_cache_dir=tmp_path / "engines",
        )


@pytest.mark.unit
def test_requested_gpu_provider_must_be_available(tmp_path, monkeypatch) -> None:
    fake_ort = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(intra_op_num_threads=0),
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    model = tmp_path / "modnet.onnx"
    model.write_bytes(b"model")
    with pytest.raises(RuntimeError, match="unavailable"):
        load_modnet_session(model, execution_provider="cuda")


@pytest.mark.unit
def test_cpu_inference_thread_budget_is_forwarded(tmp_path, monkeypatch):
    captured = {}

    def create(*args, **kwargs):
        captured['threads'] = kwargs['sess_options'].intra_op_num_threads
        return SimpleNamespace(get_providers=lambda: ['CPUExecutionProvider'],
                               disable_fallback=lambda: None)

    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(),
        get_available_providers=lambda: ['CPUExecutionProvider'], InferenceSession=create,
    ))
    model = tmp_path / 'modnet.onnx'
    model.write_bytes(b'model')
    load_modnet_session(model, execution_provider='cpu', inference_threads=2)
    assert captured['threads'] == 2


@pytest.mark.unit
@pytest.mark.parametrize('threads', [0, -1, 9])
def test_invalid_inference_thread_budget_is_rejected(tmp_path, threads):
    model = tmp_path / 'modnet.onnx'
    model.write_bytes(b'model')
    with pytest.raises(ValueError, match='inference_threads'):
        load_modnet_session(model, inference_threads=threads)
