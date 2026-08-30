from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from media_analysis.app import create_app, reset_manifest
from media_analysis.config import Settings, reset_settings
from media_analysis.jobs import registry
from media_analysis.models_manifest import sha256_file

TEST_KEY = "test-media-analysis-key"


def write_stub_models(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "yolox_tiny.onnx": b"stub-yolox-tiny",
        "face_detection_yunet_2023mar.onnx": b"stub-yunet",
        "PP-OCRv5_mobile_det.onnx": b"stub-ocr-det",
        "silero_vad.onnx": b"stub-silero-vad",
    }
    models = []
    for name, filename, blob in (
        ("yolox-tiny", "yolox_tiny.onnx", files["yolox_tiny.onnx"]),
        ("yunet", "face_detection_yunet_2023mar.onnx", files["face_detection_yunet_2023mar.onnx"]),
        ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_det.onnx", files["PP-OCRv5_mobile_det.onnx"]),
        ("silero-vad", "silero_vad.onnx", files["silero_vad.onnx"]),
    ):
        path = model_dir / filename
        path.write_bytes(blob)
        models.append(
            {
                "name": name,
                "file": filename,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "license": "Apache-2.0" if "yunet" not in name and "silero" not in name else "MIT",
                "version": "stub",
                "stub": True,
            }
        )
    (model_dir / "manifest.json").write_text(json.dumps({"models": models}), encoding="utf-8")


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "models"
    write_stub_models(directory)
    return directory


@pytest.fixture
def settings(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("MEDIA_ANALYSIS_KEY", TEST_KEY)
    monkeypatch.setenv("MEDIA_ANALYSIS_ALLOWED_HOSTS", "127.0.0.1,localhost")
    monkeypatch.setenv("MEDIA_ANALYSIS_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("MEDIA_ANALYSIS_IMAGE", "analysis-cpu")
    monkeypatch.setenv("MEDIA_ANALYSIS_ALLOW_STUB_MODELS", "true")
    monkeypatch.setenv("MEDIA_ANALYSIS_MAX_DURATION_SEC", "60")
    monkeypatch.setenv("MEDIA_ANALYSIS_MAX_BYTES", "80000000")
    reset_settings()
    reset_manifest()
    registry.clear()
    return Settings()


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"X-Media-Analysis-Key": TEST_KEY}


def make_color_mp4(
    path: Path,
    *,
    seconds: float = 1.0,
    fps: int = 30,
    size: str = "320x240",
) -> Path:
    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:s={size}:d={seconds}:r={fps}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


class _FileHandler(BaseHTTPRequestHandler):
    file_path: Path
    last_path: str = ""

    def do_GET(self) -> None:  # noqa: N802
        type(self).last_path = self.path
        data = self.file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: object) -> None:
        return


@pytest.fixture
def source_server(tmp_path: Path):
    video = make_color_mp4(tmp_path / "clip.mp4")

    class Handler(_FileHandler):
        file_path = video

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield {
        "url": f"http://{host}:{port}/clip.mp4?X-Amz-Signature=fake",
        "path": video,
        "sha256": sha256_file(video),
        "handler": Handler,
    }
    server.shutdown()
    thread.join(timeout=2)


@pytest.fixture
def future_expiry() -> str:
    return "2099-01-01T00:00:00Z"
