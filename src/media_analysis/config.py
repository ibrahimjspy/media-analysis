from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CPU_FEATURES = frozenset(
    {
        "subjects",
        "faces",
        "ocr",
        "shots",
        "motion",
        "quality",
        "audio",
        "exposure",
        "waveform",
        "thumbnails",
        "focus",
        "saliency",
        "visual_regions",
        "rhythm",
    }
)
MATTE_FEATURES = frozenset({"person_matte"})
ALL_FEATURES = CPU_FEATURES | MATTE_FEATURES
V1_FEATURES = frozenset({"subjects", "faces", "ocr", "shots"})
IMPLEMENTED_CPU = CPU_FEATURES
IMAGE_FEATURES = frozenset(
    {
        "quality",
        "exposure",
        "subjects",
        "faces",
        "ocr",
        "thumbnails",
        "focus",
        "saliency",
        "visual_regions",
    }
)
AUDIO_FEATURES = frozenset({"audio", "waveform", "rhythm"})
VIDEO_FEATURES = ALL_FEATURES - {"focus", "saliency", "visual_regions"}
FEATURES_BY_KIND = {"image": IMAGE_FEATURES, "audio": AUDIO_FEATURES, "video": VIDEO_FEATURES}
# 1920x1920 at 30 fps for the default 60s duration envelope.
DEFAULT_MAX_DECODED_PIXELS = 1920 * 1920 * 1800


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    media_analysis_key: str = Field(default="")
    media_analysis_allowed_hosts: str = Field(default="localhost,127.0.0.1")
    media_analysis_max_duration_sec: float = 60
    media_analysis_max_width: int = 1920
    media_analysis_max_height: int = 1920
    media_analysis_max_bytes: int = 80 * 1024 * 1024
    media_analysis_max_decoded_pixels: int = DEFAULT_MAX_DECODED_PIXELS
    media_analysis_model_dir: Path = Path("./models")
    media_analysis_image: str = "analysis-cpu"
    media_analysis_worker_role: Literal["combined", "general", "ocr", "audio"] = "combined"
    # Default-off soundtrack pilot; weights are loaded only during startup.
    media_analysis_neural_beats_enabled: bool = False
    media_analysis_neural_beats_max_duration_sec: float = Field(default=180, gt=0, le=600)
    media_analysis_visual_enabled: bool = False
    media_analysis_enabled_features: list[str] | None = None
    media_analysis_max_image_pixels: int = Field(default=40_000_000, gt=0)
    media_analysis_image_max_dimension: int = Field(default=1280, ge=64, le=4096)
    media_analysis_max_audio_duration_sec: float = Field(default=600, gt=0)
    media_analysis_max_audio_channels: int = Field(default=8, ge=1, le=32)
    media_analysis_max_audio_sample_rate: int = Field(default=192000, gt=0)
    media_analysis_allow_stub_models: bool = False
    media_analysis_download_timeout_sec: float = 30
    media_analysis_job_timeout_sec: float = 240
    media_analysis_stage_timeout_sec: float = 60
    media_analysis_feature_workers: int = Field(default=3, ge=1, le=8)
    media_analysis_frame_cache_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1024 * 1024,
    )
    media_analysis_motion_max_dimension: int = Field(default=640, ge=64, le=1920)
    media_analysis_frame_spill_bytes: int = Field(default=2 * 1024 * 1024 * 1024, ge=0)
    media_analysis_cache_enabled: bool = True
    media_analysis_cache_dir: Path = Path("/var/tmp/media-analysis/cache")
    media_analysis_cache_max_bytes: int = Field(
        default=2 * 1024 * 1024 * 1024,
        ge=1024 * 1024,
    )
    media_analysis_cache_ttl_sec: float = Field(default=24 * 60 * 60, ge=0)
    media_analysis_matte_keyframe_interval: int = Field(default=3, ge=1, le=30)
    media_analysis_matte_execution_provider: str = "auto"
    media_analysis_matte_inference_threads: int = Field(default=1, ge=1, le=8)
    media_analysis_tensorrt_cache_dir: Path = Path("/var/tmp/media-analysis/tensorrt")

    @field_validator("media_analysis_enabled_features")
    @classmethod
    def valid_configured_features(cls, features):
        if features is not None and (not features or set(features) - ALL_FEATURES):
            raise ValueError("enabled features must be a nonempty list of known features")
        return features

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return frozenset(
            host.strip().lower()
            for host in self.media_analysis_allowed_hosts.split(",")
            if host.strip()
        )

    @property
    def worker_role(self) -> str:
        if self.media_analysis_image.startswith("matte"):
            return "matte"
        return self.media_analysis_worker_role


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    get_settings.cache_clear()
