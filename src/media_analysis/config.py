from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
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
    }
)
MATTE_FEATURES = frozenset({"person_matte"})
ALL_FEATURES = CPU_FEATURES | MATTE_FEATURES
V1_FEATURES = frozenset({"subjects", "faces", "ocr", "shots"})
IMPLEMENTED_CPU = CPU_FEATURES
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
    media_analysis_allow_stub_models: bool = False
    media_analysis_download_timeout_sec: float = 30
    media_analysis_job_timeout_sec: float = 240
    media_analysis_stage_timeout_sec: float = 60

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return frozenset(
            host.strip().lower()
            for host in self.media_analysis_allowed_hosts.split(",")
            if host.strip()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    get_settings.cache_clear()
