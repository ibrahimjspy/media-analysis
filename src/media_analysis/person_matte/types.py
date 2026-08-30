"""Person matte Stage 1 contracts (worker-side candidate, not caller manifest)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from media_analysis.frames import Rational


@dataclass(frozen=True, slots=True)
class ShotBoundary:
    start_frame: int
    end_frame_exclusive: int


@dataclass(frozen=True, slots=True)
class CanonicalMediaSpec:
    fps: Rational
    time_base: Rational
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class MatteOutputGrant:
    label: str
    signed_put_url: str
    expires_at: str


@dataclass(frozen=True, slots=True)
class MatteFrame:
    source_frame: int
    alpha: Any  # H×W uint8 grayscale matte at matte resolution


@dataclass(frozen=True, slots=True)
class MatteUploadResult:
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class PersonMatteCandidate:
    type: Literal["person_matte_v1"]
    matte_target: dict[str, Any]
    temporal_policy_version: str
    reset_policy: Literal["shot_boundaries"]
    matte_resolution: dict[str, int]
    source_fps: dict[str, int]
    source_time_base: dict[str, int]
    source_frame_count: int
    coverage_ranges: list[dict[str, int]]
    coverage_fraction: float
    encoding: dict[str, Any]
    delivery: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "matteTarget": self.matte_target,
            "temporalPolicyVersion": self.temporal_policy_version,
            "resetPolicy": self.reset_policy,
            "matteResolution": self.matte_resolution,
            "sourceFps": self.source_fps,
            "sourceTimeBase": self.source_time_base,
            "sourceFrameCount": self.source_frame_count,
            "coverageRanges": self.coverage_ranges,
            "coverageFraction": self.coverage_fraction,
            "encoding": self.encoding,
            "delivery": self.delivery,
        }
