from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from media_analysis.config import ALL_FEATURES
from media_analysis.source import parse_expires_at

_NORM_EPS = 1e-6

FeatureName = Literal[
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
    "person_matte",
]

ScoreType = Literal["raw_model", "heuristic", "calibrated_probability"]


class RationalIn(BaseModel):
    numerator: int
    denominator: int

    @field_validator("denominator")
    @classmethod
    def denominator_nonzero(cls, value: int) -> int:
        if value == 0:
            raise ValueError("denominator must be non-zero")
        return value


class CanonicalMediaIn(BaseModel):
    fps: RationalIn | None = None
    timeBase: RationalIn | None = None
    frameCount: int | None = None
    durationSec: float | None = None
    width: int | None = None
    height: int | None = None
    videoRange: str | None = None

    @model_validator(mode="after")
    def fps_and_timebase_have_positive_numerators(self) -> CanonicalMediaIn:
        for name in ("fps", "timeBase"):
            rational = getattr(self, name)
            if rational is not None and rational.numerator <= 0:
                raise ValueError(f"{name}.numerator must be positive")
        return self


class SourceIn(BaseModel):
    signedGetUrl: str = Field(min_length=1, max_length=8192)
    expiresAt: str = Field(min_length=1, max_length=128)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")

    @field_validator("expiresAt")
    @classmethod
    def expires_at_is_iso(cls, value: str) -> str:
        parse_expires_at(value)
        return value


class SizeIn(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class SignedPutGrantIn(BaseModel):
    signedPutUrl: str = Field(min_length=1, max_length=8192)
    expiresAt: str = Field(min_length=1, max_length=128)

    @field_validator("expiresAt")
    @classmethod
    def expires_at_is_iso(cls, value: str) -> str:
        parse_expires_at(value)
        return value


class ThumbnailOutputGrantIn(SignedPutGrantIn):
    index: int = Field(ge=0)


class MatteAssetOutputGrantIn(SignedPutGrantIn):
    label: str = Field(min_length=1, max_length=128)


class OutputGrantsIn(BaseModel):
    canonicalMp4: SignedPutGrantIn | None = None
    thumbnails: list[ThumbnailOutputGrantIn] | None = None
    matteAssets: list[MatteAssetOutputGrantIn] | None = None

    @model_validator(mode="after")
    def unique_grant_keys(self) -> OutputGrantsIn:
        if self.thumbnails:
            indices = [grant.index for grant in self.thumbnails]
            if len(set(indices)) != len(indices):
                raise ValueError("outputGrants.thumbnails indices must be unique")
        if self.matteAssets:
            labels = [grant.label for grant in self.matteAssets]
            if len(set(labels)) != len(labels):
                raise ValueError("outputGrants.matteAssets labels must be unique")
        return self


class FrameRangeIn(BaseModel):
    startFrame: int = Field(ge=0)
    endFrameExclusive: int

    @model_validator(mode="after")
    def half_open_positive(self) -> FrameRangeIn:
        if self.endFrameExclusive <= self.startFrame:
            raise ValueError("endFrameExclusive must be greater than startFrame")
        return self


class MatteTargetAllPeopleIn(BaseModel):
    mode: Literal["all_people"]


class MatteTargetSubjectTracksIn(BaseModel):
    mode: Literal["subject_tracks"]
    subjectTrackIds: list[str] = Field(min_length=1)

    @field_validator("subjectTrackIds")
    @classmethod
    def nonempty_unique_ids(cls, value: list[str]) -> list[str]:
        cleaned = [track_id.strip() for track_id in value]
        if not cleaned or any(not track_id for track_id in cleaned):
            raise ValueError("subjectTrackIds must contain non-empty IDs")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("subjectTrackIds must be unique")
        return cleaned


MatteTargetIn = Annotated[
    MatteTargetAllPeopleIn | MatteTargetSubjectTracksIn,
    Field(discriminator="mode"),
]


class NormalizedBoxIn(BaseModel):
    x: float
    y: float
    width: float
    height: float

    @model_validator(mode="after")
    def finite_normalized_coordinates(self) -> NormalizedBoxIn:
        for name, value in (
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.x < -_NORM_EPS or self.y < -_NORM_EPS:
            raise ValueError("box origin must be >= 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("box width and height must be > 0")
        if self.x + self.width > 1.0 + _NORM_EPS:
            raise ValueError("box exceeds canonical width")
        if self.y + self.height > 1.0 + _NORM_EPS:
            raise ValueError("box exceeds canonical height")
        return self


class TrackSampleIn(BaseModel):
    sourceFrame: int = Field(ge=0)
    presentationTimeSecApprox: float
    box: NormalizedBoxIn
    sampleKind: Literal["detected", "tracked", "interpolated"]
    detectorScore: float | None = None
    associationScore: float | None = None
    scoreType: ScoreType
    occluded: bool
    uncertainty: float | None = None


class SubjectTrackIn(BaseModel):
    trackId: str = Field(min_length=1)
    type: Literal["person"] = "person"
    aggregateScore: float | None = None
    scoreType: ScoreType | None = None
    samples: list[TrackSampleIn] = Field(min_length=1)


class ShotAnalysisIn(BaseModel):
    startFrame: int = Field(ge=0)
    endFrameExclusive: int
    startSecApprox: float
    endSecApprox: float
    boundaryKind: Literal["hard_cut", "dissolve", "fade", "start", "end"]
    boundaryScore: float | None = None
    boundaryScoreType: ScoreType | None = None
    classification: Literal[
        "talking_head",
        "wide",
        "no_person",
        "multiple_people",
        "other",
    ]
    classificationScore: float | None = None
    scoreType: ScoreType | None = None

    @model_validator(mode="after")
    def half_open_positive(self) -> ShotAnalysisIn:
        if self.endFrameExclusive <= self.startFrame:
            raise ValueError("shot endFrameExclusive must be greater than startFrame")
        return self


class PriorPolicyVersionsIn(BaseModel):
    """Provisional policy pins from a prior analysis run (benchmark-gated)."""

    qualityPolicyVersion: str | None = None
    motionAnalyzerVersion: str | None = None
    exposureAnalyzerVersion: str | None = None
    audioAnalyzerVersion: str | None = None
    samplingPolicyVersion: str | None = None


class PriorFactsIn(BaseModel):
    shots: list[ShotAnalysisIn] | None = None
    subjects: list[SubjectTrackIn] | None = None
    policyVersions: PriorPolicyVersionsIn | None = None

    @model_validator(mode="after")
    def has_canonical_facts(self) -> PriorFactsIn:
        if not self.shots and not self.subjects:
            raise ValueError("priorFacts must include shots and/or subjects")
        return self


class AnalyzeRequest(BaseModel):
    idempotencyKey: str = Field(min_length=1, max_length=256)
    canonicalMedia: CanonicalMediaIn | None = None
    source: SourceIn
    canonicalize: bool = False
    features: list[str] = Field(min_length=1, max_length=len(ALL_FEATURES))
    analysisResolution: SizeIn | None = None
    matteFrameRanges: list[FrameRangeIn] | None = None
    matteTarget: MatteTargetIn | None = None
    outputGrants: OutputGrantsIn | None = None
    priorFacts: PriorFactsIn | None = None

    @field_validator("features")
    @classmethod
    def features_known(cls, value: list[str]) -> list[str]:
        unknown = [name for name in value if name not in ALL_FEATURES]
        if unknown:
            raise ValueError(f"unknown features: {', '.join(unknown)}")
        if len(set(value)) != len(value):
            raise ValueError("features must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_prior_facts_structure(self) -> AnalyzeRequest:
        if self.priorFacts is None:
            return self
        frame_count = self.canonicalMedia.frameCount if self.canonicalMedia else None
        shots = self.priorFacts.shots
        if shots:
            for left, right in zip(shots, shots[1:], strict=False):
                if left.startFrame > right.startFrame:
                    raise ValueError("priorFacts.shots must be sorted by startFrame")
                if left.endFrameExclusive > right.startFrame:
                    raise ValueError("priorFacts.shots must not overlap")
            if frame_count is not None:
                for shot in shots:
                    if shot.endFrameExclusive > frame_count:
                        raise ValueError("priorFacts.shots exceed canonicalMedia.frameCount")
        if self.priorFacts.subjects and frame_count is not None:
            for track in self.priorFacts.subjects:
                frames = [sample.sourceFrame for sample in track.samples]
                if frames != sorted(frames):
                    raise ValueError("priorFacts subject samples must be sorted by sourceFrame")
                if any(frame >= frame_count for frame in frames):
                    raise ValueError("priorFacts subject sample out of canonical frame range")
        return self
