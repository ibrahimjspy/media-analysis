"""Additive schema-v2 result branches. Video retains its schema-v1 wire contract."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class CanonicalArtifact(BaseModel):
    model_config = {"extra": "forbid"}
    sourceSha256: str
    sha256: str | None = None
    byteCount: int | None = None
    mimeType: str | None = None
    canonicalizationVersion: str


class CanonicalImage(CanonicalArtifact):
    kind: Literal["image"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    aspectRatio: float = Field(gt=0)
    sourceFormat: str
    sourceMimeType: str
    originalWidth: int = Field(gt=0)
    originalHeight: int = Field(gt=0)
    exifOrientation: int
    originalToCanonicalNormalized: list[list[float]]
    colorSpace: str
    colorPolicy: str
    alphaPolicy: str
    canonicalizationVersion: str
    analysisWidth: int = Field(gt=0)
    analysisHeight: int = Field(gt=0)
    analysisToCanonicalScale: dict[str, float]
    decoderOrientationPolicy: str
    decoderReportedOrientation: int | None = None


class AudioClock(BaseModel):
    origin: Literal["first-decoded-playback-sample"]
    analysisToPlaybackOffsetSec: float
    decoderDelayPolicy: str
    sourcePtsOriginSec: float


class CanonicalAudio(CanonicalArtifact):
    kind: Literal["audio"]
    durationSec: float = Field(gt=0)
    sourceDurationSec: float = Field(gt=0)
    codec: str
    sampleRate: int = Field(gt=0)
    channels: int = Field(gt=0)
    sourceCodec: str
    sourceStartTimeSec: float
    analysisSampleRate: int = Field(gt=0)
    analysisSampleCount: int = Field(gt=0)
    analysisChannels: Literal[1]
    clock: AudioClock


class CapabilityOutcome(BaseModel):
    status: Literal["completed", "unavailable", "partial", "failed"]
    warningCodes: list[str] | None = None


class NativeEnvelope(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[2]
    overallStatus: Literal["completed", "partial", "failed"]
    requestedFeatures: list[str]
    capabilities: dict[str, CapabilityOutcome]
    provenance: dict[str, Any]
    warningCodes: list[str]
    telemetry: dict[str, Any] | None = None


class MeasurementModel(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}


class Point(MeasurementModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class Box(Point):
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class ImageQuality(MeasurementModel):
    laplacianVariance: float = Field(ge=0)
    darkClipFraction: float = Field(ge=0, le=1)
    brightClipFraction: float = Field(ge=0, le=1)
    clipPolicy: str
    scoreType: Literal["measurement"]
    workingWidth: int = Field(gt=0)
    workingHeight: int = Field(gt=0)
    policyVersion: str


class ImageExposure(MeasurementModel):
    meanLuma: float
    lumaStdDev: float
    lumaPercentiles: list[float]
    meanSrgb: list[float]
    lumaColorSpace: str
    policyVersion: str


class DetectionRegion(MeasurementModel):
    type: Literal["person"] | None = None
    box: Box | None = None
    polygon: list[Point] | None = None
    score: float
    scoreType: Literal["raw_model"]


class ImageRegions(MeasurementModel):
    regions: list[DetectionRegion]


class SaliencyRegion(MeasurementModel):
    box: Box
    point: Point
    score: float = Field(ge=0, le=1)
    contrastMass: float = Field(gt=0)


class Saliency(MeasurementModel):
    regions: list[SaliencyRegion]
    algorithmVersion: str
    model: None
    scoreType: Literal["heuristic"]
    scoreMeaning: str
    validity: Literal["contrast-evidence", "insufficient-contrast"]
    productionQualified: Literal[False]


class FocusEvidence(MeasurementModel):
    feature: Literal["saliency"]
    regionIndex: int = Field(ge=0)


class FocusCandidate(MeasurementModel):
    point: Point
    box: Box
    evidence: FocusEvidence
    selection: Literal["inferred"]


class Focus(MeasurementModel):
    candidates: list[FocusCandidate]
    policyVersion: str
    evidenceAlgorithmVersion: str
    validity: Literal["contrast-evidence", "insufficient-contrast"]


class ImageThumbnail(MeasurementModel):
    index: Literal[0]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str
    byteCount: int = Field(gt=0)
    mimeType: Literal["image/jpeg"]


class AcousticEvent(MeasurementModel):
    timeSec: float = Field(ge=0)
    sampleIndex: int = Field(ge=0)
    strength: float = Field(ge=0)
    scoreType: Literal["measurement"]


class RhythmSegment(MeasurementModel):
    startSec: float = Field(ge=0)
    endSec: float = Field(gt=0)
    bpm: float = Field(gt=0)


class NeuralBeatEvent(MeasurementModel):
    """Model time and rounded coordinate; no confidence or strength is implied."""

    timeSec: float = Field(ge=0, allow_inf_nan=False)
    sampleIndex: int = Field(ge=0)


class NeuralBeatDiagnostics(MeasurementModel):
    """Downbeats are diagnostic only and cannot be selected as anchors."""

    downbeatTimesSec: list[float]
    downbeatsQualified: Literal[False] = False


class NeuralRhythm(MeasurementModel):
    """Separate opt-in evidence; failed inference is never successful silence."""

    status: Literal["completed", "unavailable", "failed", "cancelled"]
    algorithmVersion: Literal["beat-this-small0-v1"]
    checkpointSha256: Literal["6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b"]
    sampleRate: Literal[22050]
    frameHopSec: Literal[0.02]
    timestampOrigin: Literal["decoded-playback-start"]
    evidenceKind: Literal["model-estimate"]
    beats: list[NeuralBeatEvent]
    warningCodes: list[str]
    productionQualified: Literal[False]
    diagnostics: NeuralBeatDiagnostics | None = None


class Rhythm(MeasurementModel):
    neural: NeuralRhythm | None = None
    beats: list[AcousticEvent]
    onsetCandidates: list[AcousticEvent]
    bpm: float | None
    segments: list[RhythmSegment]
    algorithmVersion: str
    sampleRate: int = Field(gt=0)
    hopSec: float = Field(gt=0)
    timestampOrigin: Literal["decoded-playback-start"]
    evidenceTier: Literal["none", "periodic-acoustic-attacks"]
    reasons: list[str]
    productionQualified: Literal[False]


class Waveform(MeasurementModel):
    hopSec: float = Field(gt=0)
    min: list[float]
    max: list[float]


class Rms(MeasurementModel):
    hopSec: float = Field(gt=0)
    values: list[float]


class Onset(MeasurementModel):
    sec: float = Field(ge=0)
    strength: float = Field(ge=0)
    band: str | None = None


class SpeechInterval(MeasurementModel):
    startSec: float = Field(ge=0)
    endSecExclusive: float = Field(gt=0)
    score: float
    scoreType: Literal["raw_model"]


class AudioMeasurements(MeasurementModel):
    durationSec: float = Field(gt=0)
    rms: Rms
    onsets: list[Onset]
    speech: list[SpeechInterval]
    integratedLufs: float | None
    truePeakDb: float | None
    bpm: float | None
    signalState: Literal["present", "silence"]
    speechState: Literal["measured", "reference", "unknown"]


class ImageResult(NativeEnvelope):
    mediaKind: Literal["image"]
    canonicalMedia: CanonicalImage
    quality: ImageQuality | None = None
    exposure: ImageExposure | None = None
    subjects: ImageRegions | None = None
    faces: ImageRegions | None = None
    ocr: ImageRegions | None = None
    saliency: Saliency | None = None
    focus: Focus | None = None
    visual_regions: dict[str, Any] | None = None
    thumbnails: list[ImageThumbnail] | None = None


class AudioResult(NativeEnvelope):
    mediaKind: Literal["audio"]
    canonicalMedia: CanonicalAudio
    audio: AudioMeasurements | None = None
    waveform: Waveform | None = None
    rhythm: Rhythm | None = None


NativeResult = Annotated[ImageResult | AudioResult, Field(discriminator="mediaKind")]
