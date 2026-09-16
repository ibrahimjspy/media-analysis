# Native media contract and caller handoff

`POST /analyze` still processes one authorized source. Public job IDs, storage ownership,
transcription, collection orchestration and editing remain caller responsibilities.
No application database migration is needed.

## Request branches

Omitted `mediaKind` and explicit `"video"` retain the video schema-v1 result and identical
legacy request digests. Image and audio results use schemaVersion 2 with `mediaKind` and
a typed `canonicalMedia.kind`. Input types live in `schemas.py`; native output branches
live in `native_schemas.py`. Unknown kinds, incompatible features and video-only metadata
on native requests fail with `INVALID_REQUEST`.

| Kind | Features | Canonical grant |
|---|---|---|
| image | quality, exposure, subjects, faces, ocr, saliency, focus, thumbnails | canonicalImage, image/png |
| audio | audio, waveform, rhythm | canonicalAudio, audio/wav |
| video | existing features plus rhythm | canonicalMp4, video/mp4 |

Image/audio requests require `source.sha256` (64 hexadecimal characters); downloaded and
cached source bytes are verified. Signed URLs and expiry values remain excluded from
request identity. Kind, image/rhythm options and stable grant roles affect identity.
Canonical grants require `canonicalize: true` for native media. Metadata-only requests
need no canonical grant. Image thumbnails require exactly one grant with `index: 0`.

```json
{
  "idempotencyKey": "photo-analysis-1",
  "mediaKind": "image",
  "features": ["quality", "exposure", "saliency", "focus"],
  "source": {
    "signedGetUrl": "https://media.example/photo",
    "expiresAt": "2099-01-01T00:00:00Z",
    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "imageOptions": {"maxDimension": 1280, "alphaBackground": "white"}
}
```

The example hash is a placeholder; use the actual source byte digest. For an audio
request use `mediaKind: "audio"`, features such as `["audio", "waveform", "rhythm"]`,
and optionally `rhythmOptions: {"minBpm": 50, "maxBpm": 200}`. Rhythm options also apply
to video, including rhythm-only requests with canonicalization disabled.

## Image appearance and coordinates

Supported decoders are JPEG, PNG, WebP and HEIF/HEIC with up to 8-bit channels.
Higher-depth integer/float raster modes are rejected rather than implicitly clipped.
Animated/multiple images are
rejected. HEIF accepts 8-bit SDR sRGB or a valid ICC profile; HDR/non-sRGB NCLX without
ICC is explicitly rejected. A tested SDR HEIF decoder does not establish support for
all phone capture modes. Invalid color profiles fail decoding rather than silently
changing appearance.

Canonical images are full-resolution opaque RGB PNGs, EXIF oriented, converted from
ICC to sRGB when tagged (untagged RGB assumes sRGB), and composited over the requested
white/black background in sRGB. Preview derivatives and measurements use that same
appearance. Original alpha is not retained in the canonical derivative. Source bytes
remain caller-owned. `canonicalMedia` records source format, source hash, original and
canonical dimensions, orientation, normalized original-to-canonical transform, analysis
size, analysis-to-canonical scale and the pinned canonicalization policy.

For HEIF, libheif supplies a container-normalized decoded raster. `originalWidth`,
`originalHeight` and the transform refer to that raster, not the encoded HEVC tile grid.
`decoderOrientationPolicy` records this distinction; `decoderReportedOrientation`
retains the plugin's original EXIF value without applying it twice.

The default source bound is 40 million decoded pixels, independent of video's 1920-pixel
source-dimension limit. Analysis defaults to at most 1280 pixels on the longest side.
`analysisResolution` must preserve aspect, avoid upscaling and fit the configured working
bound. Regions are normalized against oriented canonical pixels. No image output invents
fps, duration, shots, source frames or temporal track identities.

| Body | Meaning |
|---|---|
| quality | Raw grayscale Laplacian variance, dark/bright clip fractions, thresholds and working dimensions |
| exposure | Linear sRGB/BT709 luma mean, standard deviation, percentiles and mean encoded sRGB color |
| subjects, faces | `regions` containing canonical normalized boxes and raw model scores |
| ocr | `regions` containing canonical normalized polygons and raw scores; no recognized text |
| saliency | Versioned Lab contrast heuristic, up to eight connected regions, contrast scores and validity |
| focus | Inferred contrast-region centroids and boxes, tied to saliency evidence; no center fallback |
| thumbnails | One JPEG artifact with index, dimensions, MIME, checksum and byte count |

Saliency uses no additional model weights. It is explicitly heuristic, not a probability
of attention. Its exact implementation is `lab-global-local-contrast-v1`; focus is
`saliency-region-centroid-v1`. Uniform images yield empty candidates with
`insufficient-contrast`. Neither output selects a crop, pan, transition or effect.

## Audio clocks and measurements

The source probe rejects a real video timeline in an audio request; attached cover art
is allowed. Exactly one audio stream is supported. Supported codec names and configured
duration/channel/sample-rate bounds are exposed by `/capabilities`.

The default source envelope is 600 seconds, eight channels and 192 kHz. Analysis uses
mono 16 kHz float PCM; VAD settings remain unchanged. Decode output is bounded even if
container duration metadata is incorrect. Optional canonical WAV uses PCM s16 at the
original rate/channel count. It preserves stereo playback rather than publishing mono
analysis PCM. Canonical output size is checked before encoding.

`clock.origin` is the first decoded playback sample, after FFmpeg applies container
skip-sample/delay/padding metadata. `sourceStartTimeSec`/`sourcePtsOriginSec` retain
container timestamps. Analysis sample indices use `analysisSampleRate`; timestamps use
seconds. `durationSec` is decoded PCM duration, while `sourceDurationSec` retains the
probe value. Neither is derived from invented video fps. Raw ADTS or other containers
without gapless metadata retain whatever delay exists in actual decoded playback;
there is no promise to recover an encoder's unknown original recording origin.

`audio` reuses RMS, onsets, VAD, loudness and true peak. `signalState` distinguishes
measured silence from present signal. `speechState` distinguishes
`measured`, `reference` and `unknown`; absent VAD makes audio partial with
`VAD_UNAVAILABLE`, not verified no-speech. Stub VAD is test/reference evidence only.
Existing video audio fields retain their previous shapes and source-frame meanings.

`rhythm` includes `onsetCandidates`, timestamped `beats`, local `segments`, optional
aggregate BPM, sample rate, hop, algorithm version, evidence tier and reasons. The
`rms-attack-periodic-runs-v1` policy accepts runs of at least four acoustic attacks with
consistent intervals. Missing beats are not filled; tempo changes can produce separate
segments. Periodic attacks are not verified musical downbeats, phrases or semantic
impacts. Silence, absence, insufficient periodic evidence and irregular/ambient outcomes
remain distinct. Musical alignment is not production-qualified.

## Availability, errors and delivery

`GET /capabilities` is a private-network read-only endpoint; `/ready.capabilities` uses
the same response. Each kind/feature reports implemented, configured, model-ready,
available, reference-mode and production-qualified states. Input policies, canonical
artifact MIME types and algorithm identities are explicit. Existing readiness fields
remain present. Production quality is not certified by this endpoint.

`MEDIA_ANALYSIS_WORKER_ROLE=audio` avoids visual model dependencies.
`MEDIA_ANALYSIS_ENABLED_FEATURES` is an optional JSON list restricting configured
features and model loading, for example `["rhythm","waveform"]` needs no model files.
With `audio` enabled, the role loads Silero through the manifest/warmup path. Existing
combined/general/OCR/matte roles retain their defaults.

Unrequested bodies are omitted. Completed empty detections remain completed with empty
regions; unavailable models omit the feature body and report `unavailable`; inference
exceptions report `failed`. Native stub visual models are unavailable, not fake successful
empty detections. Overall status is completed, partial or failed. Existing error codes
cover invalid requests, decode failures, limits, checksum mismatch, cancellation,
timeouts and upload failures.

Renewed output grants always execute delivery again. Native measurement JSON is cached
before upload; images/audio are decoded again to enforce current limits and derivatives
are regenerated before delivery. Renewed grants do not repeat cached successful feature
inference. Missing local derivative files therefore do not imply a fictitious upload.
Video thumbnail/matte requests retain the artifact-aware recomputation path. Canonical
video requests can reuse existing measurements/canonical media and deliver to the current
grant. Failed delivery leaves cached measurements reusable by retry.

The measurement cache now uses namespace/version changes plus source, request, decoder,
model, runtime-build and analyzer/policy identities. Old measurement entries invalidate
safely; legacy request digests remain unchanged. No frontend editing or template state
enters raw measurement keys. In-memory jobs remain process-local; caller persistence
and retry are required for cross-worker coordination/recovery.

Native telemetry reports kind-prefixed stages, download/upload bytes, decoded pixels or
analysis sample count, and cache hits. `processPeakRssBytes` reports the process lifetime high-water RSS
(not memory attributable to a single request). `/ready.runtimeInitializationMs` measures
model initialization/warmup separately from warm-request stage timings.

## Verification boundary

Deterministic tests cover orientation, color/alpha, formats, bounds, schema/hash rules,
quality behavior, synthetic beat timing, all advertised audio codecs, renewed delivery,
restart retries and current-limit/checksum enforcement on cache hits. Existing video,
VFR/rotation, audio, cancellation and reference-matte suites remain in place.

These checks do not approve real-image saliency usefulness, real-music beat accuracy,
real speech VAD, hair/hands/motion matte quality, Linux production parity, or frontend
preview/export. Caller parser fixtures and rights-cleared real-media evaluation were
not supplied in this workspace. Those integration/quality gates remain open, as do the
repository's pre-existing OCR/matte production gates. Do not label this implementation
as production-qualified based on stub or synthetic tests.
