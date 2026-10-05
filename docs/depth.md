# Optional relative image depth

Disabled by default. `relative_depth` is an image-only feature requiring
canonicalize=true and a signed `outputGrants.depthMap`. It uses checksum-pinned
Depth Anything V2 Small HF weights; it does not infer room measurements or render
an animation. Read `features/depth.py` for the immutable recipe and file hashes.

Install `.[depth]`, then run the setup-only command:

```bash
python -m media_analysis.tools.vendor_depth /models/depth-v1
```

Enable `MEDIA_ANALYSIS_DEPTH_ENABLED=true` on a general/combined worker with
`MEDIA_ANALYSIS_MODEL_DIR=/models`. Startup verifies all three files, loads the
model locally and warms it. Missing/corrupt weights leave optional depth
unavailable while ordinary worker readiness remains independently evaluated.
`/capabilities` exposes the exact recipe, runtime availability and map format.

`docker/depth-cpu.Dockerfile` can layer dependencies/weights over a same-checkout
base that retains other optional features. Pin the deployment image digest and
keep prior images for rollback. Downloads never occur during startup/inference.

The bounded 518px tensor produces a <=512px map on the entire canonical image.
Output is row-major little-endian uint16 inverse-relative depth: far=0,
near=65535, normalized from raw 1st/99th percentiles. The JSON response has
`relative_depth` metadata and SHA/byte count, never inline samples. The binary is
uploaded to the supplied scoped grant using the existing allowlist/expiry rules.
`canonicalSha256` binds it to the exact orientation-corrected PNG.

Inference is serialized per process. Limit worker-process/container concurrency
and benchmark memory on deployment hardware. Deadline/cancellation checks surround
native kernels; an in-flight CPU kernel is not forcibly interrupted. Results after
cancellation cannot publish. The existing job timeout remains authoritative.

Successful private-cache entries retain bounded encoded samples for grant-renewal
replay without inference. They are stripped from public JSON. Corrupt bytes or a
canonical mismatch trigger recomputation. Upload failures retain valid cached
measurements for later retries and mark the optional feature failed. Cancellation
and shared deadline errors still abort the job.

Run `pytest tests/unit/test_depth.py tests/e2e/test_depth_pipeline.py` for numeric,
request, cache, upload-failure and cancellation regressions. Actual pinned-model
inference additionally needs the depth extra and vendored files; four original
property-photo review copies were tested during implementation. No rendered or
cloud deployment acceptance is implied by these tests.

Small model licensing is Apache-2.0; see NOTICE and the pinned upstream model
card. Do not replace it with non-commercial V2 Base/Large/Giant weights.

Review hardening: exactly uniform images are rejected before inference (including
legacy cached predictions), with DEPTH_NO_IMAGE_VARIATION. Cache recipe/model/
geometry/normalization metadata is validated in addition to binary hashes. These
checks do not provide a suitability score for low-texture or reflective scenes.
