# Worker branch fixtures

These fixtures are generated from local synthetic image/audio/video sources through
`POST /analyze`. They contain no real property media, credentials or signed grants.
Source URLs are nonfunctional examples. Environment-specific telemetry/build metadata
is omitted. The video fixture records the retained schema-v1 shape; exact pre-change
request identity is pinned independently in `test_native_media.py`.

Image/audio fixture validation uses the actual native result types. They are worker
contracts, not evidence that an external caller's existing parser has been executed.
Caller IDs, public result versions and authorized delivery URLs must be added by the
caller; they are not invented here.
