# Golden fixtures

This public repository does not commit third-party video. Tests generate
redistribution-safe clips at runtime with ffmpeg:

| Clip | What it covers |
|---|---|
| people shapes | Upright moving boxes as a people-timeline stand-in |
| face circles | Face-sized blobs without using real faces |
| OCR text | Burned-in Latin text, or `testsrc2` timestamps if no font is installed |
| speech tone | 440 Hz tone so audio/VAD paths see real samples |
| VFR concat | 15 fps + 30 fps segments copied into one source |
| rotation | Display-matrix 90° metadata on a landscape encode |

Do not add a developer `Downloads` file. If a rights-cleared CC0 reel is added
later, pin it by SHA-256 in a fixture manifest and keep generation as the
default CI path.
