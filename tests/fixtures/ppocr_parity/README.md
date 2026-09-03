# PP-OCR Paddle/ONNX parity reference

`reference.npz` is created by a linux/amd64 owned export and is not committed
until that export is recorded. The file must contain:

- `input`: float32 NCHW tensor used for both Paddle and ONNX
- `paddle_prob`: Paddle DB probability map for that tensor

When `gates.ownedPpOcrExportRecorded` is true, CI requires this file and the
vendored ONNX must stay within `models/ppocr-export.lock.json` parity limits.
