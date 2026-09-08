# Owned MODNet export — evaluation, not production approval

The `matte-cpu` profile now downloads a real photographic MODNet ONNX model,
not the former constant-gray stub. The official checkpoint comes from the
pretrained directory linked by [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet).
Upstream states that the code and models are Apache-2.0 licensed. The export
release includes that license, the source/toolchain lock, exporter, ONNX file,
parity fixtures, and numerical report. No user media is published.

## Reproduction

Use a separate export environment; serving images do not need PyTorch.

```sh
pip install torch==2.5.1 torchvision==0.20.1 onnx==1.17.0 onnxruntime==1.22.0 gdown==5.2.0
git clone https://github.com/ZHKKKe/MODNet.git /tmp/MODNet-export
git -C /tmp/MODNet-export checkout 28165a451e4610c9d77cfdf925a94610bb2810fb
gdown 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz -O /tmp/modnet.ckpt
python -m media_analysis.tools.export_modnet --upstream /tmp/MODNet-export \
  --checkpoint /tmp/modnet.ckpt --out /tmp/modnet.onnx
```

The exporter rejects altered source/checkpoint inputs, loads tensor weights
with `weights_only=True`, exports opset 17 with dynamic spatial dimensions,
and checks three shapes against PyTorch. The recorded export's maximum
absolute error was below 0.00003. These are numerical compatibility checks,
not person-recall or edge-quality scores. Binary hashes can differ across
export platforms; serving always checks the published artifact hash in the lock.

Run the downloaded-artifact checks with:

```sh
pytest tests/production/test_prod_real_modnet.py -m real_models -o addopts='-q --strict-markers'
```

## Known quality failures and open gates

Local visual checks on the user's 35-second test video found real silhouettes
in the street and hallway scenes, replacing the flat-gray mask. However,
the distant, backlit window scene loses the person and includes furniture;
some person-free views also contain spurious foreground. Person-box crop and
official webcam-checkpoint experiments did not consistently solve this case.
Those experiments are not enabled in the serving pipeline.

Therefore `matteProductionEnabled` and `benchmarkGatePassed` remain false.
Real-model runs under the explicit evaluation mode emit `MATTE_EVALUATION_MODE`;
test stubs still emit `MATTE_REFERENCE_MODE`. Neither means production-approved.
Do not enable production behind-subject compositing based on inference success
or an MP4's existence alone. Remaining work includes a stronger validated model
or an effective low-confidence fallback, labeled multi-scene quality evaluation,
temporal edge/ghosting checks, browser decoded-luma/compositing parity, and GPU
latency validation. CPU evaluation is not a GPU benchmark.

The full 1,059-frame test-video run took 220.3 seconds on the local Mac CPU,
including 195.6 seconds of neural inference, 10.6 seconds of optical flow,
and 7.8 seconds of refinement. This is not an EC2 measurement or an accepted SLO.
It confirms that real-model inference, unlike the former stub, is now a major cost.
