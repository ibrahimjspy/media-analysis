# Benchmarks

Local baseline files are generated and must not be committed:

```bash
python -m media_analysis.tools.benchmark --out benchmarks/local-latest.json
```

With a video, the harness compares two identical sampled-frame passes using a
memory-only cache against memory plus bounded disk spill:

```bash
python -m media_analysis.tools.benchmark --video clip.mp4 --samples 5 --stride 6
```

It records machine/build identity, decode counts, cache hits, disk usage, and
p50/p95 second-pass latency. Use a clip with more than 32 sampled frames to
exercise memory eviction. Without `--video` it remains a timer sanity check.
This is a decoder/cache microbenchmark; measure `/analyze` wall time and peak RSS
on representative real media and GPU hardware before publishing end-to-end SLOs
or flipping `benchmarkGatePassed`.
