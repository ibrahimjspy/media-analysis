# Benchmarks

Local baseline files are generated and must not be committed:

```bash
python -m media_analysis.tools.benchmark --out benchmarks/local-latest.json
```

The harness currently records machine identity, FFmpeg/runtime provenance, and a
timer sanity check. Replace the samples with `/analyze` wall-clock, CPU, and
peak RSS measurements on a generated 60s 1080p reel before publishing p50/p95
targets or flipping `benchmarkGatePassed`.
