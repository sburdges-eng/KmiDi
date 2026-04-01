# Audio JEPA Latency Report

**Date:** 2026-03-31 19:52
**Platform:** macOS-26.4-arm64-arm-64bit-Mach-O
**Input shape:** (1, 1, 128, 512) — 128 mels, 512 frames @ 22050 Hz
**Output shape:** (1, 512, 256)

## Artifacts

| Format | Path | Verified |
|--------|------|----------|
| ONNX | `models/audio_jepa_v01.onnx` | PASS |
| Core ML | `models/audio_jepa_v01.mlpackage` | PASS |

## Latency (warm-started, batch=1)

| Runtime | p50 (ms) | p95 (ms) | p99 (ms) | min (ms) | max (ms) |
|---------|----------|----------|----------|----------|----------|
| ONNX Runtime | 9.733 | 12.466 | 12.845 | 8.077 | 13.082 |
| Core ML (ALL compute units) | 0.561 | 0.617 | 0.637 | 0.516 | 0.655 |

## Target

Phase 2 acceptance gate: forward latency <= 8 ms at batch=1 on Apple Silicon.

## Notes

- Exported with coremltools, Python 3.13.12, compute_units=ALL (ANE-preferred)
- mlProgram format uses fp16 weights by default (macOS13+ deployment target)
