# Audio JEPA Latency Report

**Date:** 2026-04-01 20:15
**Platform:** macOS-26.4-arm64-arm-64bit-Mach-O
**Input shape:** (1, 1, 128, 512) — 128 mels, 512 frames @ 22050 Hz
**Output shape:** (1, 512, 256)

## Artifacts

| Format | Path | Verified |
|--------|------|----------|
| ONNX | `models/audio_jepa_v01.onnx` | PASS |
| Core ML | `models/audio_jepa_v01.mlpackage` | N/A |

## Latency (warm-started, batch=1)

| Runtime | p50 (ms) | p95 (ms) | p99 (ms) | min (ms) | max (ms) |
|---------|----------|----------|----------|----------|----------|
| ONNX Runtime | 8.678 | 9.382 | 9.712 | 8.095 | 9.867 |
| Core ML (ALL compute units) | 0.564 | 0.598 | 0.653 | 0.542 | 0.677 |

## Target

Phase 2 acceptance gate: forward latency <= 8 ms at batch=1 on Apple Silicon.

## Notes

- Exported with coremltools, Python 3.13.12, compute_units=ALL (ANE-preferred)
- mlProgram format uses fp16 weights by default (macOS13+ deployment target)
