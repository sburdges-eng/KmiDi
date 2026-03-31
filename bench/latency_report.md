# Audio JEPA Latency Report

**Date:** 2026-03-31 01:58
**Platform:** macOS-26.4-arm64-arm-64bit-Mach-O
**Input shape:** (1, 1, 128, 512) — 128 mels, 512 frames @ 22050 Hz
**Output shape:** (1, 512, 256)

## Artifacts

| Format | Path | Verified |
|--------|------|----------|
| ONNX | `models/audio_jepa_v01.onnx` | PASS |
| Core ML | `N/A (deferred)` | N/A |

## Latency (warm-started, batch=1)

| Runtime | p50 (ms) | p95 (ms) | p99 (ms) | min (ms) | max (ms) |
|---------|----------|----------|----------|----------|----------|
| ONNX Runtime | 9.241 | 10.745 | 12.831 | 8.268 | 16.056 |

## Target

Phase 2 acceptance gate: forward latency <= 8 ms at batch=1 on Apple Silicon.

## Notes

- Core ML export deferred: coremltools 9.0 lacks Python 3.14 native bindings.
  Re-run with a Python 3.12 venv or when coremltools ships 3.14 support.
