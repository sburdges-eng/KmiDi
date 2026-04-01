# Demo Slice: Proof-of-Life Report

**Date:** 2026-03-31
**Platform:** macOS 26.4, Apple Silicon (arm64)
**Plugin:** Kelly Emotion Processor (AU, kAudioUnitType_MusicEffect)

## Build

- ONNX Runtime: 1.24.4 via Homebrew (`/opt/homebrew/opt/onnxruntime/`)
- CMake flags: `BUILD_PLUGINS=ON`, `KMIDI_BUILD_JUCE_UI=ON`, `ENABLE_ONNX_RUNTIME=ON`
- Build target: `KellyPlugin_AU`
- Binary: 11.8MB arm64 Mach-O bundle
- Pre-existing build fixes required:
  - `HarmonyEngine.cpp`: removed `.resize()` calls on `std::array`
  - `OSCHub.cpp`: signature mismatch between declaration and definition of `matchPattern`
  - `ONNXInference.cpp`: dangling reference from rvalue `.get()` result

## AU Validation (`auval`)

- Result: **60 PASS / 2 FAIL**
- Failures: `MusicDeviceMIDIEventList` — expected for MusicEffect type (not a synth)
- All connection, render, parameter, and scheduling tests pass

## Model Path

- ONNX model: `models/audio_jepa_v01.onnx` (3.5MB)
- Core ML model: `models/audio_jepa_v01.mlpackage` (1.8MB, ANE-preferred, p50=0.56ms)
- Resolution: bundle Resources → sibling models/ → dev fallback path

## Logic Pro Test

- Plugin loads: [ ] (manual test pending)
- Audio input received: [ ]
- Emotion parameters animate: [ ]
- MIDI output produced: [ ]
- Manual slider mode works: [ ]

## How to Test in Logic Pro

1. Open Logic Pro
2. Create a Software Instrument track
3. On the track's MIDI FX slot, insert "Kelly Emotion Processor"
4. Create an audio track, import/play any music clip
5. In the plugin window, set **ML Influence** to 1.0
6. Watch **Valence** and **Arousal** parameters — should animate as audio plays
7. Set **ML Influence** to 0.0 and adjust sliders manually — MIDI should respond
8. Fill in the checklist above with results

## Core ML Latency Reference

From `bench/latency_report.md` (Audio JEPA encoder, batch=1):

| Runtime | p50 (ms) | p99 (ms) |
|---------|----------|----------|
| ONNX Runtime | 9.733 | 12.845 |
| Core ML (ANE) | 0.561 | 0.637 |

Both well under the 8ms acceptance gate for real-time inference.

## Next Steps

- [ ] Complete manual Logic Pro test and fill in results above
- [ ] Wire Core ML backend in C++ (CoreML.framework) for 17x faster inference
- [ ] Train emotion tag classifier for meaningful valence/arousal mapping
- [ ] Build React emotion visualization UI
