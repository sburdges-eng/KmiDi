# Demo Slice: Proof-of-Life

**Date:** 2026-03-31
**Goal:** Load AU in Logic Pro, play audio, watch emotion parameters move, hear MIDI generation respond.

## Context

The AU plugin pipeline is 95% wired. AudioEmotionRunner, IntentPipeline, MidiGenerator (12+ engines), and parameter automation are all implemented. The gap: ONNX Runtime isn't installed, the model path is stubbed, and the plugin hasn't been built with ML enabled.

## Scope

Internal proof-of-life only. Demonstrate that audio input drives emotion detection, emotion drives MIDI generation, and the whole loop works in a real DAW.

## Steps

### 1. Install ONNX Runtime

`brew install onnxruntime` — formula exists, not yet installed. CMake finds it via `find_package(ONNXRuntime)` with `ONNXRuntime_ROOT` hint.

### 2. Set model path

`src/plugin/PluginProcessor.cpp` ~line 361: set `emotionConfig.model_path` to resolve `models/audio_jepa_v01.onnx`. Use executable-relative or bundle Resources path.

### 3. Build AU plugin

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PLUGINS=ON \
  -DKMIDI_BUILD_JUCE_UI=ON \
  -DENABLE_ONNX_RUNTIME=ON
cmake --build build --target KellyPlugin_AU -j8
```

### 4. Install and validate

- Copy `.component` to `~/Library/Audio/Plug-Ins/Components/`
- `auval -v aumf Kmdi Kmdi` (or equivalent manufacturer/subtype codes)
- Open Logic Pro, insert as MIDI effect

### 5. Test the flow

1. Play audio on input → watch Valence/Arousal parameters animate
2. Set ML_INFLUENCE = 1.0 → confirm detected emotion drives MIDI
3. Listen for MIDI notes responding to emotion changes
4. Toggle ML_INFLUENCE = 0.0 → confirm manual sliders work independently

### 6. Document

Write `docs/demo/2026-03-31-proof-of-life.md` with results, latency observations, issues.

## Key files

| File | Change |
|------|--------|
| `src/plugin/PluginProcessor.cpp` | Set model path (~line 361) |
| `CMakeLists.txt` | Build flags (no source changes needed) |

## Risks

- CMake may need `ONNXRuntime_ROOT` hint pointing to brew prefix
- JEPA latent-to-emotion mapping may not produce meaningful values (untrained mapping layer) — parameters will still move, proving pipeline connectivity
- AU validation may surface runtime issues not caught in unit tests

## Success criteria

- AU loads in Logic without crash
- Playing audio causes emotion parameters to change (any movement = success)
- MIDI output responds to emotion parameter changes
- Manual slider mode works independently
