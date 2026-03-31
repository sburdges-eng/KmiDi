# Plugin AudioEmotionRunner Integration — Design Spec

**Date:** 2026-03-31
**Phase:** 3 (Local AU Helper) of the 90-Day Demo Roadmap

## Goal

Wire AudioEmotionRunner into KellyPlugin so the AU/VST3 plugin detects emotion from live audio input and blends it with manual slider values via a single "ML Influence" knob.

## Integration Points

### New member in PluginProcessor

```cpp
std::unique_ptr<penta::ml::AudioEmotionRunner> emotionRunner_;
penta::RTState rtState_;           // if not already present
std::vector<float> monoMixBuffer_; // pre-allocated scratch for mono downmix
```

### Lifecycle

- **prepareToPlay():** If `mlInferenceEnabled_` is true, initialize AudioEmotionRunner with `{model_path="", sample_rate=sampleRate, ring_capacity=524288, slew_time_ms=20.0f, confidence_threshold=0.3f}`. Pre-allocate `monoMixBuffer_` to `samplesPerBlock`.
- **processBlock():** If ML enabled and emotionRunner initialized, mono-mix input → pushSamples → updateParams → blend.
- **releaseResources():** Shutdown emotionRunner.

### New APVTS Parameter

| Parameter | ID | Range | Default | Description |
|-----------|-----|-------|---------|-------------|
| ML Influence | `ml_influence` | 0.0–1.0 | 0.0 | 0=fully manual, 1=fully ML-detected |

### Blend Logic (in processBlock)

```
blend = getRawParameterValue("ml_influence")
detected_v = rtState_.valence.load(relaxed)
detected_a = rtState_.arousal.load(relaxed)
manual_v = getRawParameterValue("valence")
manual_a = getRawParameterValue("arousal")

blended_v = (1 - blend) * manual_v + blend * detected_v
blended_a = (1 - blend) * manual_a + blend * detected_a
```

Blended values feed into MIDI generation logic via local variables. No APVTS writes from the audio thread.

Dominance and confidence from AudioEmotionRunner stay internal (in RTState) — not exposed as APVTS parameters. Available for future use.

## RT-Safety

- `pushSamples()`: lock-free ring buffer enqueue (noexcept)
- `updateParams()`: lock-free SPSC dequeue + slew limiting (noexcept)
- Mono-mix: simple arithmetic on pre-allocated buffer
- Blend: pure arithmetic, no allocations
- No `setValueNotifyingHost()` from audio thread
- AudioEmotionRunner worker thread runs inference off the audio thread

## Gating

- AudioEmotionRunner is only initialized/active when `mlInferenceEnabled_` is true
- When `ml_influence` is 0.0, detected values are computed but not used (zero-cost blend)
- When ML is disabled, pushSamples is skipped entirely

## Files Modified

| File | Change |
|------|--------|
| `src/plugin/PluginProcessor.h` | Add emotionRunner_, rtState_, monoMixBuffer_ members |
| `src/plugin/PluginProcessor.cpp` | Init in prepareToPlay, push+blend in processBlock, shutdown in releaseResources |
| `src/plugin/PluginProcessor.cpp` | Add `ml_influence` to createParameterLayout |

## What This Does NOT Include

- UI controls for ml_influence (host exposes it as an automatable parameter)
- Core ML inference backend (uses ONNX stub mode for now)
- Per-dimension blend knobs
- New "detected_valence" read-only parameters
- Changes to AudioEmotionRunner itself

## Acceptance Criteria

1. AudioEmotionRunner initializes in prepareToPlay when ML is enabled
2. processBlock pushes mono audio and reads back emotion values
3. `ml_influence` parameter blends manual and detected values
4. No allocations on audio thread — all buffers pre-allocated
5. Plugin compiles with existing KellyCore linkage (AudioEmotionRunner is part of KellyCore)
6. Existing tests unaffected
