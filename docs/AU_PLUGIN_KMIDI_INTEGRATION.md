# AU Plugin ↔ KmiDi Engine Integration Reference

> How KmiDi's emotion/intent engine, ML models, and MIDI generation map
> to an Audio Unit plugin's real-time constraints. This is the bridge
> document between the music brain and the plugin wrapper.

## 1. Component Map

```
┌──────────────────────────────────────────────────────────────────┐
│                    KmiDi AU Plugin (.component)                   │
│                                                                  │
│  ┌────────────────────┐    ┌──────────────────────────────────┐  │
│  │  PluginEditor      │    │  PluginProcessor                 │  │
│  │  (JUCE UI)         │◄──►│  (Audio Thread)                  │  │
│  │                    │    │                                  │  │
│  │  EmotionWorkstation│    │  processBlock()                  │  │
│  │  MasterEQComponent │    │    ├─ Read host tempo/position   │  │
│  │  IntentIR Inspector│    │    ├─ Read MIDI input             │  │
│  │  PluginLogger      │    │    ├─ Read params (atomic)        │  │
│  │                    │    │    ├─ Output generated MIDI       │  │
│  └────────────────────┘    │    └─ MasterEQ (if aumf)         │  │
│                            │                                  │  │
│  ┌─────────────────────┐   │  ┌──────────────────────────┐    │  │
│  │  UI Thread          │   │  │  KellyBrain              │    │  │
│  │                     │   │  │  (Intent → MIDI)         │    │  │
│  │  generateMidi()     │──►│  │                          │    │  │
│  │  setWoundDescription│   │  │  IntentPipeline          │    │  │
│  │  setSelectedEmotion │   │  │  MidiGenerator           │    │  │
│  │  setMusicTheory     │   │  │  Engines (Melody, Bass,  │    │  │
│  └─────────────────────┘   │  │    Drums, Pad, etc.)     │    │  │
│                            │  └──────────────────────────┘    │  │
│  ┌─────────────────────┐   │                                  │  │
│  │  Inference Thread   │   │  ┌──────────────────────────┐    │  │
│  │                     │   │  │  ML Models               │    │  │
│  │  ONNX / RTNeural    │──►│  │  MultiModelProcessor     │    │  │
│  │  JEPA embeddings    │   │  │  FeatureExtractor        │    │  │
│  │  Emotion classifier │   │  │  LatencyManager          │    │  │
│  └─────────────────────┘   │  └──────────────────────────┘    │  │
│                            └──────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## 2. IntentIR → AU Parameter Mapping

The `IntentIRAdapter` converts between the IntentIR frame format and
parameters that the plugin processor understands. This mapping is critical
for AU because **every automatable parameter must be declared up-front**
in the AU parameter tree.

### Current IntentIR fields → Plugin parameters

| IntentFrame Field | Plugin Param | AU Range | Mapping |
|---|---|---|---|
| `emotion.valence` | `valence` | 0.0–1.0 | Direct |
| `emotion.arousal` | `arousal` | 0.0–1.0 | Direct |
| `emotion.tension` | `intensity` | 0–100% | `tension * 100` |
| `musical.tempo_bias` | Host tempo | N/A | Host-provided; plugin reads via `getPlayHead()` |
| `musical.mode_preference` | Internal | N/A | -1=minor, 0=modal, +1=major; not a user param |
| `musical.rhythmic_density` | `feel` | 0.0–1.0 | Maps to swing/straight |
| `musical.groove_strength` | `humanize` | 0–100% | Syncopation amount |
| `musical.harmonic_tension` | `complexity` | 0.0–1.0 | Chromaticism level |
| `musical.dynamic_range` | `dynamics` | 0.0–1.0 | Velocity range |
| `constraints.bar_count` | `bars` | 1–64 | Direct integer |

### Fields NOT exposed as AU parameters

These are internal engine state, not user-facing automation targets:

- `musical.melodic_activity` — controlled by ML inference
- `musical.contour_variance` — controlled by ML inference
- `scope.*` — timing context from host transport
- `emotion.dominance` — derived from valence + arousal

## 3. MIDI Generation Pipeline

### Trigger flow (user clicks "Generate" in plugin UI)

```
1. UI Thread: user adjusts emotion wheel / enters wound text
2. UI Thread: calls PluginProcessor::generateMidi()
3. UI Thread: IntentPipeline processes emotion → IntentResult
4. UI Thread: MidiGenerator generates MIDI from IntentResult
5. UI Thread: stores result in generatedMidi_ (mutex-protected)
6. UI Thread: sets hasPendingMidi_ = true
7. Audio Thread: processBlock() sees hasPendingMidi_
8. Audio Thread: try_lock on midiMutex_, copies MIDI to outputBuffer_
9. Audio Thread: writes outputBuffer_ to host MidiBuffer
10. Host: routes MIDI to downstream instrument tracks
```

### Real-time ML path (audio-driven)

```
1. Audio Thread: processBlock() receives audio from host
2. Audio Thread: extracts features (128-dim) into lookahead buffer
3. Inference Thread: picks up features, runs ONNX model
4. Inference Thread: writes mlValence_, mlArousal_ atomically
5. Audio Thread: reads ml values, adjusts emotion parameters
6. UI Thread: periodically regenerates MIDI from updated emotion
```

## 4. Engine Components Available for AU

### Already real-time safe (can use in processBlock)

| Component | File | Notes |
|---|---|---|
| `MasterEQProcessor` | `src/plugin/MasterEQProcessor.cpp` | 6-band biquad EQ, no allocation |
| `MLFeatureExtractor` | `src/ml/MLFeatureExtractor.cpp` | 128-dim feature extraction |
| `PluginLatencyManager` | `src/ml/PluginLatencyManager.h` | Lookahead buffer management |
| APVTS parameter reads | `getRawParameterValue()` | Lock-free atomic reads |

### UI-thread only (NOT safe for processBlock)

| Component | File | Notes |
|---|---|---|
| `IntentPipeline` | `src/engine/IntentPipeline.cpp` | Heavy processing, locks |
| `KellyBrain` | `src/engine/KellyBrain.cpp` | Text processing, allocates |
| `MidiGenerator` | `src/midi/MidiGenerator.cpp` | Generates from IntentResult |
| All `src/engines/*` | Melody, Bass, Drums, etc. | Generation engines |

### Background-thread safe

| Component | File | Notes |
|---|---|---|
| `InferenceThreadManager` | `src/ml/InferenceThreadManager.h` | Manages async inference |
| `MultiModelProcessor` | `src/ml/MultiModelProcessor.cpp` | Runs ONNX/RTNeural models |
| `ONNXInference` | `src/ml/ONNXInference.cpp` | ONNX Runtime wrapper |

## 5. JEPA Integration in AU Context

The JEPA models (`music_brain/jepa/`) produce latent embeddings that can
drive the emotion parameters. In the AU plugin:

```
Audio In (if aumf) ──► mel-spectrogram ──► AudioJEPAEncoder
                                                │
                                         latent embedding
                                                │
                                    ┌───────────┴──────────┐
                                    │   Emotion Classifier  │
                                    │   (ONNX, ~5ms)       │
                                    └───────────┬──────────┘
                                                │
                                    valence, arousal, tension
                                                │
                                    ┌───────────┴──────────┐
                                    │   IntentIR Update     │
                                    │   (atomic writes)     │
                                    └───────────┬──────────┘
                                                │
                                    ┌───────────┴──────────┐
                                    │   MIDI Regeneration   │
                                    │   (UI thread)         │
                                    └──────────────────────┘
```

**Constraint:** JEPA encoder inference (~5ms on Apple Silicon) must run on
the inference thread, not the audio thread. The `InferenceThreadManager`
already handles this pattern.

## 6. AU-Specific Edge Cases

### 1. Logic Pro MIDI FX Routing

When loaded as MIDI FX (`'aumi'`), Logic Pro:
- Does **not** send audio to the plugin
- Sends MIDI from the track's MIDI input
- Routes plugin's MIDI output to the track's instrument
- Supports automation of all declared parameters

**Implication:** `processBlock()` audio buffer will be empty. Only use
MIDI buffer and parameter values.

### 2. Host Tempo Sync

```cpp
void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    if (auto* playHead = getPlayHead()) {
        if (auto pos = playHead->getPosition()) {
            if (auto bpm = pos->getBpm())
                hostTempoBpm_.store(static_cast<float>(*bpm));
            if (auto ppq = pos->getPpqPosition())
                playheadPosition_ = *ppq;
            isHostPlaying_.store(pos->getIsPlaying());
        }
    }
    // ... rest of processing
}
```

### 3. AU Preset Management

Logic Pro expects AU presets via the standard AU preset mechanism.
JUCE handles this through `getStateInformation()` / `setStateInformation()`.
Additionally, implement program support if offering factory presets:

```cpp
int getNumPrograms() override { return factoryPresets_.size(); }
const juce::String getProgramName(int index) override {
    return factoryPresets_[index].name;
}
```

### 4. Sandbox Considerations (AUv3)

If building AUv3, the plugin runs in an app extension sandbox:
- No filesystem access outside container
- No network access by default
- Model files must be bundled in the app extension
- Use `AU_SANDBOX_SAFE TRUE` in CMake

### 5. Apple Silicon AudioWorkgroup

```cpp
void audioWorkgroupContextChanged(const juce::AudioWorkgroup& workgroup) override
{
    audioWorkgroup_ = workgroup;
    // Inference thread should join this workgroup for scheduling priority
    if (inferenceManager_.isRunning())
        inferenceManager_.joinWorkgroup(workgroup);
}
```

## 7. Migration Path: VST3-only → AU+VST3

The existing `PluginProcessor` is **already AU-compatible** in code. The only
change required is the CMake configuration (adding `AU` to FORMATS). No C++
changes are needed for basic AU support because:

1. JUCE's AU wrapper calls the same `processBlock()`, `prepareToPlay()`, etc.
2. `AudioProcessorValueTreeState` maps directly to AU parameter tree
3. `getStateInformation()` / `setStateInformation()` handle AU preset storage
4. `isMidiEffect() = true` correctly maps to `'aumi'` type
5. Bus layout negotiation is handled by JUCE's AU wrapper

### What might need adjustment

| Area | Risk | Mitigation |
|---|---|---|
| `MasterEQProcessor` in MIDI-only mode | EQ expects audio buffers | Gate behind `if (!isMidiEffect())` or split into separate plugin |
| Thread safety | AU hosts may call `processBlock` from different thread than VST3 | Already handled — `processBlock` uses atomics and `try_lock` |
| Large model loading | AU validation (`auval`) times out if `prepareToPlay` is slow | Load models async; return immediately from `prepareToPlay` |
| Editor size | Some AU hosts have different editor size negotiation | Use `setResizable(true, true)` with min/max constraints |

## 8. Recommended Build Order

1. **Add `AU` to CMake FORMATS** (see `AU_PLUGIN_BUILD.md` §1)
2. **Build on macOS** with Xcode CLT or Ninja
3. **Run `auval`** to validate basic AU compliance
4. **Test in JUCE AudioPluginHost** for debugging
5. **Test in Logic Pro** on a MIDI track (MIDI FX slot)
6. **Test state save/load** — close and reopen Logic project
7. **Test automation** — draw automation lanes for valence/arousal
8. **Profile** — ensure no audio dropouts during generation
9. **Code sign and notarize** for distribution
10. **Add to CI** — `auval` in GitHub Actions macOS runner

## 9. File Inventory (what exists vs. what's needed)

### Exists (no changes needed)

| File | Role |
|---|---|
| `src/plugin/PluginProcessor.cpp/.h` | Core processor — already AU-ready |
| `src/plugin/PluginEditor.cpp/.h` | UI editor |
| `src/plugin/PluginState.cpp/.h` | State persistence |
| `src/plugin/MasterEQProcessor.cpp/.h` | EQ DSP |
| `src/engine/KellyBrain.cpp/.h` | Intent processing |
| `src/engine/IntentPipeline.cpp/.h` | Emotion → intent |
| `src/midi/MidiGenerator.cpp/.h` | MIDI generation |
| `src/ml/ONNXInference.cpp/.h` | ONNX runtime |
| `src/ml/MultiModelProcessor.cpp/.h` | Multi-model ML |
| `src/common/IntentIRAdapter.cpp/.h` | IntentIR ↔ params |
| `music_brain/jepa/` | JEPA models (Python, for training; ONNX for inference) |

### Needs CMake-only change

| File | Change |
|---|---|
| `CMakeLists.txt` | Add `AU` to FORMATS, add AU-specific properties |

### Optional additions

| File | Purpose |
|---|---|
| `src/plugin/AUPresets.cpp` | Factory AU presets (emotion presets) |
| `scripts/build_and_validate_au.sh` | Build + install + auval in one step |
| `.github/workflows/au_build.yml` | CI pipeline for AU builds on macOS |
