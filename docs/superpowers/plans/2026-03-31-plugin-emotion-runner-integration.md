# Plugin AudioEmotionRunner Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire AudioEmotionRunner into KellyPlugin's processBlock so live audio drives emotion detection, blended with manual sliders via an "ML Influence" parameter.

**Architecture:** AudioEmotionRunner is initialized in prepareToPlay, fed mono-mixed audio in processBlock, and its results are blended with APVTS slider values using a new `ml_influence` parameter. All operations are lock-free on the audio thread.

**Tech Stack:** C++20, JUCE 8, AudioEmotionRunner (penta::ml), APVTS

---

### Task 1: Add ml_influence Parameter to APVTS

**Files:**
- Modify: `src/plugin/PluginProcessor.h`
- Modify: `src/plugin/PluginProcessor.cpp`

- [ ] **Step 1: Add parameter ID constant**

In `src/plugin/PluginProcessor.h`, after line 120 (`PARAM_USE_HOST_TEMPO`), add:

```cpp
  static constexpr const char *PARAM_ML_INFLUENCE = "ml_influence";
```

- [ ] **Step 2: Add parameter to layout**

In `src/plugin/PluginProcessor.cpp`, find the `createParameterLayout()` function. Search for the last `layout.add` call in the emotion parameters section (before the EQ parameters). Add after it:

```cpp
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_ML_INFLUENCE, PARAM_VERSION},
        "ML Influence",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        0.0f,
        juce::AudioParameterFloatAttributes().withLabel("Blend")));
```

- [ ] **Step 3: Commit**

```bash
cd /Users/seanburdges/Dev/KmiDi
git add src/plugin/PluginProcessor.h src/plugin/PluginProcessor.cpp
git commit -m "feat: add ml_influence parameter to APVTS (0=manual, 1=ML)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add AudioEmotionRunner Member and Mono Buffer

**Files:**
- Modify: `src/plugin/PluginProcessor.h`
- Modify: `src/plugin/PluginProcessor.cpp`

- [ ] **Step 1: Add include and members to header**

In `src/plugin/PluginProcessor.h`, add include after the existing ML includes (after line 44, `#include "ml/PluginLatencyManager.h"`):

```cpp
#include "penta/ml/AudioEmotionRunner.h"
#include "penta/common/RTState.h"
```

In the private section, after line 401 (`int64_t sampleCounter_ = 0;`), add:

```cpp
  // AudioEmotionRunner — RT-safe JEPA emotion inference
  std::unique_ptr<penta::ml::AudioEmotionRunner> emotionRunner_;
  penta::RTState emotionRTState_;
  std::vector<float> monoMixBuffer_; // Pre-allocated scratch for mono downmix
```

- [ ] **Step 2: Initialize in prepareToPlay**

In `src/plugin/PluginProcessor.cpp`, in `prepareToPlay()`, after the multi-model processor initialization section (after the existing ML setup around line 305), add:

```cpp
  // Initialize AudioEmotionRunner for RT emotion detection
  if (mlInferenceEnabled_.load()) {
    penta::ml::AudioEmotionRunnerConfig emotionConfig;
    emotionConfig.model_path = "";  // Stub mode — no ONNX model yet
    emotionConfig.sample_rate = static_cast<size_t>(sampleRate);
    emotionConfig.ring_capacity = 524288;
    emotionConfig.slew_time_ms = 20.0f;
    emotionConfig.confidence_threshold = 0.3f;

    emotionRunner_ = std::make_unique<penta::ml::AudioEmotionRunner>();
    emotionRunner_->initialize(emotionConfig);
  }

  // Pre-allocate mono mix buffer (no heap alloc in processBlock)
  monoMixBuffer_.resize(static_cast<size_t>(samplesPerBlock));
```

- [ ] **Step 3: Shutdown in releaseResources**

In `src/plugin/PluginProcessor.cpp`, in `releaseResources()`, before the existing `inferenceManager_.stop();` (line 328), add:

```cpp
  if (emotionRunner_) {
    emotionRunner_->shutdown();
    emotionRunner_.reset();
  }
```

- [ ] **Step 4: Commit**

```bash
cd /Users/seanburdges/Dev/KmiDi
git add src/plugin/PluginProcessor.h src/plugin/PluginProcessor.cpp
git commit -m "feat: add AudioEmotionRunner member, init in prepareToPlay, shutdown in release

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire processBlock — Push Samples and Blend

**Files:**
- Modify: `src/plugin/PluginProcessor.cpp`

- [ ] **Step 1: Add mono-mix, push, and blend logic to processBlock**

In `src/plugin/PluginProcessor.cpp`, in `processBlock()`, find the existing ML inference block (lines 363-392, starting with `if (mlInferenceEnabled_.load() && numChannels > 0)`). Add the AudioEmotionRunner logic INSIDE this block, AFTER the existing inference code (after line 392, before the lookahead read-back). Insert:

```cpp
    // --- AudioEmotionRunner: push mono-mixed audio and blend ---
    if (emotionRunner_ && emotionRunner_->isRunning()) {
      // Mono downmix into pre-allocated buffer
      const size_t n = static_cast<size_t>(numSamples);
      if (n <= monoMixBuffer_.size()) {
        if (numChannels == 1) {
          std::memcpy(monoMixBuffer_.data(), buffer.getReadPointer(0),
                      n * sizeof(float));
        } else {
          const float *left = buffer.getReadPointer(0);
          const float *right =
              numChannels > 1 ? buffer.getReadPointer(1) : left;
          for (size_t i = 0; i < n; ++i) {
            monoMixBuffer_[i] = 0.5f * (left[i] + right[i]);
          }
        }

        // Push to ring buffer (lock-free, non-blocking)
        emotionRunner_->pushSamples(monoMixBuffer_.data(), n);

        // Read latest inference results into RTState (lock-free)
        emotionRunner_->updateParams(emotionRTState_,
                                     static_cast<size_t>(numSamples));

        // Blend detected emotion with manual slider values
        const float blend =
            apvts_.getRawParameterValue(PARAM_ML_INFLUENCE)->load();

        if (blend > 0.0f) {
          const float detectedV =
              emotionRTState_.valence.load(std::memory_order_relaxed);
          const float detectedA =
              emotionRTState_.arousal.load(std::memory_order_relaxed);
          const float manualV =
              apvts_.getRawParameterValue(PARAM_VALENCE)->load();
          const float manualA =
              apvts_.getRawParameterValue(PARAM_AROUSAL)->load();

          // Linear blend: 0=fully manual, 1=fully detected
          mlValence_.store((1.0f - blend) * manualV + blend * detectedV,
                          std::memory_order_relaxed);
          mlArousal_.store((1.0f - blend) * manualA + blend * detectedA,
                          std::memory_order_relaxed);
        } else {
          // Pure manual mode — use slider values directly
          mlValence_.store(
              apvts_.getRawParameterValue(PARAM_VALENCE)->load(),
              std::memory_order_relaxed);
          mlArousal_.store(
              apvts_.getRawParameterValue(PARAM_AROUSAL)->load(),
              std::memory_order_relaxed);
        }
      }
    }
```

Note: `mlValence_` and `mlArousal_` already exist as atomic members (lines 399-400 in header). The existing MIDI generation code should already read from these atomics. If it reads from APVTS directly instead, downstream code will need to be updated to read from `mlValence_`/`mlArousal_` — but that's existing code structure, not a new change.

- [ ] **Step 2: Verify no allocation on audio thread**

Review the added code:
- `monoMixBuffer_` — pre-allocated in prepareToPlay ✓
- `pushSamples` — lock-free ring buffer ✓
- `updateParams` — lock-free SPSC dequeue + slew ✓
- Blend arithmetic — pure math ✓
- No `new`, no `std::vector`, no `std::string`, no locks ✓

- [ ] **Step 3: Commit**

```bash
cd /Users/seanburdges/Dev/KmiDi
git add src/plugin/PluginProcessor.cpp
git commit -m "feat: wire AudioEmotionRunner into processBlock — push, read, blend

Audio thread pushes mono-mixed input to AudioEmotionRunner ring buffer,
reads slew-limited emotion values from RTState, and blends with manual
slider values using ml_influence parameter. Zero allocations on RT path.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Handle enableMLInference Toggle

**Files:**
- Modify: `src/plugin/PluginProcessor.cpp`

- [ ] **Step 1: Update enableMLInference to manage AudioEmotionRunner lifecycle**

In `src/plugin/PluginProcessor.cpp`, find the `enableMLInference` method. Read it first. Add AudioEmotionRunner init/shutdown when toggling:

```cpp
void PluginProcessor::enableMLInference(bool enable) {
  mlInferenceEnabled_.store(enable);

  if (enable) {
    // Start AudioEmotionRunner if not already running
    if (!emotionRunner_ || !emotionRunner_->isRunning()) {
      penta::ml::AudioEmotionRunnerConfig config;
      config.model_path = "";
      config.sample_rate = static_cast<size_t>(currentSampleRate_);
      config.ring_capacity = 524288;
      config.slew_time_ms = 20.0f;
      config.confidence_threshold = 0.3f;

      emotionRunner_ = std::make_unique<penta::ml::AudioEmotionRunner>();
      emotionRunner_->initialize(config);
    }
    // ... existing inference manager start code ...
  } else {
    // Stop AudioEmotionRunner
    if (emotionRunner_) {
      emotionRunner_->shutdown();
      emotionRunner_.reset();
    }
    // ... existing inference manager stop code ...
  }
}
```

Note: Read the existing method body first. Preserve all existing logic. Add AudioEmotionRunner management alongside it. The `emotionRunner_` creation involves heap allocation so this MUST be called from the message thread (not audio thread), which `enableMLInference` already is.

- [ ] **Step 2: Commit**

```bash
cd /Users/seanburdges/Dev/KmiDi
git add src/plugin/PluginProcessor.cpp
git commit -m "feat: AudioEmotionRunner lifecycle managed by ML inference toggle

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Verification

**Files:** None new

- [ ] **Step 1: Verify compilation**

This can only be verified if JUCE is installed at `external/JUCE/` and `KMIDI_BUILD_JUCE_UI=ON`. Run:

```bash
cd /Users/seanburdges/Dev/KmiDi
cmake -S . -B build -G Ninja -DBUILD_PLUGINS=ON -DKMIDI_BUILD_JUCE_UI=ON 2>&1 | tail -5
cmake --build build --target KellyPlugin_AU -j8 2>&1 | tail -20
```

If JUCE is not available, verify at minimum that the header includes resolve:

```bash
grep -n "AudioEmotionRunner" src/plugin/PluginProcessor.h
grep -n "RTState" src/plugin/PluginProcessor.h
grep -n "ml_influence" src/plugin/PluginProcessor.cpp
grep -n "emotionRunner_" src/plugin/PluginProcessor.cpp
grep -n "monoMixBuffer_" src/plugin/PluginProcessor.cpp
```

Expected: All greps return matches at the expected locations.

- [ ] **Step 2: Verify existing tests unaffected**

```bash
cd /Users/seanburdges/Dev/KmiDi
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest tests/unit/ -q --ignore=tests/unit/test_export_audio_jepa.py 2>&1 | tail -5
cd src-tauri && cargo test 2>&1 | grep "test result"
```

Expected: No new failures.

- [ ] **Step 3: Commit any fixes**

If any adjustments were needed:

```bash
git add -u
git commit -m "fix: adjustments from plugin integration verification

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
