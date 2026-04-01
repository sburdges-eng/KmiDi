# Demo Slice: Proof-of-Life Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the AU plugin with ONNX Runtime, load it in Logic Pro, and verify audio-driven emotion detection drives MIDI generation end-to-end.

**Architecture:** The AU plugin (JUCE 8, `kAudioUnitType_MusicEffect`) receives audio → AudioEmotionRunner extracts mel spectrogram → ONNX Runtime runs audio_jepa_v01 → latent output mapped to valence/arousal → IntentPipeline → MidiGenerator → MIDI output. All in-process C++, RT-safe.

**Tech Stack:** C++20, JUCE 8, ONNX Runtime 1.24, CMake/Ninja, Logic Pro (host)

---

### Task 1: Install ONNX Runtime and verify CMake can find it

**Files:**
- Create: `cmake/FindONNXRuntime.cmake` (CMake find module)
- Modify: `CMakeLists.txt:226-234` (add module path)

- [ ] **Step 1: Install ONNX Runtime via Homebrew**

Run:
```bash
brew install onnxruntime
```

Expected: installs to `/opt/homebrew/opt/onnxruntime/`

- [ ] **Step 2: Verify the installation layout**

Run:
```bash
ls /opt/homebrew/opt/onnxruntime/include/onnxruntime/
ls /opt/homebrew/opt/onnxruntime/lib/libonnxruntime*
```

Expected: header files in `include/onnxruntime/` and `libonnxruntime.dylib` in `lib/`

- [ ] **Step 3: Create FindONNXRuntime.cmake**

Brew doesn't ship CMake config files for onnxruntime. Create a find module so `find_package(ONNXRuntime)` works:

```cmake
# cmake/FindONNXRuntime.cmake
# Find ONNX Runtime installed via Homebrew or manual install.
#
# Sets:
#   ONNXRuntime_FOUND
#   ONNXRuntime_INCLUDE_DIRS
#   ONNXRuntime_LIBRARIES

find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATH_SUFFIXES onnxruntime/core/session
    HINTS
        ${ONNXRuntime_ROOT}
        $ENV{ONNXRuntime_ROOT}
        /opt/homebrew/opt/onnxruntime
        /usr/local/opt/onnxruntime
    PATH_SUFFIXES include
)

find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    HINTS
        ${ONNXRuntime_ROOT}
        $ENV{ONNXRuntime_ROOT}
        /opt/homebrew/opt/onnxruntime
        /usr/local/opt/onnxruntime
    PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR
)

if(ONNXRuntime_FOUND)
    set(ONNXRuntime_LIBRARIES ${ONNXRuntime_LIBRARY})
    set(ONNXRuntime_INCLUDE_DIRS ${ONNXRuntime_INCLUDE_DIR})
endif()
```

- [ ] **Step 4: Add cmake/ to module path in CMakeLists.txt**

In `CMakeLists.txt`, after line 7 (`CMAKE_EXPORT_COMPILE_COMMANDS`), add:

```cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
```

- [ ] **Step 5: Verify CMake finds ONNX Runtime**

Run:
```bash
cd /Users/seanburdges/Dev/KmiDi
cmake -S . -B build-demo -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_KELLY_CORE=ON \
  -DBUILD_PLUGINS=ON \
  -DKMIDI_BUILD_JUCE_UI=ON \
  -DENABLE_ONNX_RUNTIME=ON \
  2>&1 | grep -i "onnx"
```

Expected: `ONNX Runtime enabled: /opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib`

- [ ] **Step 6: Commit**

```bash
git add cmake/FindONNXRuntime.cmake CMakeLists.txt
git commit -m "build: add FindONNXRuntime.cmake, enable ONNX Runtime discovery via brew"
```

---

### Task 2: Set model path in PluginProcessor

**Files:**
- Modify: `src/plugin/PluginProcessor.cpp:337`

- [ ] **Step 1: Set model path to resolve audio_jepa_v01.onnx**

In `src/plugin/PluginProcessor.cpp`, change line 337 from:

```cpp
    emotionConfig.model_path = "";  // Stub mode — no ONNX model yet
```

to:

```cpp
    // Resolve model path: prefer bundle Resources, then sibling models/ dir
    auto pluginFile = juce::File::getSpecialLocation(
        juce::File::currentApplicationFile);
    auto modelFile = pluginFile.getChildFile("Contents/Resources/models/audio_jepa_v01.onnx");
    if (!modelFile.existsAsFile())
        modelFile = pluginFile.getParentDirectory().getChildFile("models/audio_jepa_v01.onnx");
    if (!modelFile.existsAsFile()) {
        // Dev fallback: project root models/ directory
        modelFile = juce::File("/Users/seanburdges/Dev/KmiDi/models/audio_jepa_v01.onnx");
    }
    emotionConfig.model_path = modelFile.getFullPathName().toStdString();
```

Note: The hardcoded dev fallback path is acceptable for proof-of-life. Production would use env vars or ModelConfigManager.

- [ ] **Step 2: Verify the code compiles (syntax check)**

Run:
```bash
cmake --build build-demo --target KellyPlugin -j8 2>&1 | tail -20
```

Expected: compiles without errors (or at least PluginProcessor.cpp compiles). Full build may have other issues — that's Task 3.

- [ ] **Step 3: Commit**

```bash
git add src/plugin/PluginProcessor.cpp
git commit -m "feat: wire audio_jepa_v01.onnx model path in PluginProcessor"
```

---

### Task 3: Build the AU plugin

**Files:**
- No new files — build system only

- [ ] **Step 1: Configure the build**

Run:
```bash
cd /Users/seanburdges/Dev/KmiDi
cmake -S . -B build-demo -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_KELLY_CORE=ON \
  -DBUILD_PLUGINS=ON \
  -DKMIDI_BUILD_JUCE_UI=ON \
  -DENABLE_ONNX_RUNTIME=ON
```

Expected: configures successfully, mentions ONNX Runtime found.

- [ ] **Step 2: Build the AU target**

Run:
```bash
cmake --build build-demo --target KellyPlugin_AU -j8 2>&1
```

Expected: builds `KellyPlugin.component` in `build-demo/KellyPlugin_artefacts/Release/AU/`

If build fails, the errors will likely be in:
- Missing JUCE modules → check `external/JUCE/` submodule init
- Missing Qt6 → check if `KMIDI_BUILD_QT_UI` is accidentally ON (should be OFF)
- ONNX Runtime header issues → verify `ONNXRuntime_INCLUDE_DIRS` points to right path
- Linker errors → verify `ONNXRuntime_LIBRARIES` path

- [ ] **Step 3: Verify the AU component exists**

Run:
```bash
ls -la build-demo/KellyPlugin_artefacts/Release/AU/
file build-demo/KellyPlugin_artefacts/Release/AU/Kelly\ Emotion\ Processor.component/Contents/MacOS/*
```

Expected: `Kelly Emotion Processor.component` directory with a Mach-O binary inside.

---

### Task 4: Install and validate the AU

**Files:**
- No code changes

- [ ] **Step 1: Copy AU to system plugins directory**

Run:
```bash
cp -R "build-demo/KellyPlugin_artefacts/Release/AU/Kelly Emotion Processor.component" \
  ~/Library/Audio/Plug-Ins/Components/
```

- [ ] **Step 2: Copy model file to a location the AU can find**

The model path resolution in Task 2 tries the bundle Resources first, then falls back to the dev path. For dev testing, the fallback to `/Users/seanburdges/Dev/KmiDi/models/audio_jepa_v01.onnx` should work since that file exists.

Verify:
```bash
ls -la /Users/seanburdges/Dev/KmiDi/models/audio_jepa_v01.onnx
```

Expected: file exists (~3.5MB)

- [ ] **Step 3: Run AU validation**

Run:
```bash
auval -v aumf Klp1 Klly
```

(Format: `aumf` = MusicEffect, SubType: `Klp1`, Manufacturer: `Klly` — from CMakeLists.txt lines 408-409)

Expected: validation passes (or at least loads without crash). Some tests may fail if MIDI routing isn't standard — that's okay for proof-of-life.

- [ ] **Step 4: Test in Logic Pro**

Manual steps:
1. Open Logic Pro
2. Create a Software Instrument track
3. On the track's MIDI FX slot, insert "Kelly Emotion Processor"
4. Create an audio track, play audio (any music clip)
5. Route the audio to KmiDi's input (or use direct monitoring)
6. In the plugin window, set **ML Influence** slider to 1.0
7. Watch the **Valence** and **Arousal** parameters — they should animate as audio plays
8. Listen for MIDI output on the instrument track

Record observations.

---

### Task 5: Document results

**Files:**
- Create: `docs/demo/2026-03-31-proof-of-life.md`

- [ ] **Step 1: Create the demo directory**

Run:
```bash
mkdir -p /Users/seanburdges/Dev/KmiDi/docs/demo
```

- [ ] **Step 2: Write the proof-of-life report**

Create `docs/demo/2026-03-31-proof-of-life.md` with:

```markdown
# Demo Slice: Proof-of-Life Report

**Date:** 2026-03-31
**Platform:** macOS 26.4, Apple Silicon
**Host:** Logic Pro
**Plugin:** Kelly Emotion Processor (AU, MusicEffect)

## Build

- ONNX Runtime: [version, install method]
- CMake flags: BUILD_PLUGINS=ON, KMIDI_BUILD_JUCE_UI=ON, ENABLE_ONNX_RUNTIME=ON
- Build time: [X minutes]
- Build issues: [any issues encountered and how resolved]

## AU Validation

- `auval` result: [PASS/FAIL/partial]
- Issues: [any validation failures]

## Logic Pro Test

- Plugin loads: [YES/NO]
- Audio input received: [YES/NO]
- Emotion parameters animate: [YES/NO]
- MIDI output produced: [YES/NO]
- Manual slider mode works: [YES/NO]

## Latency Observations

- Perceived latency: [immediate / noticeable / sluggish]
- Parameter update rate: [smooth / stepped / erratic]

## Issues Found

1. [Issue description and severity]

## Next Steps

- [What needs to happen next based on what we learned]
```

- [ ] **Step 3: Commit**

```bash
git add docs/demo/2026-03-31-proof-of-life.md
git commit -m "docs: add proof-of-life demo report template"
```

Fill in actual results after running the demo.
