# KmiDi — Project Summary & Known Issues

**Date:** 2026-02-24
**Repository:** [sburdges-eng/KmiDi](https://github.com/sburdges-eng/KmiDi)
**Version:** 1.0.0 (Python package) / 0.1.0 (C++ Kelly core)

---

## Table of Contents

1. [What Is KmiDi?](#what-is-kmidi)
2. [Architecture Overview](#architecture-overview)
3. [Repository Layout](#repository-layout)
4. [Technology Stack](#technology-stack)
5. [Known Problems & Issues](#known-problems--issues)
6. [Code Statistics](#code-statistics)
7. [Key Code Samples](#key-code-samples)
8. [Build & Run Instructions](#build--run-instructions)
9. [PR / Branch History Summary](#pr--branch-history-summary)
10. [Repo Size & 100 MB Budget](#repo-size--100-mb-budget)

---

## What Is KmiDi?

KmiDi (also referred to as "Kelly") is a **unified music intelligence and audio workstation** that converts raw emotional intent into production-ready MIDI and audio. Its core philosophy is "Interrogate Before Generate" — every creative decision must be rooted in a stated emotional need, and human imperfection (timing drift, pitch wobble, velocity variance) is treated as a feature, not a bug.

### Core Concept

A user describes an emotion or narrative (e.g., "the feeling of losing someone but finding peace"), and KmiDi:
1. **Interrogates** the intent through a three-phase schema (Core Wound → Emotional Mapping → Technical Constraints).
2. **Maps** the emotion to production decisions (drum style, dynamics, arrangement density, tempo, groove).
3. **Generates** MIDI across multiple engines (bass, melody, pads, strings, rhythm, fills, transitions, etc.).
4. **Humanizes** the output so it feels played by a real musician, not quantized by a machine.

### The "Kelly Companion"

Kelly is the AI companion that lives inside the workstation. She processes emotional language, suggests intentional rule-breaks (parallel fifths for rawness, unresolved dissonance for tension), and guides the user from feeling → finished arrangement.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│  Side A — C++ Real-Time Engine (penta_core)              │
│  Lock-free, no heap allocations on audio thread           │
│  GrooveEngine, HarmonyEngine, ChordAnalyzer, MixerEngine │
│  SIMD (AVX2) with scalar fallback                         │
├────────────────────── ring buffer ────────────────────────┤
│  Side B — Python AI + UI (music_brain)                    │
│  Emotion thesaurus, intent schema, production mapper      │
│  FastAPI REST API at 127.0.0.1:8000                       │
│  Kelly Companion engines (bass, melody, pads, strings…)   │
├──────────────────────────────────────────────────────────┤
│  Desktop Shell — Tauri v2 + React/Vite                    │
│  Rust → C++ FFI bridge to KellyBrain                      │
│  Web UI → Tauri commands → HTTP to Music Brain API        │
└──────────────────────────────────────────────────────────┘
```

### Data Flow

```
User emotion text
    → IntentProcessor.process_intent()
        → EmotionThesaurus.find_emotion()
        → EmotionProductionMapper → ProductionPreset
        → HarmonyGenerator → chord progressions
        → Kelly Engines (bass, melody, rhythm, pads, strings, fills, transitions)
        → GrooveHumanizer → timing/velocity drift
    → MIDI output (multi-track, per-channel)
    → Optional: C++ penta_core real-time processing
    → Optional: Tauri desktop app playback
```

---

## Repository Layout

```
KmiDi/
├── music_brain/              # 7.1 MB — Python AI + music intelligence
│   ├── __init__.py           #   Package root, re-exports CompleteSongIntent
│   ├── api.py                #   FastAPI REST wrapper (DAiW API)
│   ├── session/              #   Intent schema, interrogation, rule-breaks
│   │   ├── intent_schema.py  #   Three-phase intent model + rule-break enums
│   │   └── intent_processor.py
│   ├── emotion/              #   Emotion thesaurus + production mapper
│   │   ├── emotion_thesaurus.py
│   │   └── emotion_production.py
│   ├── groove/               #   Groove templates, humanizer, presets
│   ├── harmony/              #   Harmony generation
│   ├── audio/                #   Audio analysis + rendering
│   ├── structure/            #   Chord analysis, section detection
│   ├── kelly_companion/      #   13 generation engines
│   │   └── engines/          #   bass, melody, pads, strings, rhythm, fills…
│   ├── voice/                #   AutoTune, voice modulation
│   └── orchestrator/         #   Multi-engine coordination
│
├── src_penta-core/           # 212 KB — C++ real-time audio engine
│   ├── groove/               #   GrooveEngine, OnsetDetector, TempoEstimator
│   ├── harmony/              #   HarmonyEngine, ChordAnalyzer, VoiceLeading
│   ├── mixer/                #   MixerEngine
│   ├── ml/                   #   MLInterface (ONNX/RTNeural)
│   ├── diagnostics/          #   PerformanceMonitor, AudioAnalyzer
│   ├── common/               #   RTMemoryPool, RTLogger
│   └── osc/                  #   OSC messaging (hub, client, server)
│
├── include/penta/            # 536 KB — C++ public headers
│
├── bindings/                 # 48 KB — pybind11 C++/Python bridge
│
├── src-tauri/                # 200 KB — Tauri v2 desktop shell (Rust)
│   └── src/
│       ├── main.rs           #   App entry, command registration
│       ├── commands.rs       #   KellyBrain FFI commands
│       ├── bridge.rs         #   C++ FFI bridge
│       └── state.rs          #   Reactive state management
│
├── src/                      # 4.2 MB — Web UI (React/TypeScript) + legacy C++
│   ├── components/           #   React UI components
│   ├── audio/                #   C++ audio processing
│   ├── plugin/               #   VST3/CLAP plugin code
│   └── ml/                   #   ML model integration (C++)
│
├── KmiDi_FINAL/              # 90 MB — Integrated build (pure DSP, JUCE plugins)
│   └── engine/src/dsp/       #   Pure DSP core (framework-independent)
│
├── KmiDi/                    # 47 MB — Nested project structure (JUCE, engine)
│
├── projects/musicgen-local/  # ML training infrastructure, JUCE plugin refs
│
├── tests/                    # 208 KB — Python test suites
├── docs/                     # 676 KB — Architecture, design, analysis docs
├── examples/                 # Example scripts (harmony, groove, integration)
├── training/                 # ML training scripts and configs
├── scripts/                  # Build, setup, and utility scripts
├── config/                   # Build variant configs (dev-mac, train-nvidia)
│
├── CMakeLists.txt            # Top-level CMake (C++ build, JUCE, pybind11)
├── pyproject.toml            # Python package config
├── package.json              # Node.js/Vite/Tailwind
└── vite.config.ts            # Vite build config
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Python AI** | Python 3.9+, NumPy, PyTorch, librosa, scipy | Music intelligence, emotion mapping, MIDI generation |
| **REST API** | FastAPI + Uvicorn | HTTP interface at `127.0.0.1:8000` |
| **C++ Engine** | C++20, CMake 3.27+, Ninja | Real-time audio processing, groove analysis, harmony |
| **Audio Framework** | JUCE 8 | Plugin hosting (VST3/CLAP), audio I/O |
| **Desktop Shell** | Tauri v2 (Rust) | Native app wrapper, FFI to C++ |
| **Web UI** | React, Vite, Tailwind CSS | Frontend interface |
| **ML Inference** | ONNX Runtime, RTNeural (optional) | On-device model inference |
| **Python Bindings** | pybind11 | Bridge C++ engine to Python |

---

## Known Problems & Issues

### 1. JUCE Dependency Not Bundled (Build Blocker)

**Severity:** 🔴 Critical — blocks full C++ build

The `external/JUCE` directory is expected by CMakeLists.txt but is not present in the repository. JUCE 8 must be cloned separately.

**Impact:** `GrooveEngine.cpp` and all JUCE-dependent C++ code fails to compile.

**Error:**
```
add_subdirectory given source "external/JUCE" which is not an existing directory
fatal error: juce_dsp/juce_dsp.h: No such file or directory
```

**Workaround:**
```bash
git clone --depth 1 --branch 8.0.0 \
  https://github.com/juce-framework/JUCE.git external/JUCE
```

---

### 2. API–UI Feature Gap (Generation Pipeline Mismatch)

**Severity:** 🔴 Critical — UI collects rich data that the backend ignores

The frontend collects detailed parameters (duration, song structure, instruments, techniques, genre), but the `/generate` API endpoint uses a simple `therapy_session()` that only accepts `{ key, bpm, progression, genre }`.

**Root cause:** The full `CompleteSongIntent` pipeline exists and works, but the `/generate` endpoint was shipped with an MVP shortcut and never reconnected.

**What's lost:**
- Duration not respected
- Song structure (sections, repetitions) ignored
- Instrument selection ignored (no multi-track MIDI)
- Production techniques ignored
- Audio rendering missing (MIDI only, no WAV/MP3)

**Fix needed:** Wire `/generate` → `CompleteSongIntent.from_ui_payload()` → `process_song_intent()`.

---

### 3. Input Validation Vulnerabilities (Crash Risk)

**Severity:** 🟠 High — multiple crash paths from malformed input

Several input validation issues have been identified across `api.py` and `intent_schema.py`. Some were previously documented but have since been mitigated; this section tracks remaining concerns and notes past fixes that should be re-verified over time.

- **Resolved `KeyError` crashes (historical):** Earlier versions of `CompleteSongIntent.from_dict()` accessed dictionary fields with `data["key"]`, which could raise `KeyError` for malformed payloads. The current implementation uses `data.get(...)` throughout, so this specific crash path should now be eliminated.
- **Resolved unbounded `mood_secondary_tension` (historical):** `mood_secondary_tension` used to accept arbitrary float values (e.g., `999.0`) despite the spec requiring `[0.0, 1.0]`. The current implementation clamps `mood_secondary_tension` to this range; callers should still treat it as a normalized value.
- **Type coercion failures (to re-verify):** `request.intent.technical.bpm / 20` can still crash with `TypeError` if BPM arrives as a string or non-numeric type. We should either coerce BPM safely or validate and surface a clear error before arithmetic.
- **Missing result keys (to re-verify):** Accessing `result['intent_summary']` and `result['harmony'].chords` assumes a fully populated dict; if `process_intent()` returns a partial result, these lookups may still raise errors. Defensive `.get(...)` usage or structured result types would harden this path.

Historical note: Earlier versions reproduced a `KeyError` by calling `CompleteSongIntent.from_dict({"title": "Test"})`. With the newer `.get(...)` usage and clamping logic in `intent_schema.py`, that specific crash should no longer occur, but any new callers should still validate external input before passing it into the intent layer.

---

### 4. pybind11 CMake Detection Failure

**Severity:** 🟡 Medium — Python bindings don't build without manual intervention

pybind11 is installed via pip but CMake's `find_package(pybind11)` can't locate it.

**Error:** `pybind11 not found. Python bindings will not be built.`

**Workaround:**
```bash
cmake -B build -S . \
  -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
```

---

### 5. Background Task Circular Dependency (Tauri/Rust)

**Severity:** 🟡 Medium — background state sync disabled

The Tauri background task tried to call Tauri commands from within itself, creating a circular dependency. Currently commented out.

**Impact:** State updates only happen via explicit user-triggered commands, not automatic background sync.

**Fix needed:** Use FFI directly from background tasks instead of routing through Tauri command handlers.

---

### 6. Tauri Icon Generation Issues

**Severity:** 🟢 Fixed — was blocking Tauri compilation

Missing PNG icon files caused `proc macro panicked` errors during Rust compilation. Fixed by extracting PNGs from the existing `.icns` file.

---

### 7. macOS SDK 26.2+ Compatibility (__CLOCK_AVAILABILITY)

**Severity:** 🟡 Medium — affects macOS builds with newer SDKs

The system header `_time.h` uses `__CLOCK_AVAILABILITY` which requires specific availability macros. A workaround is in CMakeLists.txt that defines these macros to empty values.

**Relevant CMake snippet:**
```cmake
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum macOS version")
    add_compile_definitions(
        MAC_OS_X_VERSION_MIN_REQUIRED=101200
        __CLOCK_AVAILABILITY=
        __WATCHOS_PROHIBITED=
        __TVOS_PROHIBITED=
        __IOS_PROHIBITED=
    )
endif()
```

---

### 8. DSP Core Purity / Framework Contamination

**Severity:** 🟡 Medium — architectural boundary violation

Some C++ files in `src/audio/`, `src/engine/`, `src/ml/` contain JUCE dependencies that should be pure DSP. The guideline is: "If I delete JUCE tomorrow, does this file still make sense?"

A pure DSP core exists in `KmiDi_FINAL/engine/src/dsp/` (zero JUCE dependencies) but the main `src/` tree hasn't been fully migrated.

---

### 9. PR Proliferation & Duplicate Branches

**Severity:** 🟡 Medium — repo hygiene

The repository has 37 branches and 74+ PRs. Many sub-PRs (e.g., #63–#72) are iterative review fragments targeting the same reconciliation script. Only the latest superset (#72) is needed; the rest are redundant.

**Recommendation:** Keep parent PRs to `main` (#73, #62, #61, #60, #74), close duplicate sub-PRs.

---

### 10. PR #60 Blocking Bugs (musicgen-local)

**Severity:** 🟠 High — 3 P1 bugs prevent merge of ML training infrastructure

1. **Broken imports:** Legacy training scripts reference `src.*` modules that don't exist → `ModuleNotFoundError` on launch.
2. **ZeroDivisionError:** Validation with small/empty datasets causes division by zero.
3. **Hanging MIDI notes:** JUCE plugin doesn't emit note-offs when stopping playback.

---

### 11. Build Determinism (FetchContent Network Pulls)

**Severity:** 🟢 Low — affects reproducibility, not functionality

The top-level CMakeLists.txt uses `FetchContent` to download `readerwriterqueue`, `RTNeural`, and `googletest` at configure time. Builds require network access and may produce different results if upstream tags change.

---

## Code Statistics

| Metric | Value |
|---|---|
| **Total code files** (`.py`, `.cpp`, `.h`, `.hpp`, `.ts`, `.tsx`, `.rs`, `.swift`) | 5,007 |
| **Total lines of code** | ~1,515,932 |
| **Total code file size** | ~52 MB |
| **Total repo working tree** (excluding `.git`) | ~384 MB |
| **Python files** | Majority of music_brain, tests, scripts, training |
| **C++ files** | penta_core engine, JUCE plugins, DSP |
| **Rust files** | Tauri desktop shell |
| **TypeScript/React files** | Web UI |

---

## Key Code Samples

### Intent Schema — Three-Phase Interrogation (Python)

The core data model that drives everything. Defines rule-break enums for intentional creative choices:

```python
# music_brain/session/intent_schema.py

class HarmonyRuleBreak(Enum):
    """Harmony rules to intentionally break."""
    AVOID_TONIC_RESOLUTION = "HARMONY_AvoidTonicResolution"
    PARALLEL_MOTION = "HARMONY_ParallelMotion"
    MODAL_INTERCHANGE = "HARMONY_ModalInterchange"
    TRITONE_SUBSTITUTION = "HARMONY_TritoneSubstitution"
    POLYTONALITY = "HARMONY_Polytonality"
    UNRESOLVED_DISSONANCE = "HARMONY_UnresolvedDissonance"

class RhythmRuleBreak(Enum):
    CONSTANT_DISPLACEMENT = "RHYTHM_ConstantDisplacement"
    TEMPO_FLUCTUATION = "RHYTHM_TempoFluctuation"
    METRIC_MODULATION = "RHYTHM_MetricModulation"
    POLYRHYTHMIC_LAYERS = "RHYTHM_PolyrhythmicLayers"
    DROPPED_BEATS = "RHYTHM_DroppedBeats"

class VulnerabilityScale(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class NarrativeArc(Enum):
    CLIMB_TO_CLIMAX = "Climb-to-Climax"
    SLOW_REVEAL = "Slow Reveal"
    REPETITIVE_DESPAIR = "Repetitive Despair"
    STATIC_REFLECTION = "Static Reflection"
    SUDDEN_SHIFT = "Sudden Shift"
    DESCENT = "Descent"
    RISE_AND_FALL = "Rise and Fall"
    SPIRAL = "Spiral"
```

### Emotion → Production Mapper (Python)

Maps emotions to concrete production decisions:

```python
# music_brain/emotion/emotion_production.py

@dataclass
class ProductionPreset:
    drum_style: str = "standard"
    dynamics_level: str = "mf"
    arrangement_density: float = 0.5
    intensity_tier: Optional[int] = None
    tempo_range: Tuple[int, int] = (100, 120)
    feel: str = "straight"
    swing: float = 0.0
    groove_motif: str = "backbeat"
    kit_hint: str = "standard kit"
    section_dynamics: Dict[str, str] = field(default_factory=dict)
    section_density: Dict[str, float] = field(default_factory=dict)
    fx: Dict[str, str] = field(default_factory=dict)
    transitions: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)
```

### C++ GrooveEngine — Real-Time Audio (C++)

RT-safe audio processing with no heap allocations on the audio thread:

```cpp
// src_penta-core/groove/GrooveEngine.cpp

void GrooveEngine::processAudio(const float *buffer, size_t frames) noexcept
{
    if (onsetDetector_)
    {
        onsetDetector_->process(buffer, frames);

        if (onsetDetector_->hasOnset())
        {
            uint64_t onsetPos = onsetDetector_->getOnsetPosition();
            float onsetStrength = onsetDetector_->getOnsetStrength();

            constexpr size_t kMaxOnsetHistory = 128;
            auto pushBounded = [](auto &vec, const auto &value)
            {
                if (vec.size() < kMaxOnsetHistory)
                {
                    vec.push_back(value);
                    return;
                }
                // Shift left by one and append at end (fixed O(N), no alloc).
                std::move(vec.begin() + 1, vec.end(), vec.begin());
                vec.back() = value;
            };

            pushBounded(analysis_.onsetPositions, onsetPos);
            pushBounded(analysis_.onsetStrengths, onsetStrength);
            pushBounded(onsetHistory_, onsetPos);
        }
    }
    samplePosition_ += frames;
}
```

### C++ GrooveEngine Header — Public API

```cpp
// include/penta/groove/GrooveEngine.h

class GrooveEngine {
public:
    struct Config {
        double sampleRate;
        size_t hopSize;
        float minTempo, maxTempo;
        bool enableQuantization;
        float quantizationStrength;
    };

    struct GrooveAnalysis {
        float currentTempo;
        float tempoConfidence;
        std::vector<uint64_t> onsetPositions;
        std::vector<float> onsetStrengths;
        uint32_t timeSignatureNum, timeSignatureDen;
        float swing;  // 0.0 = straight, 1.0 = maximum swing
    };

    // RT-safe: Process audio buffer for groove analysis
    void processAudio(const float* buffer, size_t frames) noexcept;

    // RT-safe: Quantize timestamp to grid
    uint64_t quantizeToGrid(uint64_t timestamp) const noexcept;

    // RT-safe: Get swing-adjusted position
    uint64_t applySwing(uint64_t position) const noexcept;
};
```

### Tauri Desktop Shell (Rust)

```rust
// src-tauri/src/main.rs

fn main() {
    dotenv::dotenv().ok();

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            kelly_brain_initialize,
            kelly_brain_is_initialized,
            kelly_brain_from_text,
            kelly_brain_from_emotion,
            kelly_brain_generate_midi,
            kelly_brain_generate_midi_with_params,
            kelly_brain_get_emotion_state,
            kelly_brain_set_emotion_parameters,
            kelly_brain_get_available_emotions,
            kelly_brain_get_version,
            generate_music,
            interrogate,
            get_emotions,
        ])
        .run(tauri::generate_context!())
        .expect("error while running application");
}
```

### Python Package Entry Point

```python
# music_brain/__init__.py

"""
music_brain - Music Intelligence Toolkit
"""

__version__ = "0.2.0"

from .kelly_companion.session.intent_schema import (
    CompleteSongIntent,
    SongIntent,
    SongRoot,
    TechnicalConstraints,
    SystemDirective,
)
```

### FastAPI REST Interface

```python
# music_brain/api.py

from fastapi import FastAPI, HTTPException
from music_brain.session.intent_schema import CompleteSongIntent, suggest_rule_break
from music_brain.session.intent_processor import process_intent
from music_brain.emotion.emotion_thesaurus import EmotionThesaurus

# Endpoints: /emotions, /generate, /interrogate, /diagnose
```

### CI Workflow

```yaml
# .github/workflows/tests.yml

name: Tests
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.11']
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
    - run: pip install numpy scipy torch librosa pyyaml pytest pytest-cov
    - run: pip install -e .
    - run: pytest tests/unit/ -v --cov=music_brain
```

---

## Build & Run Instructions

### Python (music_brain)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .               # Install package
pip install -e ".[dev]"        # + dev tools (pytest, black, flake8, mypy)
pytest tests/ -v               # Run tests
python -m music_brain.api      # Start API at 127.0.0.1:8000
```

### C++ (penta_core)

```bash
# Prerequisite: clone JUCE
git clone --depth 1 --branch 8.0.0 \
  https://github.com/juce-framework/JUCE.git external/JUCE

# Configure and build
cmake -S . -B build -G Ninja \
  -DBUILD_TESTS=ON \
  -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build
ctest --test-dir build --output-on-failure
```

### Desktop App (Tauri)

```bash
npm install
npm run tauri dev    # Development mode
npm run tauri build  # Production build
```

### CLI Commands

```bash
daiw intent suggest grief
daiw diagnose "F-C-Am-Dm"
python -m mcp_workstation status
python -m mcp_todo.cli list
```

---

## PR / Branch History Summary

The repository has evolved through **74+ pull requests** across **37 branches**, covering:

- **Core infrastructure** (PRs #1–#12): Initial consolidation, penta_core, training pipeline, MCP server.
- **Feature development** (PRs #20–#30): Dynamics engine, MLX workflow, standalone app, MIDI backlog.
- **Multimodal emotion** (PRs #25–#29, #41–#42): Emotion-driven arrangement, drum humanizer presets.
- **Build fixes** (PRs #33–#40, #45, #54–#55): Icon generation, branch merges, compilation, input validation.
- **Architecture** (PRs #44, #46–#47, #51–#53): Stem-JEPA integration, video generation stubs, code restructuring.
- **Recovery & reconciliation** (PRs #58, #62–#72): Lost code recovery, file reconciliation from multiple sources.
- **ML training** (PR #60): musicgen-local with rulebreak-aware inventory (has 3 P1 blocking bugs).
- **Audit & hardening** (PRs #73–#74): Pre-training audit, merge analysis.

### Currently Open PRs

| PR | Title | Status |
|---|---|---|
| #74 | Analyze and merge improvements from all branches | WIP |
| #73 | Pre-training recursive hardening audit | Pending review |
| #62 | Recovery: Reconcile 5,876 files | Pending (with sub-PRs #63–#72) |
| #61 | Analyze open PRs for merge readiness | Pending review |
| #60 | musicgen-local training infrastructure | Blocked on 3 P1 bugs |

---

## Repo Size & 100 MB Budget

| Component | Size |
|---|---|
| Working tree (excluding `.git`) | ~384 MB |
| `.git` directory | ~141 MB |
| **Total** | **~525 MB** |
| Code files only (`.py`, `.cpp`, `.h`, `.ts`, `.rs`, etc.) | ~52 MB |
| `KmiDi_FINAL/` (integrated build artifacts) | ~90 MB |
| `KmiDi/` (nested project) | ~47 MB |
| `music_brain/` | ~7.1 MB |
| `src/` | ~4.2 MB |
| All other directories | <5 MB each |

The **actual source code** (all `.py`, `.cpp`, `.h`, `.hpp`, `.ts`, `.tsx`, `.rs`, `.swift` files) totals **~52 MB across 5,007 files** (~1.5 million lines of code), well within a 100 MB budget. The remainder is build artifacts, documentation, training data, and the nested `KmiDi_FINAL` integration directory.
