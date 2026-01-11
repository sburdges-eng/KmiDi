# iDAW Comprehensive To-Do List

> Updated: 2025-12-04 | Version: 0.2.1 (Alpha)

This document provides a complete roadmap of tasks for the iDAW project, organized by priority and component.

> **Note**: See `REFINED_PRIORITY_PLANS.md` for detailed, actionable implementation plans.

---

## 📊 Project Status Overview

| Component | Status | Completion |
|-----------|--------|------------|
| DAiW-Music-Brain (Python) | ✅ Functional | ~92% |
| Penta-Core (C++ RT Engines) | ✅ Functional | ~85% |
| iDAW_Core (JUCE Plugins) | ✅ **Complete** | ~95% |
| Python/C++ Bridge | ✅ **Complete** | ~100% |
| Therapy System | ✅ Functional | ~95% |
| MCP Workstation | ✅ Functional | ~95% |
| Test Suite | 🔴 **Gaps Found** | ~50% |
| Documentation | ✅ Extensive | ~90% |

---

## 🔴 HIGH PRIORITY - Core Functionality

### 1. JUCE Plugin DSP Implementations
**Status**: ✅ **COMPLETE** - All 11 plugins have full DSP implementations

| Plugin | Lines (H/C) | DSP Features | Status |
|--------|-------------|--------------|--------|
| **Pencil** | 267/411 | 3-band tube saturation, 2nd harmonic generation | ✅ |
| **Eraser** | 280/462 | 2048-FFT spectral gating, 75% overlap, Hann window | ✅ |
| **Press** | 279/415 | RMS compression, soft knee, auto makeup gain | ✅ |
| **Palette** | 239/432 | Dual-osc wavetable, 8-voice poly, FM matrix, SVF | ✅ |
| **Parrot** | 315/879 | 4096-FFT pitch detection, YIN, harmony gen, vocoder | ✅ |
| **Smudge** | 175/445 | 1024-FFT convolution reverb, partitioned IR | ✅ |
| **Trace** | 172/251 | Circular delay, ping-pong, tape saturation, LFO | ✅ |
| **Brush** | 199/383 | SVF 4-mode filter, 6 LFO shapes, envelope follower | ✅ |
| **Chalk** | 179/391 | Bitcrusher, sample-rate reduction, vinyl crackle | ✅ |
| **Stencil** | 172/291 | Sidechain ducking, 3 modes (ext/LFO/MIDI) | ✅ |
| **Stamp** | 195/456 | Stutter/repeat, reverse, ping-pong, pitch shift | ✅ |

**Location**: `iDAW_Core/plugins/*/`

**Completed Tasks**:
- [x] ~~Implement Eraser DSP~~ Complete (spectral gating with JUCE FFT)
- [x] ~~Implement Press DSP~~ Complete (VCA compressor with soft knee)
- [x] ~~Implement Palette DSP~~ Complete (wavetable synth with FM)
- [x] ~~Implement Smudge DSP~~ Complete (convolution reverb)
- [x] ~~Implement Trace DSP~~ Complete (modulated delay with BPM sync)
- [x] ~~Implement Parrot DSP~~ Complete (pitch detection + harmony generation)
- [x] ~~Add JUCE parameter automation~~ Complete (AudioProcessorValueTreeState)
- [x] ~~Create shader effects~~ Complete (11 unique OpenGL shaders)

---

### 2. FFT Library Integration for Production
**Status**: 🟡 70% Complete - JUCE FFT working, OnsetDetector needs upgrade

| Component | Library | Status |
|-----------|---------|--------|
| Eraser Plugin | `juce::dsp::FFT` | ✅ Working (2048 FFT) |
| Smudge Plugin | `juce::dsp::FFT` | ✅ Working (1024 FFT) |
| Parrot Plugin | `juce::dsp::FFT` | ✅ Working (4096 FFT) |
| Python Analysis | `librosa.stft()` | ✅ Working |
| **OnsetDetector** | Filterbank stub | ❌ **Needs real FFT** |
| **Phase Vocoder** | Declared only | ❌ **Not implemented** |

**Remaining Tasks**:
- [x] ~~Choose FFT library~~ Using JUCE FFT (already in build)
- [ ] Update OnsetDetector to use `juce::dsp::FFT` for spectral flux
- [ ] Implement Phase Vocoder in `python/penta_core/dsp/parrot_dsp.py`
- [ ] Benchmark OnsetDetector FFT (target: < 200μs latency)
- [x] ~~Hann windowing~~ Already implemented in plugins

---

### 3. Test Suite Gaps
**Status**: 🔴 49.8% Coverage - Major gaps identified

| Component | Test LOC | Source LOC | Coverage | Priority |
|-----------|----------|------------|----------|----------|
| iDAW_Core (Plugins) | 0 | 4,816 | **0%** | 🔴 CRITICAL |
| ML Module | 0 | 2,210 | **0%** | 🔴 CRITICAL |
| Collaboration | 0 | 1,433 | **0%** | 🟡 MEDIUM |
| DSP Module | 0 | 1,130 | **0%** | 🔴 HIGH |
| Music Brain Core | ~12,000 | 15,000 | **77%** | ✅ Good |
| Penta-Core C++ | 1,815 | 3,120 | **58%** | 🟡 MEDIUM |

**Critical Tasks**:
- [ ] Create JUCE plugin test harness (`PluginTestHarness.h`)
- [ ] Add RT-safety verification (no allocations in processBlock)
- [ ] Add plugin DSP accuracy tests (compression ratio, FFT accuracy)
- [ ] Add ML module test coverage (inference, style transfer)
- [ ] Add DSP module tests (pitch detection, phase vocoder)
- [ ] Add Valgrind memory testing to CI
- [ ] Add performance regression tests (< 100μs harmony, < 200μs groove)

---

## 🟡 MEDIUM PRIORITY - Enhancement & Integration

### 4. Python/C++ Bridge Completion
**Status**: ✅ **COMPLETE** - Production-ready

| Component | Status | Evidence |
|-----------|--------|----------|
| pybind11 bindings | ✅ Complete | `bindings/*.cpp` - all 4 modules |
| Python wrapper API | ✅ Complete | `python/penta_core/__init__.py` (326 lines) |
| C++ PythonBridge | ✅ Complete | `iDAW_Core/include/PythonBridge.h` |
| Bridge API | ✅ Complete | `music_brain/orchestrator/bridge_api.py` (678 lines) |
| OSC communication | ✅ Complete | Documented in `vault/Production_Workflows/` |
| Integration tests | ✅ Complete | 11/11 passing |

**Completed Tasks**:
- [x] ~~Complete Python bindings~~ All 4 modules wrapped
- [x] ~~pybind11 wrappers for GrooveEngine~~ Complete
- [x] ~~pybind11 wrappers for HarmonyEngine~~ Complete
- [x] ~~pybind11 wrappers for DiagnosticsEngine~~ Complete
- [x] ~~pybind11 wrappers for OSCHub~~ Complete
- [x] ~~Create integration tests~~ 11/11 passing

**Remaining**:
- [ ] Document Python API with usage examples

---

### 5. Therapy/Chatbot Integration
**Status**: ✅ 95% Complete - Therapy-to-music compiler (not a chatbot)

| Component | Status | Location |
|-----------|--------|----------|
| Affect Analyzer | ✅ Complete | `music_brain/structure/comprehensive_engine.py` |
| Therapy Session | ✅ Complete | `music_brain/structure/comprehensive_engine.py` |
| Song Interrogator | ✅ Complete | `interrogator.py` (7 phases) |
| Intent Schema | ✅ Complete | `music_brain/session/intent_schema.py` |
| Rule-Breaking System | ✅ Complete | `music_brain/session/teaching.py` |
| MIDI Rendering | ✅ Complete | `render_plan_to_midi()` |
| MCP Tool | ✅ Complete | `therapy.py` → `daiw.therapy.session` |
| Optional Ollama | ✅ Available | `music_brain/agents/unified_hub.py` |

**Completed Tasks**:
- [x] ~~Define chatbot service API~~ Uses therapy session API
- [x] ~~Intent-to-chat translation~~ Via AffectAnalyzer
- [x] ~~Local LLM integration~~ Ollama support available

**Optional Enhancements**:
- [ ] Add session save/load persistence
- [ ] Add real-time Ollama streaming
- [ ] Complete voice synthesis profiles

---

### 6. CI/CD Pipeline Improvements
**Status**: Basic CI working, could be enhanced

**Tasks**:
- [ ] Add C++ build to main CI workflow
- [ ] Add Valgrind memory testing stage
- [ ] Add performance regression testing
- [ ] Add code coverage reporting (lcov for C++, coverage.py for Python)
- [ ] Add automated release builds for all platforms
- [ ] Add JUCE plugin validation (auval for macOS)

---

### 7. Penta-Core Optimization (Phase 3.5)
**Status**: Functional but not fully optimized

**Tasks**:
- [ ] Profile hot paths with Instruments (macOS) / perf (Linux)
- [ ] Implement SIMD kernels for chord pattern matching (AVX2)
- [ ] Implement SIMD kernels for RMS calculation
- [ ] Implement SIMD kernels for FFT preprocessing
- [ ] Implement SIMD kernels for autocorrelation
- [ ] Add scalar fallbacks for non-SIMD systems
- [ ] Verify < 100μs harmony latency @ 48kHz/512 samples
- [ ] Verify < 200μs groove latency @ 48kHz/512 samples

---

## 🟢 LOW PRIORITY - Polish & Future Features

### 8. Documentation & Tutorials
**Status**: Extensive but could be enhanced

**Tasks**:
- [ ] Generate C++ API documentation with Doxygen
- [ ] Create video tutorials for DAiW CLI
- [ ] Write migration guide from v0.1 to v0.2
- [ ] Add more intent schema examples (beyond Kelly song)
- [ ] Document PythonBridge usage with examples
- [ ] Create "Getting Started" guide for contributors

---

### 9. Desktop Application
**Status**: Streamlit UI exists, native wrapper incomplete

**Tasks**:
- [ ] Complete PyWebView wrapper for native desktop
- [ ] Add system tray integration (daiw_menubar.py)
- [ ] Create macOS .app bundle with PyInstaller
- [ ] Create Windows .exe installer
- [ ] Create Linux AppImage
- [ ] Add auto-update mechanism

---

### 10. DAW Integration Testing
**Status**: Logic Pro integration exists, needs expansion

**Tasks**:
- [ ] Test Logic Pro integration with real projects
- [ ] Add Ableton Live integration (via OSC)
- [ ] Add Reaper integration (via OSC)
- [ ] Add Pro Tools integration (via AAX format)
- [ ] Document DAW-specific setup instructions
- [ ] Create DAW template projects

---

### 11. Mobile/Web Expansion
**Status**: Streamlit web UI works, mobile not started

**Tasks**:
- [ ] Deploy Streamlit app to cloud (Streamlit Cloud / Railway)
- [ ] Create PWA wrapper for mobile access
- [ ] Evaluate React Native or Flutter for native mobile
- [ ] Create iOS Audio Unit version of plugins
- [ ] Create Android AAP version of plugins

---

## 🔵 FUTURE ENHANCEMENTS (Nice to Have)

### 12. ML Model Integration
**Status**: Not started, architecture supports it

**Tasks**:
- [ ] Evaluate real-time ML inference frameworks (ONNX Runtime, TensorFlow Lite)
- [ ] Design ML model interface for penta-core
- [ ] Implement chord prediction model
- [ ] Implement style transfer model for groove
- [ ] Add GPU acceleration option (CUDA/Metal)

---

### 13. Advanced Harmony Features
**Status**: Basic implementation complete

**Tasks**:
- [ ] Add jazz voicing generation
- [ ] Implement neo-Riemannian transformations
- [ ] Add counterpoint generation
- [ ] Implement tension/release analysis
- [ ] Add microtonal support (24-TET, just intonation)

---

### 14. Advanced Groove Features
**Status**: Basic implementation complete

**Tasks**:
- [ ] Add polyrhythm detection
- [ ] Implement groove DNA extraction (like The Pocket Queen)
- [ ] Add humanization presets by artist/style
- [ ] Implement live performance timing analysis
- [ ] Add drum replacement with timing preservation

---

### 15. Collaboration Features
**Status**: MCP multi-AI exists, user collaboration not started

**Tasks**:
- [ ] Design real-time collaboration protocol
- [ ] Implement session sharing via WebSocket
- [ ] Add version control for song intents
- [ ] Create collaborative editing UI
- [ ] Add comment/annotation system

---

## 📋 Task Summary by Component

### Python (DAiW-Music-Brain)
| Task | Priority | Status |
|------|----------|--------|
| Python bindings (pybind11) | MEDIUM | ✅ Complete |
| Therapy system integration | MEDIUM | ✅ Complete |
| Document Python API with examples | LOW | Pending |
| Desktop app polish | LOW | Pending |
| More intent examples | LOW | Pending |

### C++ (Penta-Core)
| Task | Priority | Status |
|------|----------|--------|
| OnsetDetector FFT upgrade | HIGH | Pending |
| Phase Vocoder implementation | HIGH | Pending |
| SIMD optimization | MEDIUM | Pending |
| Memory testing (Valgrind) | MEDIUM | Pending |
| Performance benchmarks | MEDIUM | Pending |

### C++ (iDAW_Core - JUCE)
| Task | Priority | Status |
|------|----------|--------|
| Pencil DSP | HIGH | ✅ Complete |
| Eraser DSP | HIGH | ✅ Complete |
| Press DSP | HIGH | ✅ Complete |
| Palette DSP | MEDIUM | ✅ Complete |
| Smudge DSP | MEDIUM | ✅ Complete |
| Trace DSP | LOW | ✅ Complete |
| Parrot DSP | LOW | ✅ Complete |
| Brush DSP | LOW | ✅ Complete |
| Chalk DSP | LOW | ✅ Complete |
| Stencil DSP | LOW | ✅ Complete |
| Stamp DSP | LOW | ✅ Complete |

### Testing
| Task | Priority | Status |
|------|----------|--------|
| JUCE plugin test harness | HIGH | Pending |
| RT-safety verification | HIGH | Pending |
| ML module test coverage | HIGH | Pending |
| DSP module tests | HIGH | Pending |
| Integration tests | MEDIUM | ✅ Complete (11/11) |
| Coverage reporting | LOW | Pending |

### DevOps
| Task | Priority | Status |
|------|----------|--------|
| C++ CI build | MEDIUM | Pending |
| Memory testing CI (Valgrind) | MEDIUM | Pending |
| Performance regression CI | MEDIUM | Pending |
| Release automation | LOW | Pending |

---

## 🎯 Recommended Sprint Plan

### Sprint A: Test Coverage & Quality ✅ (Partially Complete)
**Completed:**
- ✅ All 11 JUCE plugin DSP implementations
- ✅ JUCE parameter automation (AudioProcessorValueTreeState)
- ✅ Python bindings (pybind11) for all 4 modules
- ✅ Integration tests (11/11 passing)

**Remaining:**
1. Create JUCE plugin test harness
2. Add RT-safety verification tests
3. Add ML module and DSP module test coverage

### Sprint B: Performance & FFT
1. Upgrade OnsetDetector to use `juce::dsp::FFT`
2. Implement Phase Vocoder in Python DSP module
3. Profile hot paths and identify bottlenecks
4. Implement SIMD optimizations (AVX2 with scalar fallback)
5. Benchmark: < 100μs harmony, < 200μs groove latency

### Sprint C: CI/CD & Memory Safety
1. Add C++ build to main CI workflow
2. Add Valgrind memory testing stage
3. Add performance regression testing
4. Add code coverage reporting (lcov + coverage.py)

### Sprint D: Documentation & Polish
1. Document Python API with usage examples
2. Generate C++ API docs with Doxygen
3. Complete desktop app packaging (macOS/Windows/Linux)
4. Add more intent schema examples

### Sprint E: Future Enhancements
1. ML model integration (ONNX Runtime evaluation)
2. Collaboration features (real-time session sharing)
3. Mobile/web expansion (PWA, native mobile)
4. Advanced harmony/groove features

---

## 📝 Quick Reference: File Locations

| Component | Primary Location |
|-----------|------------------|
| Python Music Brain | `DAiW-Music-Brain/music_brain/` |
| Python CLI | `DAiW-Music-Brain/music_brain/cli.py` |
| C++ Penta-Core | `src_penta-core/` |
| C++ Headers | `include/penta/` |
| JUCE Plugins | `iDAW_Core/plugins/` |
| Python Tests | `tests_music-brain/` |
| C++ Tests | `tests_penta-core/` |
| CI Workflows | `.github/workflows/` |
| Documentation | `docs_music-brain/`, `vault/` |

---

## ✅ Recently Completed (For Reference)

- ✅ All code-level TODOs resolved
- ✅ Harmony/Scale history tracking in HarmonyEngine
- ✅ Lock-free RTMessageQueue
- ✅ OSC Client/Server/Hub implementation
- ✅ OnsetDetector spectral flux
- ✅ TempoEstimator with confidence
- ✅ RhythmQuantizer with swing
- ✅ Kelly intent JSON example
- ✅ 37 CLI commands tested
- ✅ Windows TTS support
- ✅ AudioAnalyzer implementation

---

*"Interrogate Before Generate" - The tool shouldn't finish art for people. It should make them braver.*

*Last updated: 2025-12-04*
