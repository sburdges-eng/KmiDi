# KmiDi Project Completion Plan
**Created**: 2025-01-07  
**Status**: Active Development  
**Goal**: Complete product implementation with optimal architecture

---

## Executive Summary

This document provides a comprehensive TODO list organized by priority to ensure KmiDi reaches production readiness with optimal implementation. Based on comprehensive codebase analysis, the project is **~90% complete** with critical gaps in:

1. **ML Model Integration** (trained models may exist in RECOVERY_OPS - need verification/integration)
2. **ML/DSP Test Coverage** (plugin tests ✅ complete, but ML/DSP tests needed)
3. **Production Infrastructure** (CI/CD enhancements, deployment)
4. **UI Completion** (Qt GUI, Streamlit demo)

**Note**: `/Users/seanburdges/RECOVERY_OPS` contains 99% of completed project work - may include trained models, completed features, and other assets that need integration.

---

## Priority Framework

- **P0 (Critical)**: Blocks production, must fix immediately
- **P1 (High)**: Core functionality gaps, fix within 2-4 weeks
- **P2 (Medium)**: Enhancements and optimizations, fix within 1-3 months
- **P3 (Low)**: Polish and future features, fix as time permits

---

## RECOVERY_OPS Integration Priority

**Location**: `/Users/seanburdges/RECOVERY_OPS`  
**Status**: Contains 99% of completed project work  
**Action Required**: Verify and integrate assets from RECOVERY_OPS

### Items Found in RECOVERY_OPS

1. **Trained ML Models** (`/ML_TRAINED_MODELS/`)
   - `harmonypredictor_best.pt`
   - `emotionrecognizer_best.pt`
   - `melodytransformer_best.pt`
   - `groovepredictor_best.pt`
   - `dynamicsengine_best.pt`
   - Multiple checkpoint epochs

2. **Completed Code** (`/ARCHIVE/kelly-music-brain-clean/`)
   - Full project structure
   - All modules and components
   - Test infrastructure
   - Documentation

3. **Audio/MIDI Data** (`/AUDIO_MIDI_DATA/`)
   - Processed datasets
   - Training manifests
   - Audio libraries

### Integration Steps

1. **Verify model compatibility**:
   ```bash
   python scripts/validate_models.py --source /Users/seanburdges/RECOVERY_OPS/ML_TRAINED_MODELS/...
   ```

2. **Copy validated models**:
   ```bash
   cp validated_models/*.pt models/checkpoints/
   ```

3. **Check for code differences**:
   ```bash
   diff -r /Users/seanburdges/RECOVERY_OPS/ARCHIVE/kelly-music-brain-clean/ ./ --exclude='.git'
   ```

4. **Integrate missing features** from RECOVERY_OPS as needed

---

## P0: Critical - Production Blockers

### 1. ML Model Training ⚠️ **BLOCKER**

**Status**: Models appear to be stubs in current repo, but trained checkpoints found in `/Users/seanburdges/RECOVERY_OPS/ML_TRAINED_MODELS/`  
**Impact**: System cannot generate meaningful music without trained models  
**Action Required**: Verify and integrate trained models from RECOVERY_OPS  
**Effort**: 1-2 days to verify/integrate existing models, or 4-8 weeks per model to retrain if needed

#### Training Priority Matrix

| Model | Data Status | Complexity | Priority | Timeline |
|-------|-------------|------------|----------|----------|
| **HarmonyPredictor** | 🟢 High (have progressions DB) | Low (MLP) | **START HERE** | 2-3 weeks |
| **EmotionRecognizer** | 🟡 Medium (RAVDESS needed) | Medium (CNN) | **2nd** | 3-4 weeks |
| **MelodyTransformer** | 🟢 High (Lakh MIDI needed) | Medium (Transformer) | **3rd** | 4-6 weeks |
| **GroovePredictor** | 🟡 Medium (FMA needed) | Medium (RNN) | **4th** | 3-4 weeks |
| **DynamicsEngine** | 🔴 Low (needs custom) | Low (small MLP) | **5th** | 2-3 weeks |
| **InstrumentRecognizer** | 🟡 Medium (NSynth) | High (dual-head) | **Later** | 6-8 weeks |
| **EmotionNodeClassifier** | 🔴 Low (needs 6×6×6 data) | High (multi-head) | **Later** | 6-8 weeks |

#### Actions Required

**FIRST: Check RECOVERY_OPS for existing trained models**

Found in `/Users/seanburdges/RECOVERY_OPS/ML_TRAINED_MODELS/`:
- `emotionrecognizer_best.pt`
- `harmonypredictor_best.pt`
- `melodytransformer_best.pt`
- `groovepredictor_best.pt`
- `dynamicsengine_best.pt`

**Verify and integrate**:
```bash
# Check model files in RECOVERY_OPS
ls -lh /Users/seanburdges/RECOVERY_OPS/ML_TRAINED_MODELS/Desktop/*/ml_training/trained_models/checkpoints/

# Copy trained models to current project
cp /Users/seanburdges/RECOVERY_OPS/ML_TRAINED_MODELS/.../checkpoints/*.pt models/checkpoints/

# Validate model loading
python scripts/validate_models.py --checkpoint models/checkpoints/harmonypredictor_best.pt
```

**If models are missing or corrupted, then:**

1. **Prepare training datasets**:
   ```bash
   # HarmonyPredictor (has data)
   python scripts/prepare_chord_dataset.py --input data/chord_progressions_db.json
   
   # EmotionRecognizer (download RAVDESS)
   python scripts/bulk_download_data.py --dataset ravdess --output data/emotion_audio
   
   # MelodyTransformer (download Lakh MIDI)
   python scripts/bulk_download_data.py --dataset lakh_midi --output data/midi_training
   ```

2. **Train first model (HarmonyPredictor)**:
   ```bash
   python scripts/train.py --config config/harmony_predictor.yaml
   # Expected: 2-3 hours on RTX 4060
   ```

3. **Validate and export**:
   ```bash
   python scripts/export_to_rtneural.py \
     --checkpoint checkpoints/harmony_predictor/best_model.pt \
     --output models/harmony_predictor.json
   ```

**Dependencies**: RTX 4060 training machine or AWS p3.2xlarge  
**Success Criteria**: Model achieves >80% accuracy on validation set, exports successfully

---

### 2. Test Coverage Gaps 🟡 **HIGH PRIORITY**

**Status**: Plugin test harness exists (✅ Complete), but ML/DSP tests needed  
**Impact**: Cannot guarantee ML inference quality or DSP accuracy

#### Coverage by Component

| Component | Current Coverage | Target | Gap |
|-----------|------------------|--------|-----|
| JUCE Plugins | ✅ **Complete** | N/A | ✅ Plugin test harness exists at `tests/penta_core/plugin_test_harness.cpp` |
| ML Module | **0%** | 75% | 🔴 CRITICAL |
| DSP Module | **0%** | 80% | 🔴 CRITICAL |
| Music Brain Core | 77% | 85% | 🟡 MEDIUM |
| Penta-Core C++ | 58% | 80% | 🟡 MEDIUM |
| Collaboration | 0% | 60% | 🟢 LOW |

#### ✅ JUCE Plugin Tests - ALREADY COMPLETE

**Status**: Plugin test harness already exists at `tests/penta_core/plugin_test_harness.cpp` (689 lines)

**Existing Features**:
- Mock Audio Device with RT callbacks
- RT-Safety Validator (tracks allocations, locks)
- Integration tests for all 15 components
- Performance benchmarks (<100μs targets)
- 100% component coverage across Penta-Core engines

**Note**: Individual plugin DSP tests (test_pencil.cpp, etc.) are NOT needed - testing is done at the engine level, which is more comprehensive.

**Test Types**:
1. **RT-Safety Verification**: Ensure no allocations in `processBlock()`
2. **DSP Accuracy**: Compression ratio, FFT accuracy, saturation curves
3. **Performance**: Latency measurements, CPU usage benchmarks
4. **Parameter Validation**: Range checks, automation smoothness

**Example Test Structure**:
```cpp
// tests/juce/test_press.cpp
#include "PluginTestHarness.h"

TEST(PressPlugin, RT_Safety) {
    auto plugin = std::make_unique<PressProcessor>();
    
    // Allocate buffers
    juce::AudioBuffer<float> buffer(2, 512);
    
    // Fill with test signal
    // ...
    
    // Verify no allocations in processBlock
    auto before = getAllocatedBytes();
    plugin->processBlock(buffer, midiBuffer);
    auto after = getAllocatedBytes();
    
    EXPECT_EQ(before, after) << "processBlock allocated memory!";
}
```

#### ML Module Tests (P0-2)

**Files to Create**:
- `tests/ml/test_inference.py` - Model inference tests
- `tests/ml/test_export.py` - ONNX/RTNeural export tests
- `tests/ml/test_style_transfer.py` - Style transfer tests

**Test Cases**:
- Model loading (PyTorch, ONNX, RTNeural)
- Inference latency (<5ms for EmotionRecognizer)
- Output shape validation
- Edge cases (empty input, out-of-range values)

#### DSP Module Tests (P0-3)

**Files to Create**:
- `tests/dsp/test_pitch_detection.py` - YIN algorithm tests
- `tests/dsp/test_phase_vocoder.py` - Phase vocoder tests
- `tests/dsp/test_fft_accuracy.py` - FFT accuracy tests

**Success Criteria**: All tests pass, coverage >75% for critical paths

---

### 3. CI/CD Pipeline Enhancement 🔴 **BLOCKER**

**Status**: Basic CI exists but missing critical validation  
**Impact**: Cannot ensure code quality or catch regressions automatically

#### Required CI Stages

1. **C++ Build & Test**
   ```yaml
   - name: Build C++ with CMake
     run: |
       mkdir build && cd build
       cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
       ninja penta_core
       ninja penta_tests
   
   - name: Run C++ Tests
     run: |
       cd build
       ctest --output-on-failure
   ```

2. **Valgrind Memory Testing**
   ```yaml
   - name: Valgrind Memory Check
     run: |
       valgrind --leak-check=full --error-exitcode=1 \
         ./build/penta_tests
   ```

3. **Performance Regression Tests**
   ```yaml
   - name: Performance Benchmarks
     run: |
       ./build/penta_tests --gtest_filter="*Performance*"
       # Fail if latency > 200μs
   ```

4. **Code Coverage Reporting**
   ```yaml
   - name: Generate Coverage Report
     run: |
       # C++: lcov
       lcov --capture --directory build --output-file coverage.info
       
       # Python: coverage.py
       pytest --cov=music_brain --cov-report=xml
   ```

5. **JUCE Plugin Validation** (macOS only)
   ```yaml
   - name: Validate JUCE Plugins
     if: runner.os == 'macOS'
     run: |
       auval -v aufx KmDi Pncl  # Pencil plugin
       auval -v aufx KmDi Ersr  # Eraser plugin
       # ... repeat for all 11 plugins
   ```

**Success Criteria**: All CI stages pass, coverage reports generated, no memory leaks

---

### 4. YAML Configuration Fixes 🟡 **HIGH**

**Status**: Partially fixed (see CHANGES_2025-12-30.md)  
**Impact**: Config files may fail to parse, causing runtime errors

#### Remaining Tasks

1. ✅ Fixed tabs→spaces in `config/build-dev-mac.yaml` (DONE)
2. ✅ Added missing fields to `config/train-mac-smoke.yaml` (DONE)
3. ⏳ **Verify external SSD path strategy**:
   - Document expected path structure
   - Add fallback paths for dev containers
   - Update `config/emotion_recognizer.yaml:38` and `config/harmony_predictor.yaml:33`

4. ⏳ **Test all config files**:
   ```bash
   python -c "import yaml; yaml.safe_load(open('config/build-dev-mac.yaml'))"
   # Repeat for all config files
   ```

**Success Criteria**: All configs parse without errors, external paths handled gracefully

---

## P1: High Priority - Core Functionality

### 5. FFT OnsetDetector Upgrade

**Current**: Using filterbank stub  
**Target**: Real FFT-based spectral flux detection  
**Effort**: 1-2 weeks

**Implementation**:
```cpp
// src_penta-core/groove/OnsetDetector.cpp
#include <juce_dsp/juce_dsp.h>

class OnsetDetector {
    juce::dsp::FFT fft;
    juce::AudioBuffer<float> fftBuffer;
    
    float calculateSpectralFlux(const float* audio, int numSamples) {
        // FFT → magnitude spectrum
        // Positive differences between frames
        // Sum = spectral flux
    }
};
```

**Success Criteria**: <200μs latency @ 48kHz/512 samples, >90% onset detection accuracy

---

### 6. Phase Vocoder Implementation

**Current**: Declared but not implemented  
**Target**: Full phase vocoder for Parrot plugin  
**Effort**: 2-3 weeks

**Files**:
- `python/penta_core/dsp/parrot_dsp.py` - Phase vocoder class
- `tests/dsp/test_phase_vocoder.py` - Unit tests

**Algorithm**: FFT → phase unwrapping → time-stretch/pitch-shift → IFFT

---

### 7. Training Data Pipeline

**Effort**: 2-3 weeks

**Tasks**:
1. Download datasets (RAVDESS, Lakh MIDI, FMA)
2. Create data loaders with preprocessing
3. Implement augmentation (pitch shift, time stretch, noise injection)
4. Create validation splits

**Scripts to Create**:
- `scripts/prepare_chord_dataset.py`
- `scripts/bulk_download_data.py`
- `scripts/augment_training_data.py`

---

### 8. Qt GUI Completion

**Current**: Design phase (see `docs/UI_REDESIGN_Qt.md`)  
**Target**: Functional desktop application  
**Effort**: 4-6 weeks

**Architecture** (3-layer separation):
```
GUI Layer (Qt) → Controller Layer → Core Logic (headless)
```

**Components**:
- Main window with emotion input
- Parameter controls (valence, arousal, intensity)
- Results display (chords, MIDI preview)
- AI assistant dock
- Logs panel

**Files to Create**:
- `kmidi_gui/core/engine.py` - Core business logic
- `kmidi_gui/controllers/actions.py` - Event routing
- `kmidi_gui/gui/main_window.py` - Qt UI

---

### 9. Production FastAPI Service

**Effort**: 2-3 weeks

**Features**:
- REST API endpoints (`/generate`, `/emotions`, `/interrogate`)
- Docker containerization
- Error handling and logging
- Rate limiting
- Health checks
- Monitoring (Prometheus metrics)

**Files to Create**:
- `api/main.py` - FastAPI application
- `api/Dockerfile` - Container definition
- `api/docker-compose.yml` - Development environment

---

### 10. Streamlit Demo

**Effort**: 1 week

**Features**:
- Simple form: emotion input → music output
- Audio playback in browser
- Download MIDI/audio files
- Share demo link

**Deploy**: Streamlit Cloud (free tier)

---

## P2: Medium Priority - Optimizations

### 11. SIMD Optimizations

**Target**: <100μs harmony, <200μs groove latency  
**Effort**: 3-4 weeks

**Kernels to Implement**:
- Chord pattern matching (AVX2)
- RMS calculation (AVX2)
- FFT preprocessing (AVX2)
- Autocorrelation (AVX2)

**Files**:
- `include/penta/common/SIMDKernels.h`
- `src_penta-core/common/SIMDKernels.cpp`

---

### 12. Production Guides Integration

**Effort**: 2-3 weeks

**Modules to Create**:
- `music_brain/emotion/emotion_production.py` - Maps emotions → production techniques
- `music_brain/groove/drum_humanizer.py` - Applies drum programming guide rules
- `music_brain/production/dynamics_engine.py` - Applies dynamics guide

**Benefit**: Guides become executable code, not just documentation

---

### 13. Cross-Cultural Music Support

**Effort**: 4-6 weeks

**Cultures**:
- Indian (Raga-based)
- Arabic (Maqam-based)
- East Asian (pentatonic)

**Tasks**:
- Expand emotion mappings
- Create validation datasets
- Recruit native therapists for validation
- Generate 20-50 samples per culture

---

## P3: Low Priority - Polish & Future

### 14-17. New ML Models

**Intent Classifier**, **Audio Feel Analyzer**, **Structure Predictor**, **Micro-Timing Model**

**Effort**: 6-8 weeks each  
**Benefit**: Enhanced capabilities, but not blockers

---

### 18. Documentation Completion

**Effort**: 2-3 weeks

**Tasks**:
- Generate C++ API docs (Doxygen)
- Create video tutorials
- Write migration guides
- Add usage examples

---

### 19. Desktop App Polish

**Effort**: 3-4 weeks

**Tasks**:
- Finish Tauri wrapper
- Add native menu integration
- Implement auto-updater
- Create installers (macOS/Windows/Linux)

---

## Implementation Timeline

### Phase 1: Critical Fixes (Weeks 1-4)
- ✅ Fix YAML configs
- Train HarmonyPredictor
- Add JUCE plugin tests
- Enhance CI/CD pipeline

### Phase 2: Core Features (Weeks 5-8)
- Train EmotionRecognizer
- Complete Qt GUI
- Create FastAPI service
- Deploy Streamlit demo

### Phase 3: Optimizations (Weeks 9-12)
- Train MelodyTransformer
- Implement SIMD optimizations
- Integrate production guides
- Add ML/DSP tests

### Phase 4: Polish (Weeks 13-16)
- Train remaining models
- Complete documentation
- Cross-cultural validation
- Desktop app polish

---

## Success Metrics

### Technical
- ✅ All models trained and validated (>80% accuracy)
- ✅ Test coverage >80% for critical components
- ✅ CI/CD pipeline passing all stages
- ✅ Latency <200ms for real-time processing

### Product
- ✅ Functional desktop application
- ✅ Production API deployed
- ✅ Beta testing with 10+ therapists
- ✅ 30+ therapy sessions collected

### Business
- ✅ Clinical RCT protocol approved
- ✅ Cross-cultural validation complete
- ✅ Published paper submitted
- ✅ 100+ customers acquired

---

## Risk Mitigation

### Risk 1: Model Training Takes Longer Than Expected
**Mitigation**: Start with easiest model (HarmonyPredictor), use synthetic data where possible

### Risk 2: Test Coverage Slows Development
**Mitigation**: Write tests alongside code, not after. Use TDD for critical paths.

### Risk 3: External Dependencies Delay CI/CD
**Mitigation**: Use containerized builds, cache dependencies, have fallback options

---

## Next Steps

1. **Today**: Review this plan, prioritize based on business needs
2. **This Week**: Start HarmonyPredictor training, begin JUCE test harness
3. **This Month**: Complete P0 items, begin P1 items
4. **This Quarter**: Complete P1 items, begin P2 items

---

**Questions or Concerns?** Review with team, adjust priorities as needed.

**Last Updated**: 2025-01-07

