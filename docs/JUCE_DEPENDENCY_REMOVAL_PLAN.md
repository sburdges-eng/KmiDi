# JUCE Dependency Removal Plan

**Date:** January 18, 2026  
**Status:** Planning  
**Priority:** High (Week 2)

## Overview

This document outlines the plan to remove JUCE dependencies from audio processing code (`src/audio/`, `src/engine/`, `src/ml/`) and replace them with pure DSP implementations to maintain real-time safety and architectural boundaries.

## Current State

### Files with JUCE Dependencies

#### src/audio/
- `AudioAnalyzer.h/cpp` - Uses `juce_audio_basics`
- `SpectralAnalyzer.h/cpp` - Uses `juce_audio_basics` and `juce_dsp`
- `F0Extractor.h/cpp` - Uses `juce_audio_basics`

#### src/engine/
- Multiple files may have JUCE dependencies (needs audit)

#### src/ml/
- Multiple files may have JUCE dependencies (needs audit)

### Existing Pure DSP Code

- `src/dsp/audio_buffer.cpp` - Pure DSP audio buffer (no JUCE)
- `src/dsp/filters.cpp` - Pure DSP filters
- `src/dsp/simd_ops.cpp` - SIMD operations

## Migration Strategy

### Phase 1: Audit and Document (Day 1)

1. **Complete Dependency Audit**
   - List all files with JUCE includes
   - Document which JUCE classes/functions are used
   - Identify dependencies by category:
     - Audio buffers/format conversion
     - DSP operations (FFT, filters)
     - Audio file I/O
     - Threading/utilities

2. **Create Replacement Mapping**
   - Map JUCE classes to pure DSP alternatives
   - Identify gaps requiring new implementations
   - Document API compatibility requirements

### Phase 2: Build Pure DSP Foundation (Day 2)

1. **Extend src/dsp/ with Missing Functionality**
   - Audio buffer operations (already exists)
   - FFT implementation (replace JUCE FFT)
   - Window functions (for spectral analysis)
   - Audio format conversion utilities
   - Thread-safe utilities (if needed)

2. **Reference KmiDi_FINAL Pure DSP**
   - Review `KmiDi-1/KmiDi_FINAL/engine/src/dsp/` for implementations
   - Identify reusable pure DSP code
   - Plan integration approach

### Phase 3: Migrate src/audio/ (Days 3-4)

1. **AudioAnalyzer Migration**
   - Replace `juce::AudioBuffer` with `daiw::AudioBuffer`
   - Replace JUCE audio format with pure format handling
   - Test audio analysis functionality

2. **SpectralAnalyzer Migration**
   - Replace `juce::dsp::FFT` with pure FFT implementation
   - Replace `juce::dsp::WindowingFunction` with pure windowing
   - Test spectral analysis functionality

3. **F0Extractor Migration**
   - Replace JUCE audio buffer with pure DSP buffer
   - Test pitch extraction functionality

### Phase 4: Migrate src/engine/ and src/ml/ (Days 5-6)

1. **Engine Code Migration**
   - Audit engine files for JUCE dependencies
   - Replace with pure DSP alternatives
   - Test engine functionality

2. **ML Code Migration**
   - Audit ML files for JUCE dependencies
   - Replace with pure DSP alternatives
   - Test ML inference functionality

### Phase 5: Build System Updates (Day 7)

1. **Update CMakeLists.txt**
   - Remove JUCE from audio/engine/ml target dependencies
   - Ensure pure DSP code is linked
   - Update include paths

2. **Verify Build**
   - Clean build from scratch
   - Verify no JUCE dependencies in audio paths
   - Test compilation on macOS and Linux

## Replacement Mappings

### JUCE Audio Basics → Pure DSP

| JUCE Class | Pure DSP Replacement | Status |
|------------|---------------------|--------|
| `juce::AudioBuffer<float>` | `daiw::AudioBuffer` | ✅ Exists |
| `juce::AudioFormat` | Custom format handlers | ⚠️ Needs implementation |
| `juce::AudioFormatReader` | Custom file readers | ⚠️ Needs implementation |
| `juce::AudioFormatWriter` | Custom file writers | ⚠️ Needs implementation |

### JUCE DSP → Pure DSP

| JUCE Class | Pure DSP Replacement | Status |
|------------|---------------------|--------|
| `juce::dsp::FFT` | Custom FFT (FFTW or kiss_fft) | ⚠️ Needs implementation |
| `juce::dsp::WindowingFunction` | Custom windowing | ⚠️ Needs implementation |
| `juce::dsp::ProcessContext` | Custom process context | ⚠️ Needs implementation |

## Implementation Notes

### FFT Implementation Options

1. **FFTW** (Fastest Fourier Transform in the West)
   - High performance
   - Requires separate library
   - License: GPL (or commercial)

2. **kiss_fft**
   - Lightweight, BSD-licensed
   - Good performance
   - Pure C implementation

3. **Custom FFT**
   - Full control
   - More implementation work
   - Can optimize for specific use cases

**Recommendation:** Use kiss_fft for pure DSP, or reference KmiDi_FINAL implementation.

### Audio File I/O

For audio file reading/writing without JUCE:
- Use libsndfile (WAV, AIFF, FLAC support)
- Use dr_wav for WAV files (header-only)
- Use stb_vorbis for OGG Vorbis (if needed)

**Recommendation:** Use libsndfile for comprehensive format support.

## Testing Requirements

### Unit Tests

- Test each migrated component independently
- Verify output matches JUCE version (within tolerance)
- Test edge cases (empty buffers, invalid inputs)

### Integration Tests

- Test audio analysis pipeline
- Test spectral analysis pipeline
- Test ML inference with pure DSP

### Performance Tests

- Benchmark pure DSP vs JUCE performance
- Verify real-time safety (no allocations in audio thread)
- Measure latency impact

## Risk Mitigation

1. **Incremental Migration**
   - Migrate one file at a time
   - Test after each migration
   - Keep JUCE code as fallback initially

2. **Feature Parity**
   - Ensure all functionality is preserved
   - Document any behavioral differences
   - Test with real audio files

3. **Build System Safety**
   - Keep JUCE in plugin code (allowed)
   - Only remove from audio/engine/ml paths
   - Verify plugin builds still work

## Success Criteria

- [ ] No JUCE includes in `src/audio/`
- [ ] No JUCE includes in `src/engine/` (except where allowed)
- [ ] No JUCE includes in `src/ml/` (except where allowed)
- [ ] All tests pass
- [ ] Build system compiles without JUCE in audio paths
- [ ] Performance is maintained or improved
- [ ] Real-time safety verified

## References

- `docs/AI_CONTROL_LAYER.md` - AI architecture boundaries
- `docs/STRUCTURE_CROSS_EXAMINATION/03_CODE_ARCHITECTURE_REPORT.md` - Architecture analysis
- `KmiDi-1/KmiDi_FINAL/engine/src/dsp/` - Pure DSP reference implementation
