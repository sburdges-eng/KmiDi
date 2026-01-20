# Vocal Generation Robustness Improvement Plan

**Date:** 2026-01-22
**Status:** Planning phase
**Priority:** High

## Current Issues Identified

### 1. Pitch Tracking (Critical)
**Problem:** Hardcoded pitch targets (MIDI 60, confidence 0.7f)
```cpp
// Line 62-71 in PRROTEngine.cpp
for (const auto& phoneme : phoneme_sequence) {
    PitchTarget target;
    target.time_ms = phoneme.start_time_ms;
    target.midi_note = 60; // Default middle C - PLACEHOLDER
    target.cents_offset = 0.0f;
    target.confidence = 0.7f; // PLACEHOLDER
    control_data.pitch_targets.push_back(target);
}
```

**Impact:** No actual pitch tracking, all vocals will be at middle C

### 2. Phoneme Segmentation (Critical)
**Problem:** Basic energy-based segmentation with placeholder confidence
```cpp
// Line 44 in PhonemeSegmenter.cpp
// This is a placeholder - actual implementation would use more sophisticated methods
result.confidence = result.valid ? 0.7f : 0.0f; // Placeholder confidence
```

**Impact:** Low-quality segmentation, may miss phonemes or create false boundaries

### 3. Error Handling (High)
**Problem:** No recovery mechanisms, silent failures
- If segmentation fails → returns empty sequence
- No validation of results
- No fallback strategies

### 4. Input Validation (High)
**Problem:** Basic checks only
- No silence detection
- No noise floor estimation
- No clipping detection
- No sample rate validation
- No minimum duration checks

### 5. Confidence Propagation (Medium)
**Problem:** Confidence values not used for decision-making
- Hardcoded confidence values
- No confidence-based filtering
- No quality thresholds

### 6. Edge Case Handling (Medium)
**Problem:** No handling of:
- Very short audio segments (<100ms)
- Silence-only segments
- High noise segments
- Clipped audio
- Different sample rates

### 7. Adaptive Thresholds (Medium)
**Problem:** Hardcoded thresholds
- Energy threshold: 0.01f (line 45)
- Breath confidence: 0.5f (line 198)
- No adaptation to audio characteristics

---

## Robustness Improvements

### Phase 1: Critical Fixes (Immediate)

#### 1.1 Implement Real Pitch Tracking

**Current:**
```cpp
// Placeholder - hardcoded MIDI 60
target.midi_note = 60;
```

**Improved:**
```cpp
// Real pitch tracking using autocorrelation or FFT-based methods
PitchTarget trackPitch(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) noexcept;
```

**Implementation:**
- Autocorrelation for fundamental frequency (F0)
- FFT-based pitch detection
- MIDI note conversion with cents offset
- Confidence based on signal quality

#### 1.2 Improve Phoneme Segmentation

**Current:**
```cpp
// Basic energy-based segmentation
float energy_threshold = 0.01f; // Hardcoded
```

**Improved:**
```cpp
// Multi-feature segmentation
- Spectral features (formants)
- Energy analysis
- Zero-crossing rate
- Adaptive thresholds
- Confidence scoring
```

**Implementation:**
- Use SpectralAnalyzer for formant detection
- Adaptive energy threshold based on noise floor
- Multi-frame analysis for better boundaries
- Confidence based on feature agreement

#### 1.3 Add Input Validation

**Implementation:**
```cpp
struct AudioQuality {
    bool has_silence;
    bool is_clipped;
    float noise_floor;
    float signal_level;
    float snr;  // Signal-to-noise ratio
    bool is_valid;
};

AudioQuality validateAudio(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) noexcept;
```

**Checks:**
- Minimum duration (e.g., 100ms)
- Silence detection
- Clipping detection
- Noise floor estimation
- Signal level validation
- Sample rate validation

### Phase 2: Error Handling & Recovery

#### 2.1 Graceful Degradation

**Implementation:**
```cpp
PhonemeControlData processAudioSegment(...) noexcept {
    // Validate input
    auto quality = validateAudio(audio_samples, num_samples, sample_rate_hz);
    if (!quality.is_valid) {
        return createFallbackControlData(quality);
    }

    // Try segmentation with fallback
    auto phonemes = analyzePhonemes(...);
    if (phonemes.empty() || lowConfidence(phonemes)) {
        // Fallback: Use simpler segmentation
        phonemes = fallbackSegmentation(...);
    }

    // Try pitch tracking with fallback
    auto pitch = trackPitch(...);
    if (pitch.confidence < 0.3f) {
        // Fallback: Use phoneme-based pitch estimation
        pitch = estimatePitchFromPhonemes(phonemes);
    }

    // Continue with best available data
}
```

#### 2.2 Confidence-Based Filtering

**Implementation:**
```cpp
// Filter low-confidence results
void filterLowConfidence(
    std::vector<PhonemeTiming>& phonemes,
    float min_confidence = 0.5f
) noexcept;

// Use confidence for weighting
float weightedConfidence(
    const std::vector<PhonemeTiming>& phonemes
) noexcept;
```

### Phase 3: Advanced Features

#### 3.1 Adaptive Thresholds

**Implementation:**
```cpp
class AdaptiveThresholds {
    float computeEnergyThreshold(
        const float* audio_samples,
        size_t num_samples
    ) noexcept;

    float computeNoiseFloor(
        const float* audio_samples,
        size_t num_samples
    ) noexcept;

    float computeBreathThreshold(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) noexcept;
};
```

#### 3.2 Quality Metrics

**Implementation:**
```cpp
struct ProcessingQuality {
    float segmentation_confidence;
    float pitch_tracking_confidence;
    float breath_detection_confidence;
    float overall_quality;
    bool is_usable;
};

ProcessingQuality assessQuality(
    const PhonemeControlData& control_data
) noexcept;
```

#### 3.3 Multi-Pass Processing

**Implementation:**
```cpp
// First pass: Basic analysis
auto basic_result = processAudioSegmentBasic(...);

// Second pass: Refinement if quality is low
if (basic_result.quality.overall_quality < 0.7f) {
    basic_result = refineProcessing(basic_result, ...);
}

// Third pass: ML enhancement (standalone only)
if (isStandaloneMode() && basic_result.quality.overall_quality < 0.8f) {
    basic_result = enhanceWithML(basic_result, ...);
}
```

---

## Implementation Plan

### Step 1: Pitch Tracking Implementation

**File:** `engine/src/prrot/PitchTracker.h` (new)

```cpp
class PitchTracker {
public:
    struct PitchResult {
        float frequency_hz;
        int midi_note;
        float cents_offset;
        float confidence;
        bool is_valid;
    };

    PitchResult trackPitch(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

private:
    // Autocorrelation-based pitch detection
    float autocorrelationPitch(
        const float* samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    // FFT-based pitch detection
    float fftPitch(
        const float* samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    // Convert frequency to MIDI note
    int frequencyToMidi(float frequency_hz) const noexcept;
    float midiToFrequency(int midi_note) const noexcept;
};
```

### Step 2: Enhanced Phoneme Segmentation

**File:** `engine/src/prrot/PhonemeSegmenter.cpp` (enhance)

**Improvements:**
1. Use SpectralAnalyzer for formant-based segmentation
2. Adaptive energy threshold
3. Multi-frame analysis
4. Confidence scoring based on feature agreement
5. Boundary refinement

### Step 3: Input Validation

**File:** `engine/src/prrot/AudioValidator.h` (new)

```cpp
class AudioValidator {
public:
    struct ValidationResult {
        bool is_valid;
        bool has_silence;
        bool is_clipped;
        float noise_floor_db;
        float signal_level_db;
        float snr_db;
        float duration_ms;
        std::string error_message;
    };

    ValidationResult validate(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

private:
    bool detectSilence(
        const float* samples,
        size_t num_samples,
        float threshold_db = -60.0f
    ) const noexcept;

    bool detectClipping(
        const float* samples,
        size_t num_samples,
        float threshold = 0.99f
    ) const noexcept;

    float estimateNoiseFloor(
        const float* samples,
        size_t num_samples
    ) const noexcept;

    float computeSNR(
        const float* samples,
        size_t num_samples,
        float noise_floor
    ) const noexcept;
};
```

### Step 4: Error Recovery

**File:** `engine/src/prrot/PRROTEngine.cpp` (enhance)

**Add:**
- Input validation before processing
- Fallback strategies for each component
- Confidence-based filtering
- Quality assessment
- Graceful degradation

### Step 5: Confidence System

**Implementation:**
- Real confidence calculation for all components
- Confidence propagation through pipeline
- Confidence-based filtering and weighting
- Quality metrics based on confidence

---

## Code Changes Required

### Files to Create
1. `engine/src/prrot/PitchTracker.h`
2. `engine/src/prrot/PitchTracker.cpp`
3. `engine/src/prrot/AudioValidator.h`
4. `engine/src/prrot/AudioValidator.cpp`
5. `engine/src/prrot/QualityAssessor.h`
6. `engine/src/prrot/QualityAssessor.cpp`

### Files to Modify
1. `engine/src/prrot/PRROTEngine.h` - Add new components
2. `engine/src/prrot/PRROTEngine.cpp` - Integrate improvements
3. `engine/src/prrot/PhonemeSegmenter.cpp` - Enhance segmentation
4. `engine/src/prrot/PhonemeSegmenter.h` - Add confidence methods

---

## Testing Requirements

### Unit Tests
- Pitch tracking accuracy
- Phoneme segmentation accuracy
- Input validation edge cases
- Error recovery scenarios
- Confidence calculation

### Integration Tests
- Full pipeline with various audio qualities
- Silence handling
- Noise handling
- Clipping handling
- Very short segments
- Different sample rates

### Performance Tests
- Processing time for various segment sizes
- Memory usage
- RT-safety verification

---

## Success Criteria

### Phase 1 (Critical)
- ✅ Real pitch tracking (not hardcoded)
- ✅ Improved phoneme segmentation
- ✅ Input validation
- ✅ Basic error handling

### Phase 2 (Error Handling)
- ✅ Graceful degradation
- ✅ Confidence-based filtering
- ✅ Fallback strategies
- ✅ Quality assessment

### Phase 3 (Advanced)
- ✅ Adaptive thresholds
- ✅ Multi-pass processing
- ✅ ML enhancement integration
- ✅ Comprehensive quality metrics

---

## Priority Order

1. **Pitch Tracking** - Critical, currently broken
2. **Input Validation** - Prevents garbage in/garbage out
3. **Phoneme Segmentation** - Core functionality
4. **Error Handling** - Robustness
5. **Confidence System** - Quality assurance
6. **Adaptive Thresholds** - Better accuracy
7. **Multi-Pass Processing** - Quality improvement

---

**See Also:**
- `docs/FULL_PIPELINE_DOCUMENTATION.md` - Current pipeline
- `docs/STANDALONE_GENERATION_ARCHITECTURE.md` - Architecture
