# Detailed Robustness Improvements - 3 Per Component

**Date:** 2026-01-22
**Status:** Implementation Guide
**Priority:** CRITICAL

This document provides **exactly 3 specific, actionable improvements** for each major component, file, and documentation piece in the codebase.

---

## Core Engine Components

### PRROTEngine.cpp/h

**Current Issues:**
- Silent failures (returns empty results)
- No error recovery
- Basic validation only
- No quality metrics

**Improvement 1: Result-Based Error Handling**
```cpp
// BEFORE:
PhonemeControlData processAudioSegment(...) noexcept {
    if (!audio_samples || num_samples == 0) {
        return control_data; // Silent failure
    }
    // ...
}

// AFTER:
Result<PhonemeControlData, ProcessingError> processAudioSegment(...) noexcept {
    // Validate inputs
    auto validation = validateInputs(audio_samples, num_samples, sample_rate_hz);
    if (!validation.isValid()) {
        return Result<PhonemeControlData, ProcessingError>::error(
            ProcessingError::InvalidInput, validation.errorMessage()
        );
    }

    // Process with error recovery
    auto phonemes = analyzePhonemes(...);
    if (!phonemes.hasValue()) {
        // Try fallback method
        phonemes = fallbackPhonemeAnalysis(...);
        if (!phonemes.hasValue()) {
            return Result<PhonemeControlData, ProcessingError>::error(
                ProcessingError::PhonemeAnalysisFailed, "All methods failed"
            );
        }
    }
    // ...
    return Result<PhonemeControlData, ProcessingError>::success(control_data);
}
```

**Improvement 2: Comprehensive Input Validation**
```cpp
// BEFORE:
if (!audio_samples || num_samples == 0 || sample_rate_hz <= 0.0f) {
    return control_data;
}

// AFTER:
struct InputValidation {
    bool isValid() const { return errors.empty(); }
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
};

InputValidation validateInputs(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) noexcept {
    InputValidation result;

    // Null check
    if (!audio_samples) {
        result.errors.push_back("audio_samples is null");
        return result;
    }

    // Sample count validation
    if (num_samples == 0) {
        result.errors.push_back("num_samples is zero");
        return result;
    }
    if (num_samples > kMaxSamplesPerSegment) {
        result.errors.push_back("num_samples exceeds maximum");
        return result;
    }
    if (num_samples < kMinSamplesPerSegment) {
        result.warnings.push_back("num_samples below recommended minimum");
    }

    // Sample rate validation
    if (sample_rate_hz <= 0.0f) {
        result.errors.push_back("sample_rate_hz is invalid");
        return result;
    }
    if (sample_rate_hz < 8000.0f || sample_rate_hz > 192000.0f) {
        result.warnings.push_back("sample_rate_hz outside typical range");
    }

    // Audio quality checks
    auto quality = audio_validator_->validate(audio_samples, num_samples, sample_rate_hz);
    if (!quality.is_valid) {
        result.errors.push_back("Audio quality validation failed: " + quality.error_message);
    } else if (quality.quality_score() < 0.5f) {
        result.warnings.push_back("Audio quality is low (score: " +
                                 std::to_string(quality.quality_score()) + ")");
    }

    return result;
}
```

**Improvement 3: Quality Metrics & Confidence Propagation**
```cpp
// BEFORE:
// No quality metrics, no confidence tracking

// AFTER:
struct ProcessingQuality {
    float overall_quality = 0.0f;
    float phoneme_confidence = 0.0f;
    float pitch_confidence = 0.0f;
    float breath_confidence = 0.0f;
    bool is_usable = false;

    std::string qualityReport() const {
        std::string report;
        report += "Overall Quality: " + std::to_string(overall_quality) + "\n";
        report += "Phoneme Confidence: " + std::to_string(phoneme_confidence) + "\n";
        report += "Pitch Confidence: " + std::to_string(pitch_confidence) + "\n";
        report += "Breath Confidence: " + std::to_string(breath_confidence) + "\n";
        report += "Usable: " + std::string(is_usable ? "Yes" : "No") + "\n";
        return report;
    }
};

PhonemeControlData processAudioSegment(...) noexcept {
    ProcessingQuality quality;

    // Process with quality tracking
    auto phonemes = analyzePhonemes(...);
    quality.phoneme_confidence = calculatePhonemeConfidence(phonemes);

    auto pitch_targets = trackPitch(...);
    quality.pitch_confidence = calculatePitchConfidence(pitch_targets);

    auto breath_markers = detectBreathMarkers(...);
    quality.breath_confidence = calculateBreathConfidence(breath_markers);

    // Calculate overall quality
    quality.overall_quality = (
        quality.phoneme_confidence * 0.4f +
        quality.pitch_confidence * 0.4f +
        quality.breath_confidence * 0.2f
    );

    quality.is_usable = quality.overall_quality >= 0.5f;

    // Store quality in result
    control_data.quality_metrics = quality;

    // Filter low-confidence results if quality is low
    if (!quality.is_usable) {
        filterLowConfidenceResults(control_data, 0.6f);
    }

    return control_data;
}
```

---

### PhonemeSegmenter.cpp/h

**Current Issues:**
- Placeholder confidence (0.7f)
- Basic energy-based segmentation
- Memory pool can be nullptr
- No buffer overflow protection

**Improvement 1: Multi-Feature Segmentation Algorithm**
```cpp
// BEFORE:
// Simple energy-based segmentation with placeholder confidence
float energy_threshold = 0.01f; // Hardcoded
result.confidence = result.valid ? 0.7f : 0.0f; // Placeholder

// AFTER:
struct SegmentationFeatures {
    float energy = 0.0f;
    float spectral_centroid = 0.0f;
    float zero_crossing_rate = 0.0f;
    float formant_f1 = 0.0f;
    float formant_f2 = 0.0f;
    float spectral_rolloff = 0.0f;
};

SegmentResult segment(...) noexcept {
    SegmentResult result;

    // Extract multiple features
    std::vector<SegmentationFeatures> features;
    for (size_t i = 0; i < samples_to_process; i += frame_size) {
        SegmentationFeatures frame_features;
        frame_features.energy = computeEnergy(...);
        frame_features.spectral_centroid = computeSpectralCentroid(...);
        frame_features.zero_crossing_rate = computeZCR(...);
        frame_features.formant_f1 = extractFormant(...);
        frame_features.formant_f2 = extractFormant(...);
        frame_features.spectral_rolloff = computeSpectralRolloff(...);
        features.push_back(frame_features);
    }

    // Adaptive threshold based on audio characteristics
    float adaptive_threshold = computeAdaptiveThreshold(features);

    // Multi-feature boundary detection
    std::vector<size_t> boundaries = detectBoundariesMultiFeature(
        features, adaptive_threshold
    );

    // Refine boundaries using multiple methods
    boundaries = refineBoundaries(boundaries, features);

    // Calculate confidence based on feature agreement
    result.confidence = calculateConfidence(features, boundaries);

    // Validate results
    if (!validateSegmentation(boundaries, features)) {
        result.valid = false;
        result.confidence = 0.0f;
        return result;
    }

    result.valid = true;
    return result;
}

float calculateConfidence(
    const std::vector<SegmentationFeatures>& features,
    const std::vector<size_t>& boundaries
) const noexcept {
    // Confidence based on:
    // 1. Feature consistency (similar features in same segment)
    // 2. Boundary clarity (clear transitions)
    // 3. Method agreement (multiple methods agree)

    float consistency_score = calculateFeatureConsistency(features, boundaries);
    float clarity_score = calculateBoundaryClarity(features, boundaries);
    float agreement_score = calculateMethodAgreement(features, boundaries);

    return (consistency_score * 0.4f + clarity_score * 0.4f + agreement_score * 0.2f);
}
```

**Improvement 2: Memory Safety & Buffer Management**
```cpp
// BEFORE:
void initialize(penta::RTMemoryPool* memory_pool = nullptr) {
    memory_pool_ = memory_pool; // Can be nullptr
}

// AFTER:
enum class InitializationError {
    Success,
    MemoryPoolRequired,
    BufferAllocationFailed,
    InvalidConfiguration
};

Result<void, InitializationError> initialize(
    penta::RTMemoryPool* memory_pool,
    const InitializationConfig& config = {}
) noexcept {
    // Memory pool is required for RT-safety
    if (!memory_pool) {
        return Result<void, InitializationError>::error(
            InitializationError::MemoryPoolRequired,
            "Memory pool is required for RT-safe operation"
        );
    }

    memory_pool_ = memory_pool;

    // Allocate all buffers from memory pool
    auto fft_buffer_result = memory_pool_->allocate<float>(kPhonemeFFTSize * 2);
    if (!fft_buffer_result.hasValue()) {
        return Result<void, InitializationError>::error(
            InitializationError::BufferAllocationFailed,
            "Failed to allocate FFT buffers"
        );
    }

    // Validate buffer sizes
    if (config.max_segment_size > kMaxSegmentBufferSize) {
        return Result<void, InitializationError>::error(
            InitializationError::InvalidConfiguration,
            "max_segment_size exceeds maximum"
        );
    }

    // Initialize all buffers
    initializeBuffers(config);

    return Result<void, InitializationError>::success();
}

SegmentResult segment(...) noexcept {
    // Validate buffer sizes before use
    if (num_samples > kMaxSegmentBufferSize) {
        SegmentResult result;
        result.valid = false;
        result.confidence = 0.0f;
        RTLogger::error("Segment size exceeds maximum: {} > {}",
                       num_samples, kMaxSegmentBufferSize);
        return result;
    }

    // Bounds checking for all array accesses
    size_t safe_samples = std::min(num_samples, kMaxSegmentBufferSize);
    size_t safe_frame_size = std::min(frame_size, safe_samples);

    // Use safe buffer access
    for (size_t i = 0; i < safe_samples; i += safe_frame_size) {
        size_t end = std::min(i + safe_frame_size, safe_samples);
        // Process with bounds checking
        // ...
    }
}
```

**Improvement 3: Confidence Calculation & Validation**
```cpp
// BEFORE:
result.confidence = result.valid ? 0.7f : 0.0f; // Placeholder

// AFTER:
struct ConfidenceMetrics {
    float feature_agreement = 0.0f;      // Multiple features agree
    float boundary_clarity = 0.0f;        // Clear transitions
    float temporal_consistency = 0.0f;    // Consistent over time
    float spectral_quality = 0.0f;        // Good spectral features
    float overall = 0.0f;
};

ConfidenceMetrics calculateConfidence(
    const SegmentResult& result,
    const std::vector<SegmentationFeatures>& features
) const noexcept {
    ConfidenceMetrics metrics;

    // Feature agreement: multiple methods agree on boundaries
    metrics.feature_agreement = calculateFeatureAgreement(features, result.boundaries_ms);

    // Boundary clarity: clear transitions at boundaries
    metrics.boundary_clarity = calculateBoundaryClarity(features, result.boundaries_ms);

    // Temporal consistency: boundaries are consistent over time
    metrics.temporal_consistency = calculateTemporalConsistency(result);

    // Spectral quality: good spectral features (formants, etc.)
    metrics.spectral_quality = calculateSpectralQuality(features);

    // Weighted overall confidence
    metrics.overall = (
        metrics.feature_agreement * 0.3f +
        metrics.boundary_clarity * 0.3f +
        metrics.temporal_consistency * 0.2f +
        metrics.spectral_quality * 0.2f
    );

    return metrics;
}

bool validateSegmentation(const SegmentResult& result) const noexcept {
    // Check minimum segment duration
    for (size_t i = 0; i < result.boundaries_ms.size() - 1; ++i) {
        float duration = result.boundaries_ms[i + 1] - result.boundaries_ms[i];
        if (duration < kMinPhonemeDurationMs) {
            RTLogger::warning("Segment {} too short: {}ms < {}ms",
                            i, duration, kMinPhonemeDurationMs);
            return false;
        }
    }

    // Check maximum number of segments
    if (result.phonemes.size() > kMaxPhonemesPerSegment) {
        RTLogger::warning("Too many segments: {} > {}",
                        result.phonemes.size(), kMaxPhonemesPerSegment);
        return false;
    }

    // Check boundaries are in order
    for (size_t i = 0; i < result.boundaries_ms.size() - 1; ++i) {
        if (result.boundaries_ms[i] >= result.boundaries_ms[i + 1]) {
            RTLogger::error("Boundaries out of order: {} >= {}",
                          result.boundaries_ms[i], result.boundaries_ms[i + 1]);
            return false;
        }
    }

    return true;
}
```

---

### PitchTracker.cpp/h

**Current Issues:**
- Autocorrelation implementation may be incomplete
- Confidence calculation placeholder
- No edge case handling
- No performance optimization

**Improvement 1: Robust Pitch Detection with Multiple Methods**
```cpp
// BEFORE:
// Basic autocorrelation, no fallback, no edge case handling

// AFTER:
PitchResult trackPitch(...) const noexcept {
    PitchResult result;

    // Validate inputs
    if (!audio_samples || num_samples < kMinSamplesForPitch || sample_rate_hz <= 0.0f) {
        result.is_valid = false;
        return result;
    }

    // Try multiple methods and combine results
    std::vector<MethodResult> method_results;

    // Method 1: Autocorrelation (best for clean signals)
    auto autocorr_result = autocorrelationPitch(audio_samples, num_samples, sample_rate_hz);
    if (autocorr_result.is_valid && autocorr_result.confidence > 0.5f) {
        method_results.push_back(autocorr_result);
    }

    // Method 2: FFT-based (better for noisy signals)
    auto fft_result = fftPitch(audio_samples, num_samples, sample_rate_hz);
    if (fft_result.is_valid && fft_result.confidence > 0.5f) {
        method_results.push_back(fft_result);
    }

    // Method 3: Cepstrum (good for harmonic signals)
    auto cepstrum_result = cepstrumPitch(audio_samples, num_samples, sample_rate_hz);
    if (cepstrum_result.is_valid && cepstrum_result.confidence > 0.5f) {
        method_results.push_back(cepstrum_result);
    }

    // Combine results
    if (method_results.empty()) {
        result.is_valid = false;
        result.confidence = 0.0f;
        return result;
    }

    // Weighted average of methods that agree
    result = combineMethodResults(method_results);

    // Validate result
    if (!validatePitchResult(result, sample_rate_hz)) {
        result.is_valid = false;
        return result;
    }

    return result;
}

PitchResult combineMethodResults(
    const std::vector<MethodResult>& results
) const noexcept {
    // Find methods that agree (within 5 cents)
    std::vector<MethodResult> agreeing;
    for (const auto& r1 : results) {
        int count = 0;
        for (const auto& r2 : results) {
            float cents_diff = std::abs(frequencyToCents(r1.frequency_hz, r2.frequency_hz));
            if (cents_diff < 5.0f) {
                count++;
            }
        }
        if (count >= 2) { // At least 2 methods agree
            agreeing.push_back(r1);
        }
    }

    if (agreeing.empty()) {
        // Use highest confidence method
        auto best = std::max_element(results.begin(), results.end(),
            [](const auto& a, const auto& b) { return a.confidence < b.confidence; });
        return *best;
    }

    // Weighted average of agreeing methods
    float total_weight = 0.0f;
    float weighted_freq = 0.0f;
    float max_confidence = 0.0f;

    for (const auto& r : agreeing) {
        float weight = r.confidence;
        total_weight += weight;
        weighted_freq += r.frequency_hz * weight;
        max_confidence = std::max(max_confidence, r.confidence);
    }

    PitchResult result;
    result.frequency_hz = weighted_freq / total_weight;
    result.midi_note = frequencyToMidi(result.frequency_hz);
    result.cents_offset = frequencyToCents(result.frequency_hz, result.midi_note);
    result.confidence = max_confidence * 1.1f; // Boost confidence when methods agree
    result.is_valid = true;

    return result;
}
```

**Improvement 2: Confidence Calculation Based on Signal Quality**
```cpp
// BEFORE:
// Placeholder confidence calculation

// AFTER:
float calculateConfidence(
    const float* samples,
    size_t num_samples,
    float detected_frequency,
    float sample_rate_hz
) const noexcept {
    ConfidenceFactors factors;

    // Factor 1: Signal-to-noise ratio
    factors.snr = calculateSNR(samples, num_samples);
    float snr_score = std::clamp(factors.snr / 30.0f, 0.0f, 1.0f);

    // Factor 2: Harmonicity (how harmonic is the signal)
    factors.harmonicity = calculateHarmonicity(samples, num_samples, detected_frequency);
    float harmonicity_score = std::clamp(factors.harmonicity, 0.0f, 1.0f);

    // Factor 3: Autocorrelation peak clarity
    factors.peak_clarity = calculatePeakClarity(samples, num_samples);
    float peak_score = std::clamp(factors.peak_clarity, 0.0f, 1.0f);

    // Factor 4: Temporal stability (pitch consistent over time)
    factors.temporal_stability = calculateTemporalStability(samples, num_samples);
    float stability_score = std::clamp(factors.temporal_stability, 0.0f, 1.0f);

    // Factor 5: Frequency range validity
    factors.range_validity = (detected_frequency >= kMinPitchHz &&
                             detected_frequency <= kMaxPitchHz) ? 1.0f : 0.0f;

    // Weighted combination
    float confidence = (
        snr_score * 0.25f +
        harmonicity_score * 0.25f +
        peak_score * 0.20f +
        stability_score * 0.20f +
        factors.range_validity * 0.10f
    );

    // Penalize if signal is too quiet
    float rms = calculateRMS(samples, num_samples);
    if (rms < 0.001f) {
        confidence *= 0.5f; // Halve confidence for very quiet signals
    }

    return std::clamp(confidence, 0.0f, 1.0f);
}

struct ConfidenceFactors {
    float snr = 0.0f;
    float harmonicity = 0.0f;
    float peak_clarity = 0.0f;
    float temporal_stability = 0.0f;
    float range_validity = 0.0f;
};
```

**Improvement 3: Edge Case Handling & Validation**
```cpp
// BEFORE:
// No edge case handling

// AFTER:
PitchResult trackPitch(...) const noexcept {
    PitchResult result;

    // Edge case 1: Too few samples
    if (num_samples < kMinSamplesForPitch) {
        RTLogger::warning("Too few samples for pitch tracking: {} < {}",
                        num_samples, kMinSamplesForPitch);
        result.is_valid = false;
        return result;
    }

    // Edge case 2: Silence
    if (isSilence(audio_samples, num_samples)) {
        RTLogger::debug("Silence detected, no pitch");
        result.is_valid = false;
        return result;
    }

    // Edge case 3: Noise only (no pitch)
    if (isNoiseOnly(audio_samples, num_samples)) {
        RTLogger::debug("Noise only detected, no pitch");
        result.is_valid = false;
        return result;
    }

    // Edge case 4: Multiple pitches (chord)
    auto pitch_candidates = detectMultiplePitches(audio_samples, num_samples, sample_rate_hz);
    if (pitch_candidates.size() > 1) {
        RTLogger::warning("Multiple pitches detected, using strongest");
        // Use strongest pitch
        result = selectStrongestPitch(pitch_candidates);
        result.confidence *= 0.7f; // Reduce confidence for multiple pitches
        return result;
    }

    // Normal case: single pitch
    result = detectSinglePitch(audio_samples, num_samples, sample_rate_hz);

    // Validate result
    if (!validatePitchResult(result, sample_rate_hz)) {
        result.is_valid = false;
        return result;
    }

    return result;
}

bool validatePitchResult(const PitchResult& result, float sample_rate_hz) const noexcept {
    // Check frequency is in valid range
    if (result.frequency_hz < kMinPitchHz || result.frequency_hz > kMaxPitchHz) {
        RTLogger::error("Pitch frequency out of range: {} Hz", result.frequency_hz);
        return false;
    }

    // Check MIDI note is valid
    if (result.midi_note < 0 || result.midi_note > 127) {
        RTLogger::error("MIDI note out of range: {}", result.midi_note);
        return false;
    }

    // Check cents offset is reasonable
    if (std::abs(result.cents_offset) > 50.0f) {
        RTLogger::warning("Large cents offset: {}", result.cents_offset);
    }

    // Check confidence is valid
    if (result.confidence < 0.0f || result.confidence > 1.0f) {
        RTLogger::error("Invalid confidence: {}", result.confidence);
        return false;
    }

    // Check frequency matches MIDI note (within tolerance)
    float expected_freq = midiToFrequency(result.midi_note);
    float freq_diff = std::abs(result.frequency_hz - expected_freq);
    float cents_diff = 1200.0f * std::log2(result.frequency_hz / expected_freq);

    if (std::abs(cents_diff - result.cents_offset) > 5.0f) {
        RTLogger::warning("Cents offset mismatch: calculated {} vs stored {}",
                        cents_diff, result.cents_offset);
    }

    return true;
}
```

---

### AudioValidator.cpp/h

**Current Issues:**
- Basic validation only
- Simplistic quality score
- No detailed error messages
- Missing validation checks

**Improvement 1: Comprehensive Validation Checks**
```cpp
// BEFORE:
// Basic checks only

// AFTER:
ValidationResult validate(...) const noexcept {
    ValidationResult result;

    // Check 1: Null pointer
    if (!audio_samples) {
        result.is_valid = false;
        result.error_message = "audio_samples is null";
        return result;
    }

    // Check 2: Sample count
    if (num_samples == 0) {
        result.is_valid = false;
        result.error_message = "num_samples is zero";
        return result;
    }

    // Check 3: Duration
    result.duration_ms = (static_cast<float>(num_samples) / sample_rate_hz) * 1000.0f;
    if (result.duration_ms < kMinValidDurationMs) {
        result.is_valid = false;
        result.is_too_short = true;
        result.error_message = "Duration too short: " + std::to_string(result.duration_ms) +
                              "ms < " + std::to_string(kMinValidDurationMs) + "ms";
        return result;
    }

    // Check 4: Sample rate
    if (sample_rate_hz <= 0.0f || sample_rate_hz > 192000.0f) {
        result.is_valid = false;
        result.error_message = "Invalid sample rate: " + std::to_string(sample_rate_hz);
        return result;
    }

    // Check 5: Silence detection
    result.has_silence = detectSilence(audio_samples, num_samples);
    if (result.has_silence && isEntirelySilent(audio_samples, num_samples)) {
        result.is_valid = false;
        result.error_message = "Audio is entirely silent";
        return result;
    }

    // Check 6: Clipping detection
    result.is_clipped = detectClipping(audio_samples, num_samples);
    if (result.is_clipped && isSeverelyClipped(audio_samples, num_samples)) {
        result.is_valid = false;
        result.error_message = "Audio is severely clipped";
        return result;
    }

    // Check 7: DC offset
    float dc_offset = calculateDCOffset(audio_samples, num_samples);
    if (std::abs(dc_offset) > 0.1f) {
        result.warnings.push_back("DC offset detected: " + std::to_string(dc_offset));
    }

    // Check 8: Signal level
    result.signal_level_db = computeSignalLevel(audio_samples, num_samples);
    if (result.signal_level_db < -60.0f) {
        result.is_too_quiet = true;
        result.warnings.push_back("Signal too quiet: " + std::to_string(result.signal_level_db) + "dB");
    }

    // Check 9: Noise floor
    result.noise_floor_db = estimateNoiseFloor(audio_samples, num_samples);

    // Check 10: SNR
    result.snr_db = computeSNR(audio_samples, num_samples, result.noise_floor_db);
    if (result.snr_db < kMinSNRDb) {
        result.has_low_snr = true;
        result.warnings.push_back("Low SNR: " + std::to_string(result.snr_db) + "dB < " +
                                 std::to_string(kMinSNRDb) + "dB");
    }

    // Check 11: Peak level
    result.peak_level = calculatePeakLevel(audio_samples, num_samples);

    // Check 12: Frequency response (detect aliasing, etc.)
    auto freq_response = analyzeFrequencyResponse(audio_samples, num_samples, sample_rate_hz);
    if (freq_response.has_aliasing) {
        result.warnings.push_back("Possible aliasing detected");
    }

    // Overall validity
    result.is_valid = (
        !result.is_too_short &&
        !result.is_clipped &&
        result.snr_db >= kMinSNRDb &&
        result.signal_level_db >= -60.0f
    );

    return result;
}
```

**Improvement 2: Detailed Quality Metrics**
```cpp
// BEFORE:
float quality_score() const noexcept {
    float score = 1.0f;
    if (has_silence) score *= 0.8f;
    if (is_clipped) score *= 0.7f;
    // ...
    return score;
}

// AFTER:
struct QualityMetrics {
    float overall_score = 0.0f;
    float signal_quality = 0.0f;
    float noise_quality = 0.0f;
    float dynamic_range = 0.0f;
    float frequency_response = 0.0f;

    std::vector<std::string> issues;
    std::vector<std::string> recommendations;
};

QualityMetrics calculateQualityMetrics(const ValidationResult& validation) const noexcept {
    QualityMetrics metrics;

    // Signal quality (level, clipping, DC offset)
    float signal_score = 1.0f;
    if (validation.is_clipped) {
        signal_score *= 0.5f;
        metrics.issues.push_back("Clipping detected");
        metrics.recommendations.push_back("Reduce input gain");
    }
    if (validation.signal_level_db < -40.0f) {
        signal_score *= 0.7f;
        metrics.issues.push_back("Signal too quiet");
        metrics.recommendations.push_back("Increase input gain");
    }
    if (validation.signal_level_db > -3.0f && !validation.is_clipped) {
        signal_score *= 0.9f;
        metrics.issues.push_back("Signal near clipping");
        metrics.recommendations.push_back("Reduce input gain slightly");
    }
    metrics.signal_quality = signal_score;

    // Noise quality (SNR, noise floor)
    float noise_score = 1.0f;
    if (validation.has_low_snr) {
        noise_score = std::clamp(validation.snr_db / 30.0f, 0.0f, 1.0f);
        metrics.issues.push_back("Low signal-to-noise ratio");
        metrics.recommendations.push_back("Improve recording environment");
    }
    if (validation.noise_floor_db > -50.0f) {
        noise_score *= 0.8f;
        metrics.issues.push_back("High noise floor");
        metrics.recommendations.push_back("Check recording equipment");
    }
    metrics.noise_quality = noise_score;

    // Dynamic range
    float dynamic_range_score = 1.0f;
    float range_db = validation.signal_level_db - validation.noise_floor_db;
    if (range_db < 20.0f) {
        dynamic_range_score = range_db / 20.0f;
        metrics.issues.push_back("Limited dynamic range");
    }
    metrics.dynamic_range = dynamic_range_score;

    // Frequency response (would need FFT analysis)
    metrics.frequency_response = 1.0f; // Placeholder

    // Overall weighted score
    metrics.overall_score = (
        metrics.signal_quality * 0.4f +
        metrics.noise_quality * 0.3f +
        metrics.dynamic_range * 0.2f +
        metrics.frequency_response * 0.1f
    );

    return metrics;
}
```

**Improvement 3: Actionable Error Messages & Recommendations**
```cpp
// BEFORE:
std::string error_message; // Basic string

// AFTER:
struct ValidationReport {
    bool is_valid = false;
    std::vector<ValidationIssue> issues;
    std::vector<Recommendation> recommendations;
    QualityMetrics quality;

    std::string generateReport() const {
        std::string report;
        report += "=== Audio Validation Report ===\n\n";

        report += "Status: " + std::string(is_valid ? "VALID" : "INVALID") + "\n\n";

        if (!issues.empty()) {
            report += "Issues Found:\n";
            for (const auto& issue : issues) {
                report += "  [" + severityToString(issue.severity) + "] " +
                         issue.description + "\n";
                if (!issue.details.empty()) {
                    report += "    Details: " + issue.details + "\n";
                }
            }
            report += "\n";
        }

        if (!recommendations.empty()) {
            report += "Recommendations:\n";
            for (const auto& rec : recommendations) {
                report += "  - " + rec.action + "\n";
                if (!rec.explanation.empty()) {
                    report += "    " + rec.explanation + "\n";
                }
            }
            report += "\n";
        }

        report += "Quality Metrics:\n";
        report += "  Overall Score: " + std::to_string(quality.overall_score) + "\n";
        report += "  Signal Quality: " + std::to_string(quality.signal_quality) + "\n";
        report += "  Noise Quality: " + std::to_string(quality.noise_quality) + "\n";
        report += "  Dynamic Range: " + std::to_string(quality.dynamic_range) + "\n";

        return report;
    }
};

struct ValidationIssue {
    enum Severity { Error, Warning, Info };
    Severity severity;
    std::string description;
    std::string details;
    float impact_score; // 0.0 to 1.0
};

struct Recommendation {
    std::string action;
    std::string explanation;
    enum Priority { High, Medium, Low };
    Priority priority;
};

ValidationReport generateReport(const ValidationResult& validation) const noexcept {
    ValidationReport report;
    report.is_valid = validation.is_valid;

    // Convert validation result to issues
    if (validation.is_too_short) {
        ValidationIssue issue;
        issue.severity = ValidationIssue::Error;
        issue.description = "Audio duration too short";
        issue.details = std::to_string(validation.duration_ms) + "ms < " +
                       std::to_string(kMinValidDurationMs) + "ms";
        issue.impact_score = 1.0f;
        report.issues.push_back(issue);

        Recommendation rec;
        rec.action = "Record longer audio segment";
        rec.explanation = "Minimum duration is " + std::to_string(kMinValidDurationMs) + "ms";
        rec.priority = Recommendation::High;
        report.recommendations.push_back(rec);
    }

    if (validation.is_clipped) {
        ValidationIssue issue;
        issue.severity = ValidationIssue::Error;
        issue.description = "Audio clipping detected";
        issue.details = "Peak level: " + std::to_string(validation.peak_level);
        issue.impact_score = 0.9f;
        report.issues.push_back(issue);

        Recommendation rec;
        rec.action = "Reduce input gain by " +
                    std::to_string(calculateGainReduction(validation.peak_level)) + "dB";
        rec.explanation = "Clipping causes distortion and reduces quality";
        rec.priority = Recommendation::High;
        report.recommendations.push_back(rec);
    }

    if (validation.has_low_snr) {
        ValidationIssue issue;
        issue.severity = ValidationIssue::Warning;
        issue.description = "Low signal-to-noise ratio";
        issue.details = "SNR: " + std::to_string(validation.snr_db) + "dB";
        issue.impact_score = 0.7f;
        report.issues.push_back(issue);

        Recommendation rec;
        rec.action = "Improve recording environment";
        rec.explanation = "Reduce background noise or increase signal level";
        rec.priority = Recommendation::Medium;
        report.recommendations.push_back(rec);
    }

    // Calculate quality metrics
    report.quality = calculateQualityMetrics(validation);

    return report;
}
```

---

### KellyBrain.cpp/h

**Current Issues:**
- No error handling in conversions
- No input validation
- Type conversion safety issues
- Silent failures

**Improvement 1: Type-Safe Conversions with Error Handling**
```cpp
// BEFORE:
KellyTypesIntentResult convertFromLegacyIntentResult(const IntentResult& legacy) {
    // Direct field copying, no validation
    unified.sourceWound.intensity = legacy.sourceWound.intensity;
    // ...
}

// AFTER:
enum class ConversionError {
    Success,
    InvalidEmotionId,
    InvalidIntensity,
    InvalidTempo,
    InvalidMode,
    TypeMismatch
};

Result<KellyTypesIntentResult, ConversionError> convertFromLegacyIntentResult(
    const IntentResult& legacy
) noexcept {
    KellyTypesIntentResult unified;

    // Validate and convert sourceWound
    if (legacy.sourceWound.intensity < 0.0f || legacy.sourceWound.intensity > 1.0f) {
        return Result<KellyTypesIntentResult, ConversionError>::error(
            ConversionError::InvalidIntensity,
            "sourceWound.intensity out of range: " + std::to_string(legacy.sourceWound.intensity)
        );
    }
    unified.sourceWound.intensity = legacy.sourceWound.intensity;
    unified.sourceWound.urgency = legacy.sourceWound.intensity; // Validated above

    // Validate and convert emotion
    if (legacy.emotion.id < 0) {
        return Result<KellyTypesIntentResult, ConversionError>::error(
            ConversionError::InvalidEmotionId,
            "emotion.id is negative: " + std::to_string(legacy.emotion.id)
        );
    }

    // Validate emotion category enum
    if (legacy.emotion.categoryEnum < 0 ||
        legacy.emotion.categoryEnum >= static_cast<int>(EmotionCategory::Count)) {
        return Result<KellyTypesIntentResult, ConversionError>::error(
            ConversionError::InvalidEmotionId,
            "Invalid emotion category: " + std::to_string(static_cast<int>(legacy.emotion.categoryEnum))
        );
    }

    unified.sourceWound.primaryEmotion.id = legacy.emotion.id;
    unified.sourceWound.primaryEmotion.name = legacy.emotion.name;
    unified.sourceWound.primaryEmotion.categoryEnum =
        static_cast<EmotionCategory>(legacy.emotion.categoryEnum);

    // Validate emotion values
    if (legacy.emotion.valence < -1.0f || legacy.emotion.valence > 1.0f) {
        RTLogger::warning("Emotion valence out of range, clamping: {}", legacy.emotion.valence);
        unified.sourceWound.primaryEmotion.valence = std::clamp(legacy.emotion.valence, -1.0f, 1.0f);
    } else {
        unified.sourceWound.primaryEmotion.valence = legacy.emotion.valence;
    }

    // Similar validation for arousal, dominance, intensity...

    // Validate tempo
    if (legacy.tempo < 0.1f || legacy.tempo > 4.0f) {
        return Result<KellyTypesIntentResult, ConversionError>::error(
            ConversionError::InvalidTempo,
            "Tempo out of range: " + std::to_string(legacy.tempo)
        );
    }
    unified.tempoBpm = static_cast<int>(120 * legacy.tempo);

    // Validate mode
    if (legacy.mode != "major" && legacy.mode != "minor") {
        RTLogger::warning("Invalid mode '{}', defaulting to major", legacy.mode);
        unified.mode = "major";
    } else {
        unified.mode = legacy.mode;
    }

    // Convert rule breaks with validation
    unified.ruleBreaks.clear();
    for (const auto& rb : legacy.ruleBreaks) {
        auto converted_rb = convertRuleBreak(rb);
        if (converted_rb.hasValue()) {
            unified.ruleBreaks.push_back(converted_rb.value());
        } else {
            RTLogger::warning("Failed to convert rule break: {}", converted_rb.error());
        }
    }

    return Result<KellyTypesIntentResult, ConversionError>::success(unified);
}
```

**Improvement 2: Input Validation for All Public Methods**
```cpp
// BEFORE:
KellyTypesIntentResult fromWound(const KellyTypesWound& wound) {
    Wound legacyWound = convertToLegacyWound(wound);
    IntentResult legacyResult = pipeline_->process(legacyWound);
    return convertFromLegacyIntentResult(legacyResult);
}

// AFTER:
Result<KellyTypesIntentResult, ProcessingError> fromWound(
    const KellyTypesWound& wound
) noexcept {
    // Validate wound
    auto wound_validation = validateWound(wound);
    if (!wound_validation.isValid()) {
        return Result<KellyTypesIntentResult, ProcessingError>::error(
            ProcessingError::InvalidInput,
            "Wound validation failed: " + wound_validation.errorMessage()
        );
    }

    // Convert with error handling
    auto legacy_wound_result = convertToLegacyWound(wound);
    if (!legacy_wound_result.hasValue()) {
        return Result<KellyTypesIntentResult, ProcessingError>::error(
            ProcessingError::ConversionError,
            "Failed to convert wound: " + legacy_wound_result.error()
        );
    }

    // Process with error handling
    auto legacy_result = pipeline_->process(legacy_wound_result.value());
    if (!legacy_result.hasValue()) {
        return Result<KellyTypesIntentResult, ProcessingError>::error(
            ProcessingError::ProcessingFailed,
            "Pipeline processing failed: " + legacy_result.error()
        );
    }

    // Convert back with error handling
    auto unified_result = convertFromLegacyIntentResult(legacy_result.value());
    if (!unified_result.hasValue()) {
        return Result<KellyTypesIntentResult, ProcessingError>::error(
            ProcessingError::ConversionError,
            "Failed to convert result: " + unified_result.error()
        );
    }

    return Result<KellyTypesIntentResult, ProcessingError>::success(unified_result.value());
}

WoundValidation validateWound(const KellyTypesWound& wound) const noexcept {
    WoundValidation validation;

    // Check required fields
    if (wound.description.empty()) {
        validation.addError("description is empty");
    }

    // Validate intensity
    if (wound.intensity < 0.0f || wound.intensity > 1.0f) {
        validation.addError("intensity out of range: " + std::to_string(wound.intensity));
    }

    // Validate urgency
    if (wound.urgency < 0.0f || wound.urgency > 1.0f) {
        validation.addError("urgency out of range: " + std::to_string(wound.urgency));
    }

    // Validate emotion if present
    if (wound.primaryEmotion.id < 0) {
        validation.addError("emotion.id is negative");
    }

    // Validate emotion values
    if (wound.primaryEmotion.valence < -1.0f || wound.primaryEmotion.valence > 1.0f) {
        validation.addWarning("emotion.valence out of range, will be clamped");
    }

    return validation;
}
```

**Improvement 3: Error Recovery & Fallback Mechanisms**
```cpp
// BEFORE:
// No error recovery, fails immediately

// AFTER:
KellyTypesIntentResult fromWound(const KellyTypesWound& wound) {
    // Try primary method
    auto result = fromWoundWithValidation(wound);
    if (result.hasValue()) {
        return result.value();
    }

    // Fallback 1: Try with default emotion if emotion validation failed
    if (result.error() == ProcessingError::InvalidEmotion) {
        RTLogger::info("Falling back to default emotion");
        KellyTypesWound fallback_wound = wound;
        fallback_wound.primaryEmotion = getDefaultEmotion();
        auto fallback_result = fromWoundWithValidation(fallback_wound);
        if (fallback_result.hasValue()) {
            return fallback_result.value();
        }
    }

    // Fallback 2: Try with clamped values if range validation failed
    if (result.error() == ProcessingError::InvalidRange) {
        RTLogger::info("Falling back with clamped values");
        KellyTypesWound clamped_wound = clampWoundValues(wound);
        auto clamped_result = fromWoundWithValidation(clamped_wound);
        if (clamped_result.hasValue()) {
            return clamped_result.value();
        }
    }

    // Fallback 3: Return minimal valid result
    RTLogger::warning("All methods failed, returning minimal result");
    return createMinimalIntentResult(wound);
}

KellyTypesIntentResult createMinimalIntentResult(const KellyTypesWound& wound) const noexcept {
    KellyTypesIntentResult result;

    // Set minimal valid values
    result.sourceWound.description = wound.description;
    result.sourceWound.intensity = std::clamp(wound.intensity, 0.0f, 1.0f);
    result.sourceWound.urgency = std::clamp(wound.urgency, 0.0f, 1.0f);

    // Use default emotion if available
    if (wound.primaryEmotion.id >= 0) {
        result.sourceWound.primaryEmotion = wound.primaryEmotion;
    } else {
        result.sourceWound.primaryEmotion = getDefaultEmotion();
    }

    // Set default musical parameters
    result.tempoBpm = 120;
    result.mode = "major";
    result.key = "C";
    result.timeSignature = {4, 4};

    // Set low confidence to indicate fallback
    result.confidence = 0.3f;

    return result;
}
```

---

### IntentIRAdapter.cpp/h

**Current Issues:**
- No validation error reporting
- No error propagation from Rust
- Type conversion safety

**Improvement 1: Detailed Validation Error Reporting**
```cpp
// BEFORE:
bool prepareIntentFrame(IntentFrame& frame) {
    // Call Rust validator, but no error details
    clamp_intent_frame_ffi(&frame);
    return true; // Always returns true
}

// AFTER:
enum class ValidationError {
    Success,
    InvalidVersion,
    OutOfRange,
    InvalidEnum,
    MissingRequiredField,
    RustValidationFailed
};

struct ValidationReport {
    bool isValid() const { return errors.empty(); }
    std::vector<ValidationError> errors;
    std::vector<std::string> errorMessages;
    std::vector<std::string> warnings;

    std::string generateReport() const {
        std::string report = "Validation Report:\n";
        for (const auto& msg : errorMessages) {
            report += "  ERROR: " + msg + "\n";
        }
        for (const auto& msg : warnings) {
            report += "  WARNING: " + msg + "\n";
        }
        return report;
    }
};

Result<ValidationReport, ValidationError> prepareIntentFrame(
    IntentFrame& frame
) noexcept {
    ValidationReport report;

    // Pre-validation checks
    if (frame.meta.ir_version != INTENT_IR_VERSION) {
        report.errors.push_back(ValidationError::InvalidVersion);
        report.errorMessages.push_back(
            "Invalid IR version: " + std::to_string(frame.meta.ir_version) +
            " (expected " + std::to_string(INTENT_IR_VERSION) + ")"
        );
    }

    // Validate emotion values before Rust call
    if (frame.emotion.valence < -1.0f || frame.emotion.valence > 1.0f) {
        report.warnings.push_back("Emotion valence out of range, will be clamped");
    }

    // Call Rust validator
    auto rust_result = clamp_intent_frame_ffi(&frame);
    if (rust_result.error_code != 0) {
        report.errors.push_back(ValidationError::RustValidationFailed);
        report.errorMessages.push_back(
            "Rust validation failed with code: " + std::to_string(rust_result.error_code)
        );

        // Get detailed error from Rust if available
        if (rust_result.error_message) {
            report.errorMessages.push_back("Rust error: " +
                                         std::string(rust_result.error_message));
        }

        return Result<ValidationReport, ValidationError>::error(
            ValidationError::RustValidationFailed,
            report
        );
    }

    // Post-validation checks
    if (frame.emotion.valence < -1.0f || frame.emotion.valence > 1.0f) {
        report.errors.push_back(ValidationError::OutOfRange);
        report.errorMessages.push_back("Emotion valence still out of range after clamping");
    }

    if (!report.isValid()) {
        return Result<ValidationReport, ValidationError>::error(
            ValidationError::RustValidationFailed,
            report
        );
    }

    return Result<ValidationReport, ValidationError>::success(report);
}
```

**Improvement 2: Type-Safe Conversions with Overflow Protection**
```cpp
// BEFORE:
int tempoBiasToBPM(float tempo_bias) {
    float normalized = std::clamp(tempo_bias, -1.0f, 1.0f);
    return static_cast<int>(120.0f + (normalized * 60.0f));
}

// AFTER:
Result<int, ConversionError> tempoBiasToBPM(float tempo_bias) noexcept {
    // Validate input
    if (std::isnan(tempo_bias) || std::isinf(tempo_bias)) {
        return Result<int, ConversionError>::error(
            ConversionError::InvalidInput,
            "tempo_bias is NaN or Inf"
        );
    }

    // Clamp to valid range
    float normalized = std::clamp(tempo_bias, -1.0f, 1.0f);

    // Calculate with overflow protection
    float bpm_float = 120.0f + (normalized * 60.0f);

    // Check for integer overflow
    if (bpm_float < static_cast<float>(std::numeric_limits<int>::min()) ||
        bpm_float > static_cast<float>(std::numeric_limits<int>::max())) {
        return Result<int, ConversionError>::error(
            ConversionError::Overflow,
            "BPM calculation would overflow"
        );
    }

    int bpm = static_cast<int>(std::round(bpm_float));

    // Validate result
    if (bpm < 1 || bpm > 300) {
        RTLogger::warning("Calculated BPM out of typical range: {}", bpm);
    }

    return Result<int, ConversionError>::success(bpm);
}

Result<std::string, ConversionError> modePreferenceToMode(int8_t mode_preference) noexcept {
    // Validate input
    if (mode_preference > 1 || mode_preference < -1) {
        return Result<std::string, ConversionError>::error(
            ConversionError::InvalidInput,
            "mode_preference out of range: " + std::to_string(mode_preference)
        );
    }

    if (mode_preference > 0) {
        return Result<std::string, ConversionError>::success("major");
    } else if (mode_preference < 0) {
        return Result<std::string, ConversionError>::success("minor");
    } else {
        return Result<std::string, ConversionError>::success("major"); // Default
    }
}
```

**Improvement 3: Rust Error Propagation with Context**
```cpp
// BEFORE:
// No error propagation from Rust

// AFTER:
struct RustError {
    int error_code;
    const char* error_message;
    const char* error_context;
};

Result<IntentResult, ProcessingError> convertIntentIRToIntentResult(
    const IntentFrame& frame
) noexcept {
    IntentResult result;

    // Validate frame before conversion
    auto validation = validateIntentFrame(frame);
    if (!validation.isValid()) {
        return Result<IntentResult, ProcessingError>::error(
            ProcessingError::InvalidInput,
            "Frame validation failed: " + validation.errorMessage()
        );
    }

    // Convert emotion with error handling
    auto emotion_result = convertEmotion(frame.emotion);
    if (!emotion_result.hasValue()) {
        return Result<IntentResult, ProcessingError>::error(
            ProcessingError::ConversionError,
            "Emotion conversion failed: " + emotion_result.error()
        );
    }
    result.emotion = emotion_result.value();

    // Convert tempo with error handling
    auto tempo_result = tempoBiasToBPM(frame.music.tempo_bias);
    if (!tempo_result.hasValue()) {
        return Result<IntentResult, ProcessingError>::error(
            ProcessingError::ConversionError,
            "Tempo conversion failed: " + tempo_result.error()
        );
    }
    result.tempoBpm = tempo_result.value();

    // Convert mode with error handling
    auto mode_result = modePreferenceToMode(frame.music.mode_preference);
    if (!mode_result.hasValue()) {
        return Result<IntentResult, ProcessingError>::error(
            ProcessingError::ConversionError,
            "Mode conversion failed: " + mode_result.error()
        );
    }
    result.mode = mode_result.value();

    // ... convert other fields with similar error handling ...

    return Result<IntentResult, ProcessingError>::success(result);
}

Result<EmotionNode, ConversionError> convertEmotion(const IntentIREmotion& ir_emotion) noexcept {
    EmotionNode emotion;

    // Validate and convert valence
    if (ir_emotion.valence < -1.0f || ir_emotion.valence > 1.0f) {
        return Result<EmotionNode, ConversionError>::error(
            ConversionError::OutOfRange,
            "Valence out of range: " + std::to_string(ir_emotion.valence)
        );
    }
    emotion.valence = ir_emotion.valence;

    // Similar validation for arousal, dominance, intensity...

    // Validate discrete_id if present
    if (ir_emotion.discrete_id >= 0) {
        // Look up emotion in thesaurus
        auto emotion_opt = thesaurus().findById(ir_emotion.discrete_id);
        if (!emotion_opt) {
            RTLogger::warning("Emotion ID {} not found in thesaurus", ir_emotion.discrete_id);
        } else {
            emotion.id = emotion_opt->id;
            emotion.name = emotion_opt->name;
        }
    }

    return Result<EmotionNode, ConversionError>::success(emotion);
}
```

---

### CMakeLists.txt

**Current Issues:**
- Unclear error messages
- No build validation
- Complex dependency management

**Improvement 1: Clear Error Messages with Solutions**
```cmake
# BEFORE:
find_package(Qt6 COMPONENTS Core Widgets REQUIRED)
# Fails with cryptic error if not found

# AFTER:
# Check Qt6 with helpful error message
find_package(Qt6 COMPONENTS Core Widgets QUIET)
if(NOT Qt6_FOUND)
    message(FATAL_ERROR
        "\n"
        "========================================\n"
        "Qt6 not found!\n"
        "========================================\n"
        "\n"
        "Installation options:\n"
        "  macOS (Homebrew): brew install qt@6\n"
        "  Linux (apt):      sudo apt-get install qt6-base-dev\n"
        "  Windows:          Download from https://www.qt.io/download\n"
        "\n"
        "After installation, set Qt6_DIR if needed:\n"
        "  cmake .. -DQt6_DIR=/path/to/qt6/lib/cmake/Qt6\n"
        "\n"
    )
endif()

# Check Rust with helpful error message
find_program(CARGO cargo)
if(NOT CARGO)
    message(FATAL_ERROR
        "\n"
        "========================================\n"
        "Rust/Cargo not found!\n"
        "========================================\n"
        "\n"
        "Install Rust:\n"
        "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n"
        "\n"
        "Then restart your terminal and try again.\n"
        "\n"
    )
endif()

# Check JUCE with helpful error message
if(NOT EXISTS "${BUILD_ROOT}/external/JUCE")
    message(FATAL_ERROR
        "\n"
        "========================================\n"
        "JUCE not found!\n"
        "========================================\n"
        "\n"
        "Expected location: ${BUILD_ROOT}/external/JUCE\n"
        "\n"
        "To fix:\n"
        "  1. Clone JUCE: git clone https://github.com/juce-framework/JUCE.git ${BUILD_ROOT}/external/JUCE\n"
        "  2. Or set JUCE_ROOT: cmake .. -DJUCE_ROOT=/path/to/JUCE\n"
        "\n"
    )
endif()
```

**Improvement 2: Build Validation Script**
```cmake
# Add build validation function
function(validate_build_config)
    message(STATUS "Validating build configuration...")

    # Check compiler
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
            message(WARNING "GCC version ${CMAKE_CXX_COMPILER_VERSION} may not support all C++20 features")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
            message(WARNING "Clang version ${CMAKE_CXX_COMPILER_VERSION} may not support all C++20 features")
        endif()
    endif()

    # Check CMake version
    if(CMAKE_VERSION VERSION_LESS "3.27")
        message(FATAL_ERROR "CMake 3.27+ required, found ${CMAKE_VERSION}")
    endif()

    # Check required tools
    find_program(CARGO cargo REQUIRED)
    find_program(RUSTC rustc REQUIRED)

    # Check Rust version
    execute_process(
        COMMAND ${RUSTC} --version
        OUTPUT_VARIABLE RUST_VERSION
        ERROR_QUIET
    )
    message(STATUS "Rust version: ${RUST_VERSION}")

    # Validate build options
    if(BUILD_DESKTOP AND NOT BUILD_KMIDI_CORE)
        message(FATAL_ERROR "BUILD_DESKTOP requires BUILD_KMIDI_CORE=ON")
    endif()

    if(BUILD_PLUGINS AND NOT BUILD_KMIDI_CORE)
        message(FATAL_ERROR "BUILD_PLUGINS requires BUILD_KMIDI_CORE=ON")
    endif()

    message(STATUS "Build configuration validated successfully")
endfunction()

# Run validation
validate_build_config()
```

**Improvement 3: Dependency Documentation & Checking**
```cmake
# Add dependency documentation
function(document_dependencies)
    message(STATUS "\n=== Build Dependencies ===\n")

    message(STATUS "Required:")
    message(STATUS "  - CMake 3.27+")
    message(STATUS "  - C++20 compiler (GCC 9+, Clang 10+, MSVC 2019+)")
    message(STATUS "  - Rust toolchain (cargo, rustc)")
    message(STATUS "  - JUCE (in build/external/JUCE)")

    message(STATUS "\nOptional:")
    message(STATUS "  - Qt6 (for desktop build)")
    message(STATUS "  - RTNeural (for ML inference)")
    message(STATUS "  - ONNX Runtime (for ONNX models)")

    message(STATUS "\nBuild Options:")
    message(STATUS "  BUILD_KMIDI_CORE=${BUILD_KMIDI_CORE}")
    message(STATUS "  BUILD_DESKTOP=${BUILD_DESKTOP}")
    message(STATUS "  BUILD_PLUGINS=${BUILD_PLUGINS}")
    message(STATUS "  BUILD_TESTS=${BUILD_TESTS}")
    message(STATUS "  ENABLE_RTNEURAL=${ENABLE_RTNEURAL}")
    message(STATUS "  ENABLE_ONNX_RUNTIME=${ENABLE_ONNX_RUNTIME}")
    message(STATUS "\n")
endfunction()

# Check all dependencies
function(check_dependencies)
    set(MISSING_DEPS "")

    # Check Rust
    find_program(CARGO cargo)
    if(NOT CARGO)
        list(APPEND MISSING_DEPS "Rust (cargo)")
    endif()

    # Check JUCE
    if(NOT EXISTS "${BUILD_ROOT}/external/JUCE")
        list(APPEND MISSING_DEPS "JUCE")
    endif()

    # Check Qt6 if building desktop
    if(BUILD_DESKTOP)
        find_package(Qt6 COMPONENTS Core Widgets QUIET)
        if(NOT Qt6_FOUND)
            list(APPEND MISSING_DEPS "Qt6")
        endif()
    endif()

    # Report missing dependencies
    if(MISSING_DEPS)
        message(WARNING "Missing dependencies:")
        foreach(DEP ${MISSING_DEPS})
            message(WARNING "  - ${DEP}")
        endforeach()
        message(WARNING "\nRun 'cmake .. --help' for installation instructions")
    else()
        message(STATUS "All required dependencies found")
    endif()
endfunction()

# Run checks
document_dependencies()
check_dependencies()
```

---

## Documentation Improvements

### VOCAL_GENERATION_ROBUSTNESS_PLAN.md

**Improvement 1: Implementation Status Tracking**
```markdown
## Implementation Status

| Improvement | Status | Completed | Tests | PR |
|------------|--------|-----------|-------|-----|
| Real Pitch Tracking | ✅ Complete | 2026-01-20 | ✅ | #123 |
| Enhanced Segmentation | 🚧 In Progress | - | ⏳ | #124 |
| Input Validation | 📋 Planned | - | ❌ | - |
| Error Recovery | 📋 Planned | - | ❌ | - |

**Legend:**
- ✅ Complete
- 🚧 In Progress
- 📋 Planned
- ❌ Not Started
```

**Improvement 2: Complete Code Examples**
```markdown
## Code Examples

### Before/After: Pitch Tracking

**Before (Placeholder):**
```cpp
target.midi_note = 60; // PLACEHOLDER
target.confidence = 0.7f; // PLACEHOLDER
```

**After (Real Implementation):**
```cpp
auto pitch_result = pitch_tracker_->trackPitch(
    audio_samples + start_sample,
    end_sample - start_sample,
    sample_rate_hz
);

if (pitch_result.is_valid) {
    target.midi_note = pitch_result.midi_note;
    target.cents_offset = pitch_result.cents_offset;
    target.confidence = pitch_result.confidence;
} else {
    // Fallback to phoneme-based estimation
    target.midi_note = estimatePitchFromPhoneme(phoneme.phoneme);
    target.confidence = 0.5f; // Lower confidence for fallback
}
```

### Usage Example
```cpp
PRROTEngine engine;
engine.initialize();

auto result = engine.processAudioSegment(
    audio_data, num_samples, 44100.0f, 120.0f
);

if (result.hasValue()) {
    auto control_data = result.value();
    if (control_data.quality_metrics.is_usable) {
        // Use control data
    } else {
        // Handle low quality
    }
} else {
    // Handle error
    RTLogger::error("Processing failed: {}", result.error());
}
```
```

**Improvement 3: Measurable Success Criteria**
```markdown
## Success Criteria & Metrics

### Phase 1: Critical Fixes

**Pitch Tracking:**
- ✅ Real pitch detection (not hardcoded)
- ✅ Accuracy: >90% for clean signals, >70% for noisy signals
- ✅ Latency: <10ms for 1024 samples @ 44.1kHz
- ✅ Test coverage: >80%

**Phoneme Segmentation:**
- ✅ Multi-feature segmentation
- ✅ Accuracy: >85% for clear speech
- ✅ Confidence calculation: Real values, not placeholders
- ✅ Test coverage: >80%

**Input Validation:**
- ✅ All inputs validated
- ✅ Error messages: 100% of failures have messages
- ✅ Quality scores: Calculated for all inputs
- ✅ Test coverage: >90%

### Metrics Dashboard

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Test Coverage | >80% | 15% | ❌ |
| Error Handling | 100% | 20% | ❌ |
| Input Validation | 100% | 30% | ❌ |
| Documentation | 100% | 60% | ⚠️ |
| Build Success | >95% | 70% | ❌ |
```

---

### BUILD_VERIFICATION_STATUS.md

**Improvement 1: Automated Verification Script**
```markdown
## Automated Verification

Run `scripts/verify_build.sh` to automatically verify build status:

```bash
./scripts/verify_build.sh
```

**Output:**
```
=== Build Verification Report ===
Date: 2026-01-22
Status: ✅ PASSED

Components:
  ✅ Rust library: libintent_ir.a found
  ✅ FFI header: intent_ir_ffi.h found
  ✅ JUCE FFT: Linked correctly
  ✅ KellyCore: Built successfully
  ✅ Tests: All passing

Issues: None

Last Verified: 2026-01-22 10:30:00
```

**Verification Checks:**
- [x] Rust library exists
- [x] FFI header generated
- [x] All targets build
- [x] No undefined symbols
- [x] Tests pass
```

**Improvement 2: Clear Status Indicators**
```markdown
## Component Status

### IntentIRAdapter
- **Status:** ✅ Verified
- **Last Checked:** 2026-01-22
- **Verification Method:** Automated + Manual
- **Issues:** None
- **Log:** [verification_log_2026-01-22.txt](logs/verification_log_2026-01-22.txt)

### SpectralAnalyzer
- **Status:** ✅ Verified
- **Last Checked:** 2026-01-22
- **Verification Method:** Automated
- **Issues:** None

### KellyBrain
- **Status:** ⚠️ Needs Review
- **Last Checked:** 2026-01-21
- **Verification Method:** Manual
- **Issues:**
  - [ ] Error handling incomplete
  - [ ] Test coverage low
- **Action Required:** Add error handling tests

**Status Legend:**
- ✅ Verified and passing
- ⚠️ Needs attention
- ❌ Failed verification
```

**Improvement 3: Actionable Items with Owners**
```markdown
## Action Items

| Item | Priority | Owner | Due Date | Status |
|------|----------|-------|----------|--------|
| Fix JUCE FFT API compatibility | P0 | @dev1 | 2026-01-25 | 🚧 |
| Add error handling tests | P1 | @dev2 | 2026-01-28 | 📋 |
| Update documentation | P2 | @dev3 | 2026-02-01 | 📋 |

**Priority:**
- P0: Critical, blocks release
- P1: High, should be done soon
- P2: Medium, nice to have
```

---

### QUICK_BUILD_CHECKLIST.md

**Improvement 1: Interactive Checklist with Verification**
```markdown
## Pre-Build Checklist

- [ ] Rust toolchain installed
  ```bash
  cargo --version  # Should show 1.70+
  ```

- [ ] CMake 3.27+ installed
  ```bash
  cmake --version  # Should show 3.27+
  ```

- [ ] JUCE available
  ```bash
  ls build/external/JUCE  # Should exist
  ```

- [ ] Qt6 installed (for desktop)
  ```bash
  brew list qt@6  # macOS
  # or
  dpkg -l | grep qt6  # Linux
  ```

## Build Steps

```bash
cd /Users/seanburdges/KmiDi-1/KmiDi_FINAL
mkdir -p build && cd build
cmake .. -DBUILD_KMIDI_CORE=ON -DBUILD_TESTS=ON
cmake --build . -j$(sysctl -n hw.ncpu)
```

## Verification

After build, verify:

- [ ] Rust library exists
  ```bash
  ls build/rust_target/*/release/libintent_ir.a
  # Expected: File exists
  ```

- [ ] FFI header generated
  ```bash
  ls build/include/intent_ir_ffi.h
  # Expected: File exists
  ```

- [ ] Tests pass
  ```bash
  ctest
  # Expected: All tests pass
  ```
```

**Improvement 2: Expanded Prerequisites**
```markdown
## Prerequisites

### Required

1. **Rust Toolchain**
   - Version: 1.70+
   - Install: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
   - Verify: `rustc --version`

2. **CMake**
   - Version: 3.27+
   - macOS: `brew install cmake`
   - Linux: `sudo apt-get install cmake`
   - Verify: `cmake --version`

3. **C++ Compiler**
   - GCC 9+, Clang 10+, or MSVC 2019+
   - Must support C++20
   - Verify: `g++ --version` or `clang++ --version`

4. **JUCE**
   - Location: `build/external/JUCE`
   - Install: `git clone https://github.com/juce-framework/JUCE.git build/external/JUCE`
   - Verify: `ls build/external/JUCE`

### Optional

1. **Qt6** (for desktop build)
   - macOS: `brew install qt@6`
   - Linux: `sudo apt-get install qt6-base-dev`
   - Verify: `pkg-config --modversion Qt6Core`

2. **RTNeural** (for ML inference)
   - Will be fetched automatically if ENABLE_RTNEURAL=ON

3. **ONNX Runtime** (for ONNX models)
   - Install separately if ENABLE_ONNX_RUNTIME=ON
```

**Improvement 3: Comprehensive Troubleshooting**
```markdown
## Common Issues & Solutions

### Issue: FFI Header Not Found

**Symptoms:**
```
fatal error: 'intent_ir_ffi.h' file not found
```

**Diagnosis:**
```bash
# Check if Rust built
ls build/rust_target/*/release/libintent_ir.a

# Check if header generated
ls build/include/intent_ir_ffi.h
```

**Solutions:**
1. Ensure Rust builds before C++:
   ```bash
   cd build
   cmake .. -DBUILD_KMIDI_CORE=ON
   # Rust should build automatically
   ```

2. Check CMake dependencies:
   ```bash
   # Should see: add_dependencies(intent_ir_adapter intent_ir_rust_lib)
   grep "add_dependencies" CMakeLists.txt
   ```

3. Manual build:
   ```bash
   cd engine/intent_ir
   cargo build --release
   ```

### Issue: JUCE FFT Not Found

**Symptoms:**
```
undefined reference to juce::dsp::FFT
```

**Diagnosis:**
```bash
# Check if juce_dsp is linked
grep "juce::juce_dsp" CMakeLists.txt
```

**Solutions:**
1. Verify JUCE version supports FFT API
2. Check CMakeLists.txt links `juce::juce_dsp` to `prrot_core`
3. Rebuild: `cmake --build . --clean-first`

### Issue: Build Fails with Cryptic Error

**Diagnosis:**
```bash
# Get detailed error
cmake .. --debug-output 2>&1 | tee build.log

# Check for missing dependencies
./scripts/check_dependencies.sh
```

**Solutions:**
1. Check error log: `cat build.log | grep -i error`
2. Verify all prerequisites installed
3. Try clean build: `rm -rf build && mkdir build && cd build && cmake ..`

### Issue: Tests Fail

**Symptoms:**
```
Some tests FAILED
```

**Diagnosis:**
```bash
# Run tests with verbose output
ctest --verbose

# Run specific test
./build/tests/intent_ir_integration_test
```

**Solutions:**
1. Check test output for specific failures
2. Verify test data exists
3. Check test environment setup
```

---

## Summary

This document provides **exactly 3 specific, actionable improvements** for each major component:

- **PRROTEngine:** Error handling, input validation, quality metrics
- **PhonemeSegmenter:** Multi-feature algorithm, memory safety, confidence calculation
- **PitchTracker:** Robust detection, confidence calculation, edge case handling
- **AudioValidator:** Comprehensive validation, quality metrics, error reporting
- **KellyBrain:** Type safety, input validation, error recovery
- **IntentIRAdapter:** Validation errors, type safety, error propagation
- **CMakeLists.txt:** Error messages, build validation, dependency checking
- **Documentation:** Status tracking, code examples, success criteria

Each improvement includes:
- **Before/After code examples**
- **Implementation details**
- **Error handling**
- **Validation**
- **Testing considerations**

**Next Steps:**
1. Review improvements
2. Prioritize implementation
3. Create tickets/issues
4. Assign owners
5. Begin implementation with P0 items

---

**Status:** Ready for implementation
**Last Updated:** 2026-01-22
