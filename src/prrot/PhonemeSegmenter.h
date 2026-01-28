#pragma once

/**
 * PhonemeSegmenter.h - RT-Safe Phoneme Segmentation
 * ==================================================
 *
 * PRROT/PARROT Voice-Instrument Compiler - Tier C (Embedded Core)
 *
 * RT-safe phoneme segmentation using rule-based methods and DSP.
 * All operations use pre-allocated buffers - no dynamic allocation
 * in audio callbacks.
 *
 * RT-Safe Guarantees:
 * - No dynamic memory allocation
 * - No Python dependencies
 * - No disk I/O
 * - Deterministic execution
 * - Pre-allocated buffers only
 */

#include "prrot/VoiceProfile.h"
// #include "penta/common/RTMemoryPool.h"  // TODO: Add when RTMemoryPool is available
#include <array>
#include <memory>
#include <vector>
#include <cstdint>

// Forward declare RTMemoryPool
namespace penta {
class RTMemoryPool;
}

namespace prrot {

// Maximum number of phonemes that can be segmented in a single pass
constexpr size_t kMaxPhonemesPerSegment = 256;
constexpr size_t kMaxSegmentBufferSize = 8192; // Samples
// FFT size - using local constant to avoid conflicts with SpectralAnalyzer::kFFTSize
static constexpr size_t kPhonemeFFTSize = 2048;

/**
 * PhonemeSegmenter - RT-safe phoneme segmentation engine
 *
 * Uses rule-based segmentation with DSP analysis for onset/offset detection.
 * All processing uses pre-allocated buffers from RTMemoryPool.
 */
class PhonemeSegmenter {
public:
    PhonemeSegmenter();
    ~PhonemeSegmenter();  // Must be defined in .cpp for PIMPL

    // Initialize with memory pool (must be called before use)
    // Note: memory_pool can be nullptr for now (will use stack-allocated buffers)
    void initialize(penta::RTMemoryPool* memory_pool = nullptr);

    // Segment phonemes from audio buffer
    // Input: audio samples (mono, assumed 44.1kHz)
    // Output: detected phoneme boundaries
    struct SegmentResult {
        std::vector<PhonemeType> phonemes;
        std::vector<float> boundaries_ms; // Boundary times in milliseconds

        bool valid = false;
        float confidence = 0.0f; // Overall confidence [0-1]
    };

    SegmentResult segment(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) noexcept;

    // Get phoneme classification for a segment
    PhonemeType classifyPhoneme(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) noexcept;

    // Get onset/offset timing for a segment
    struct OnsetOffset {
        size_t onset_sample = 0;
        size_t offset_sample = 0;
        float onset_time_ms = 0.0f;
        float offset_time_ms = 0.0f;
        float confidence = 0.0f;
    };

    OnsetOffset detectOnsetOffset(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) noexcept;

private:
    // Memory pool for RT-safe allocation
    penta::RTMemoryPool* memory_pool_ = nullptr;

    // JUCE FFT instance (optimized, RT-safe)
    // Using PIMPL pattern to avoid exposing JUCE headers in public API
    struct FFTImpl;
    std::unique_ptr<FFTImpl> fft_;

    // Pre-allocated analysis buffers
    std::array<float, kMaxSegmentBufferSize> analysis_buffer_;
    std::array<float, kPhonemeFFTSize> magnitude_spectrum_;
    std::array<float, kPhonemeFFTSize / 2> power_spectrum_;

    // Pre-allocated FFT working buffers
    mutable std::array<float, kPhonemeFFTSize> fft_real_;
    mutable std::array<float, kPhonemeFFTSize> fft_imag_;

    // Energy-based onset detection
    float detectEnergyOnset(const float* samples, size_t num_samples) const noexcept;
    float detectEnergyOffset(const float* samples, size_t num_samples) const noexcept;

    // Spectral-based classification helpers
    float computeSpectralCentroid(const float* magnitude, size_t num_bins) const noexcept;
    float computeSpectralRolloff(const float* magnitude, size_t num_bins, float percentile = 0.85f) const noexcept;
    float computeZeroCrossingRate(const float* samples, size_t num_samples) const noexcept;

    // Rule-based phoneme classification
    PhonemeType classifyByFeatures(
        float spectral_centroid,
        float spectral_rolloff,
        float zero_crossing_rate,
        float energy
    ) const noexcept;

    // Vowel/consonant distinction
    bool isVowelLike(float spectral_centroid, float spectral_rolloff) const noexcept;
    bool isConsonantLike(float zero_crossing_rate, float energy) const noexcept;

    // Simple FFT (using pre-allocated buffers)
    void computeFFT(const float* input, float* real, float* imag, size_t size) noexcept;
    void computeMagnitudeSpectrum(const float* real, const float* imag, float* magnitude, size_t size) noexcept;

    // Adaptive threshold calculation
    float computeAdaptiveEnergyThreshold(
        const float* samples,
        size_t num_samples
    ) const noexcept;

    // Confidence calculation
    float calculateSegmentationConfidence(
        const SegmentResult& result,
        const std::vector<float>& energies,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    // Validation
    bool validateSegmentation(
        const SegmentResult& result,
        float sample_rate_hz
    ) const noexcept;
};

} // namespace prrot
