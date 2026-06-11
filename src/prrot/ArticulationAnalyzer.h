#pragma once

/**
 * ArticulationAnalyzer.h - Vowel/Consonant Classification and Onset/Offset Timing
 * ===============================================================================
 *
 * PRROT/PARROT Voice-Instrument Compiler - Tier C (Embedded Core)
 *
 * RT-safe articulation analysis for vowel/consonant classification and
 * onset/offset timing detection. Uses pre-allocated buffers only.
 */

#include "prrot/VoiceProfile.h"
#include "prrot/PhonemeSegmenter.h"
#include <array>
#include <cstdint>
#include <cstddef>

namespace prrot {

constexpr size_t kMaxAnalysisWindow = 4096; // Samples

/**
 * ArticulationAnalyzer - RT-safe articulation analysis
 */
class ArticulationAnalyzer {
public:
    ArticulationAnalyzer();
    ~ArticulationAnalyzer() = default;

    // Classify phoneme as vowel or consonant
    enum class ArticulationClass {
        Vowel,
        Consonant,
        Unknown
    };

    ArticulationClass classifyArticulation(PhonemeType phoneme) const noexcept;

    // Detect onset timing in audio segment
    struct OnsetInfo {
        size_t sample_index = 0;
        float time_ms = 0.0f;
        float sharpness = 0.0f; // Onset sharpness [0-1]
        float confidence = 0.0f;
    };

    OnsetInfo detectOnset(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    // Detect offset timing in audio segment
    struct OffsetInfo {
        size_t sample_index = 0;
        float time_ms = 0.0f;
        float smoothness = 0.0f; // Offset smoothness [0-1]
        float confidence = 0.0f;
    };

    OffsetInfo detectOffset(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    // Get articulation envelope (attack/sustain/release shape)
    struct ArticulationShape {
        float attack_time_ms = 0.0f;
        float sustain_level = 1.0f;
        float release_time_ms = 0.0f;
        float attack_curve = 0.5f;   // 0=linear, 1=exponential
        float release_curve = 0.5f;
    };

    ArticulationShape analyzeArticulationShape(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz,
        PhonemeType phoneme
    ) const noexcept;

private:
    // Pre-allocated analysis buffers
    mutable std::array<float, kMaxAnalysisWindow> analysis_buffer_;
    mutable std::array<float, kMaxAnalysisWindow> energy_buffer_;
    mutable std::array<float, kMaxAnalysisWindow> derivative_buffer_;

    // Helper functions
    float computeEnergy(const float* samples, size_t num_samples) const noexcept;
    void computeDerivative(const float* samples, size_t num_samples, float* derivative) const noexcept;
    float findPeak(const float* samples, size_t num_samples) const noexcept;
};

} // namespace prrot
