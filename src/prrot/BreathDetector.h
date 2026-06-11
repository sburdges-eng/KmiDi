#pragma once

/**
 * BreathDetector.h - Breath and Noise Estimation
 * ===============================================
 *
 * PRROT/PARROT Voice-Instrument Compiler - Tier C (Embedded Core)
 *
 * RT-safe breath detection and noise floor estimation.
 */

#include <array>
#include <cstdint>
#include <cstddef>

namespace prrot {

constexpr size_t kMaxBreathAnalysisWindow = 4096;

/**
 * BreathDetector - RT-safe breath and noise detection
 */
class BreathDetector {
public:
    BreathDetector();
    ~BreathDetector() = default;

    // Detect breath markers in audio
    struct BreathMarker {
        size_t sample_index = 0;
        float time_ms = 0.0f;
        float intensity = 0.0f;      // Breath intensity [0-1]
        float duration_ms = 0.0f;
        float confidence = 0.0f;
    };

    BreathMarker detectBreath(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    // Estimate noise floor
    float estimateNoiseFloor(
        const float* audio_samples,
        size_t num_samples
    ) const noexcept;

    // Estimate breath weighting
    float estimateBreathWeight(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

private:
    // Pre-allocated buffers
    mutable std::array<float, kMaxBreathAnalysisWindow> analysis_buffer_;

    // Helper functions
    float computeEnergy(const float* samples, size_t num_samples) const noexcept;
    float computeHighFrequencyEnergy(const float* samples, size_t num_samples, float sample_rate_hz) const noexcept;
};

} // namespace prrot
