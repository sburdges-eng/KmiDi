#pragma once

/**
 * EnvelopeGenerator.h - Articulation Envelope Generators
 * ======================================================
 *
 * PRROT/PARROT Voice-Instrument Compiler - Tier C (Embedded Core)
 *
 * RT-safe envelope generation for articulation control using
 * pre-allocated memory pools.
 */

#include "prrot/VoiceProfile.h"
#include "prrot/PhonemeControlData.h"
#include <array>
#include <cstdint>

namespace prrot {

constexpr size_t kMaxEnvelopePoints = 1024;

/**
 * EnvelopeGenerator - RT-safe envelope generation
 */
class EnvelopeGenerator {
public:
    EnvelopeGenerator();
    ~EnvelopeGenerator() = default;

    // Generate articulation envelope for a phoneme
    void generateArticulationEnvelope(
        const ArticulationEnvelope& params,
        float* output_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    // Generate ADSR envelope (Attack-Decay-Sustain-Release)
    void generateADSR(
        float attack_time_ms,
        float decay_time_ms,
        float sustain_level,
        float release_time_ms,
        float* output_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    // Generate exponential envelope
    void generateExponential(
        float attack_time_ms,
        float release_time_ms,
        float curve_exponent,
        float* output_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    // Generate linear envelope
    void generateLinear(
        float attack_time_ms,
        float release_time_ms,
        float* output_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

private:
    // Pre-allocated working buffer
    mutable std::array<float, kMaxEnvelopePoints> working_buffer_;

    // Helper functions
    float interpolateValue(float t, float start, float end, float curve) const noexcept;
};

} // namespace prrot
