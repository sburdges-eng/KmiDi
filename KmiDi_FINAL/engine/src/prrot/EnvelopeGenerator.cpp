#include "prrot/EnvelopeGenerator.h"
#include <algorithm>
#include <cmath>

namespace prrot {

EnvelopeGenerator::EnvelopeGenerator() {
    working_buffer_.fill(0.0f);
}

void EnvelopeGenerator::generateArticulationEnvelope(
    const ArticulationEnvelope& params,
    float* output_samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    if (!output_samples || num_samples == 0) {
        return;
    }

    size_t attack_samples = static_cast<size_t>((params.attack_time_ms / 1000.0f) * sample_rate_hz);
    size_t sustain_samples = static_cast<size_t>((params.sustain_time_ms / 1000.0f) * sample_rate_hz);
    size_t release_samples = static_cast<size_t>((params.release_time_ms / 1000.0f) * sample_rate_hz);

    size_t total_envelope_samples = attack_samples + sustain_samples + release_samples;
    size_t samples_to_generate = std::min(num_samples, total_envelope_samples);

    size_t sample_idx = 0;

    // Attack phase
    size_t attack_end = std::min(attack_samples, samples_to_generate);
    for (size_t i = 0; i < attack_end; ++i, ++sample_idx) {
        float t = static_cast<float>(i) / static_cast<float>(attack_samples);
        output_samples[sample_idx] = interpolateValue(t, 0.0f, params.sustain_level, params.attack_curve);
    }

    // Sustain phase
    if (sample_idx < samples_to_generate) {
        size_t sustain_end = std::min(sample_idx + sustain_samples, samples_to_generate);
        for (; sample_idx < sustain_end; ++sample_idx) {
            output_samples[sample_idx] = params.sustain_level;
        }
    }

    // Release phase
    if (sample_idx < samples_to_generate && release_samples > 0) {
        for (size_t i = 0; sample_idx < samples_to_generate; ++i, ++sample_idx) {
            float t = static_cast<float>(i) / static_cast<float>(release_samples);
            output_samples[sample_idx] = interpolateValue(t, params.sustain_level, 0.0f, params.release_curve);
        }
    }

    // Zero-pad remaining samples
    for (; sample_idx < num_samples; ++sample_idx) {
        output_samples[sample_idx] = 0.0f;
    }
}

void EnvelopeGenerator::generateADSR(
    float attack_time_ms,
    float decay_time_ms,
    float sustain_level,
    float release_time_ms,
    float* output_samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    if (!output_samples || num_samples == 0) {
        return;
    }

    size_t attack_samples = static_cast<size_t>((attack_time_ms / 1000.0f) * sample_rate_hz);
    size_t decay_samples = static_cast<size_t>((decay_time_ms / 1000.0f) * sample_rate_hz);
    size_t release_samples = static_cast<size_t>((release_time_ms / 1000.0f) * sample_rate_hz);

    size_t sustain_samples = num_samples - attack_samples - decay_samples - release_samples;
    if (sustain_samples > num_samples) {
        sustain_samples = 0; // Not enough samples for full envelope
    }

    size_t sample_idx = 0;

    // Attack phase
    for (size_t i = 0; i < attack_samples && sample_idx < num_samples; ++i, ++sample_idx) {
        float t = static_cast<float>(i) / static_cast<float>(attack_samples);
        output_samples[sample_idx] = t; // Linear attack
    }

    // Decay phase
    for (size_t i = 0; i < decay_samples && sample_idx < num_samples; ++i, ++sample_idx) {
        float t = static_cast<float>(i) / static_cast<float>(decay_samples);
        output_samples[sample_idx] = 1.0f - t * (1.0f - sustain_level);
    }

    // Sustain phase
    for (size_t i = 0; i < sustain_samples && sample_idx < num_samples; ++i, ++sample_idx) {
        output_samples[sample_idx] = sustain_level;
    }

    // Release phase
    for (size_t i = 0; i < release_samples && sample_idx < num_samples; ++i, ++sample_idx) {
        float t = static_cast<float>(i) / static_cast<float>(release_samples);
        output_samples[sample_idx] = sustain_level * (1.0f - t);
    }

    // Zero-pad remaining
    for (; sample_idx < num_samples; ++sample_idx) {
        output_samples[sample_idx] = 0.0f;
    }
}

void EnvelopeGenerator::generateExponential(
    float attack_time_ms,
    float release_time_ms,
    float curve_exponent,
    float* output_samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    if (!output_samples || num_samples == 0) {
        return;
    }

    size_t attack_samples = static_cast<size_t>((attack_time_ms / 1000.0f) * sample_rate_hz);
    size_t release_samples = static_cast<size_t>((release_time_ms / 1000.0f) * sample_rate_hz);
    size_t sustain_samples = (num_samples > attack_samples + release_samples) ?
                            (num_samples - attack_samples - release_samples) : 0;

    size_t sample_idx = 0;

    // Attack phase
    for (size_t i = 0; i < attack_samples && sample_idx < num_samples; ++i, ++sample_idx) {
        float t = static_cast<float>(i) / static_cast<float>(attack_samples);
        output_samples[sample_idx] = std::pow(t, curve_exponent);
    }

    // Sustain phase
    for (size_t i = 0; i < sustain_samples && sample_idx < num_samples; ++i, ++sample_idx) {
        output_samples[sample_idx] = 1.0f;
    }

    // Release phase
    for (size_t i = 0; i < release_samples && sample_idx < num_samples; ++i, ++sample_idx) {
        float t = static_cast<float>(i) / static_cast<float>(release_samples);
        output_samples[sample_idx] = std::pow(1.0f - t, curve_exponent);
    }

    // Zero-pad remaining
    for (; sample_idx < num_samples; ++sample_idx) {
        output_samples[sample_idx] = 0.0f;
    }
}

void EnvelopeGenerator::generateLinear(
    float attack_time_ms,
    float release_time_ms,
    float* output_samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    generateExponential(attack_time_ms, release_time_ms, 1.0f, output_samples, num_samples, sample_rate_hz);
}

float EnvelopeGenerator::interpolateValue(float t, float start, float end, float curve) const noexcept {
    t = std::clamp(t, 0.0f, 1.0f);

    // Apply curve: 0 = linear, 1 = exponential
    float curved_t = (curve > 0.5f) ? std::pow(t, 2.0f - curve * 2.0f) :
                                      (t * (1.0f + curve * 2.0f));

    return start + curved_t * (end - start);
}

} // namespace prrot
