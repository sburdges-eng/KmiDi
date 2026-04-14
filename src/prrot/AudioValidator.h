#pragma once

/**
 * AudioValidator.h - Audio Input Validation
 * ==========================================
 *
 * PRROT/PARROT Voice-Instrument Compiler - Tier C (Embedded Core)
 *
 * RT-safe audio validation and quality assessment.
 * Detects silence, clipping, noise, and other quality issues.
 */

#include <cstdint>
#include <string>

namespace prrot {

// Validation constants
constexpr float kMinValidDurationMs = 50.0f;      // Minimum 50ms for processing
constexpr float kSilenceThresholdDb = -60.0f;     // Below -60dB considered silence
constexpr float kClippingThreshold = 0.99f;       // Above 99% of max considered clipping
constexpr float kMinSNRDb = 10.0f;                // Minimum 10dB SNR for good quality

/**
 * AudioValidator - RT-safe audio validation
 */
class AudioValidator {
public:
    AudioValidator();
    ~AudioValidator() = default;

    /**
     * Validation result
     */
    struct ValidationResult {
        bool is_valid = false;              // Overall validity
        bool has_silence = false;            // Contains silence
        bool is_clipped = false;             // Contains clipping
        bool is_too_short = false;           // Duration too short
        bool is_too_quiet = false;           // Signal too quiet
        bool has_low_snr = false;            // Low signal-to-noise ratio

        float noise_floor_db = -80.0f;       // Estimated noise floor (dB)
        float signal_level_db = -80.0f;      // Signal level (dB)
        float snr_db = 0.0f;                 // Signal-to-noise ratio (dB)
        float duration_ms = 0.0f;            // Duration in milliseconds
        float peak_level = 0.0f;             // Peak level [0.0, 1.0]

        std::string error_message;           // Error description if invalid

        // Quality score [0.0, 1.0] - improved calculation
        float quality_score() const noexcept {
            if (!is_valid) return 0.0f;

            // Factor 1: Signal quality (level, clipping)
            float signal_score = 1.0f;
            if (is_clipped) {
                signal_score *= 0.5f; // Severe penalty for clipping
            }
            if (signal_level_db < -40.0f) {
                signal_score *= 0.7f; // Penalty for low signal
            } else if (signal_level_db > -3.0f && !is_clipped) {
                signal_score *= 0.9f; // Slight penalty for near-clipping
            }

            // Factor 2: Noise quality (SNR)
            float noise_score = 1.0f;
            if (has_low_snr) {
                noise_score = std::clamp(snr_db / 30.0f, 0.0f, 1.0f);
            } else if (snr_db > 20.0f) {
                noise_score = 1.1f; // Boost for good SNR
            }

            // Factor 3: Dynamic range
            float dynamic_range = signal_level_db - noise_floor_db;
            float range_score = std::clamp(dynamic_range / 40.0f, 0.0f, 1.0f);

            // Factor 4: Silence (partial silence is OK, but reduces score)
            float silence_score = has_silence ? 0.8f : 1.0f;

            // Weighted combination
            float score = (
                signal_score * 0.4f +
                noise_score * 0.3f +
                range_score * 0.2f +
                silence_score * 0.1f
            );

            return std::clamp(score, 0.0f, 1.0f);
        }
    };

    /**
     * Validate audio input
     * RT-Safe: Uses pre-allocated buffers only
     *
     * @param audio_samples Input audio samples
     * @param num_samples Number of samples
     * @param sample_rate_hz Sample rate in Hz
     * @return Validation result
     */
    ValidationResult validate(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const ;

private:
    /**
     * Detect silence in audio
     */
    bool detectSilence(
        const float* samples,
        size_t num_samples,
        float threshold_db = kSilenceThresholdDb
    ) const noexcept;

    /**
     * Detect clipping in audio
     */
    bool detectClipping(
        const float* samples,
        size_t num_samples,
        float threshold = kClippingThreshold
    ) const noexcept;

    /**
     * Estimate noise floor
     */
    float estimateNoiseFloor(
        const float* samples,
        size_t num_samples
    ) const ;

    /**
     * Compute signal level
     */
    float computeSignalLevel(
        const float* samples,
        size_t num_samples
    ) const noexcept;

    /**
     * Compute signal-to-noise ratio
     */
    float computeSNR(
        const float* samples,
        size_t num_samples,
        float noise_floor
    ) const noexcept;

    /**
     * Convert linear amplitude to dB
     */
    float linearToDb(float linear) const noexcept;

    /**
     * Convert dB to linear amplitude
     */
    float dbToLinear(float db) const noexcept;

    /**
     * Calculate DC offset
     */
    float calculateDCOffset(
        const float* samples,
        size_t num_samples
    ) const noexcept;
};

} // namespace prrot
