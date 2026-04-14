#include "prrot/AudioValidator.h"
#include "prrot/InputValidation.h"
#include "penta/common/RTLogger.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace prrot {

AudioValidator::AudioValidator() {
    // No initialization needed
}

AudioValidator::ValidationResult AudioValidator::validate(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) const  {
    ValidationResult result;

    // Comprehensive input validation
    auto input_validation = validateAudioInput(audio_samples, num_samples, sample_rate_hz);
    if (input_validation.hasErrors()) {
        result.is_valid = false;
        result.error_message = "Input validation failed: " + input_validation.errorMessage();
        return result;
    }

    // Calculate duration
    result.duration_ms = (static_cast<float>(num_samples) / sample_rate_hz) * 1000.0f;
    result.is_too_short = result.duration_ms < kMinValidDurationMs;

    // Compute signal level
    result.signal_level_db = linearToDb(computeSignalLevel(audio_samples, num_samples));
    result.is_too_quiet = result.signal_level_db < kSilenceThresholdDb;

    // Detect clipping (improved detection)
    result.is_clipped = detectClipping(audio_samples, num_samples);

    // Calculate peak level safely
    if (num_samples > 0) {
        float max_val = 0.0f;
        for (size_t i = 0; i < num_samples; ++i) {
            max_val = std::max(max_val, std::abs(audio_samples[i]));
        }
        result.peak_level = max_val;
    }

    // Check for severe clipping
    bool is_severely_clipped = result.is_clipped && result.peak_level >= 0.99f;

    // Estimate noise floor
    result.noise_floor_db = linearToDb(estimateNoiseFloor(audio_samples, num_samples));

    // Compute SNR
    result.snr_db = computeSNR(audio_samples, num_samples, dbToLinear(result.noise_floor_db));
    result.has_low_snr = result.snr_db < kMinSNRDb;

    // Detect silence (check if entirely silent)
    result.has_silence = detectSilence(audio_samples, num_samples);
    bool is_entirely_silent = result.has_silence && result.signal_level_db < kSilenceThresholdDb;

    // Check DC offset
    float dc_offset = calculateDCOffset(audio_samples, num_samples);
    bool has_dc_offset = std::abs(dc_offset) > 0.1f;

    // Determine overall validity
    result.is_valid = !result.is_too_short &&
                      !is_entirely_silent &&
                      !is_severely_clipped &&
                      result.signal_level_db > kSilenceThresholdDb;

    // Build detailed error message if invalid
    if (!result.is_valid) {
        std::vector<std::string> issues;
        if (result.is_too_short) {
            issues.push_back("Duration too short (" + std::to_string(result.duration_ms) +
                           "ms < " + std::to_string(kMinValidDurationMs) + "ms)");
        }
        if (is_entirely_silent) {
            issues.push_back("Audio is entirely silent (signal level: " +
                           std::to_string(result.signal_level_db) + "dB)");
        }
        if (is_severely_clipped) {
            issues.push_back("Audio is severely clipped (peak: " +
                           std::to_string(result.peak_level) + ")");
        }
        if (result.signal_level_db <= kSilenceThresholdDb && !is_entirely_silent) {
            issues.push_back("Signal too quiet (" + std::to_string(result.signal_level_db) +
                           "dB <= " + std::to_string(kSilenceThresholdDb) + "dB)");
        }

        if (issues.empty()) {
            result.error_message = "Audio validation failed";
        } else {
            result.error_message = issues[0];
            for (size_t i = 1; i < issues.size(); ++i) {
                result.error_message += "; " + issues[i];
            }
        }
    } else {
        // Add warnings for quality issues
        std::vector<std::string> warnings;
        if (result.is_clipped && !is_severely_clipped) {
            warnings.push_back("Some clipping detected");
        }
        if (has_dc_offset) {
            warnings.push_back("DC offset detected: " + std::to_string(dc_offset));
        }
        if (result.has_low_snr) {
            warnings.push_back("Low SNR: " + std::to_string(result.snr_db) + "dB < " +
                             std::to_string(kMinSNRDb) + "dB");
        }
        if (result.signal_level_db < -40.0f) {
            warnings.push_back("Signal level low: " + std::to_string(result.signal_level_db) + "dB");
        }

        if (!warnings.empty()) {
            result.error_message = "Warnings: " + warnings[0];
            for (size_t i = 1; i < warnings.size(); ++i) {
                result.error_message += "; " + warnings[i];
            }
        }
    }

    return result;
}

bool AudioValidator::detectSilence(
    const float* samples,
    size_t num_samples,
    float threshold_db
) const noexcept {
    if (!samples || num_samples == 0) {
        return true;
    }

    float threshold_linear = dbToLinear(threshold_db);
    float signal_level = computeSignalLevel(samples, num_samples);

    return signal_level < threshold_linear;
}

bool AudioValidator::detectClipping(
    const float* samples,
    size_t num_samples,
    float threshold
) const noexcept {
    if (!samples || num_samples == 0) {
        return false;
    }

    // Count samples at or near maximum
    size_t clipped_count = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        if (std::abs(samples[i]) >= threshold) {
            clipped_count++;
        }
    }

    // If more than 1% of samples are clipped, consider it clipped
    float clipped_ratio = static_cast<float>(clipped_count) / static_cast<float>(num_samples);
    return clipped_ratio > 0.01f;
}

float AudioValidator::estimateNoiseFloor(
    const float* samples,
    size_t num_samples
) const  {
    if (!samples || num_samples == 0) {
        return 0.0f;
    }

    // Estimate noise floor from minimum energy regions
    // Use bottom 10% of energy values
    std::vector<float> energies;
    size_t frame_size = std::max(static_cast<size_t>(num_samples / 100), static_cast<size_t>(1));

    for (size_t i = 0; i < num_samples; i += frame_size) {
        size_t end = std::min(i + frame_size, num_samples);
        float energy = 0.0f;
        for (size_t j = i; j < end; ++j) {
            energy += samples[j] * samples[j];
        }
        energy = std::sqrt(energy / static_cast<float>(end - i));
        energies.push_back(energy);
    }

    if (energies.empty()) {
        return 0.0f;
    }

    // Sort and take bottom 10%
    std::sort(energies.begin(), energies.end());
    size_t bottom_count = std::max(static_cast<size_t>(energies.size() * 0.1f), static_cast<size_t>(1));

    float noise_floor = 0.0f;
    for (size_t i = 0; i < bottom_count; ++i) {
        noise_floor += energies[i];
    }
    noise_floor /= static_cast<float>(bottom_count);

    return noise_floor;
}

float AudioValidator::computeSignalLevel(
    const float* samples,
    size_t num_samples
) const noexcept {
    if (!samples || num_samples == 0) {
        return 0.0f;
    }

    // RMS (Root Mean Square) level
    float sum_squares = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        sum_squares += samples[i] * samples[i];
    }

    return std::sqrt(sum_squares / static_cast<float>(num_samples));
}

float AudioValidator::computeSNR(
    const float* samples,
    size_t num_samples,
    float noise_floor
) const noexcept {
    if (!samples || num_samples == 0 || noise_floor <= 0.0f) {
        return 0.0f;
    }

    float signal_level = computeSignalLevel(samples, num_samples);

    if (signal_level <= noise_floor) {
        return 0.0f;
    }

    float snr_linear = signal_level / noise_floor;
    return linearToDb(snr_linear);
}

float AudioValidator::linearToDb(float linear) const noexcept {
    if (linear <= 0.0f) {
        return -80.0f; // Very quiet
    }
    return 20.0f * std::log10(linear);
}

float AudioValidator::dbToLinear(float db) const noexcept {
    return std::pow(10.0f, db / 20.0f);
}

float AudioValidator::calculateDCOffset(
    const float* samples,
    size_t num_samples
) const noexcept {
    if (!samples || num_samples == 0) {
        return 0.0f;
    }

    // DC offset is the mean value
    float sum = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        sum += samples[i];
    }
    return sum / static_cast<float>(num_samples);
}

} // namespace prrot
