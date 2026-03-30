#include "prrot/InputValidation.h"
#include "penta/common/RTLogger.h"
#include <algorithm>
#include <cmath>

namespace prrot {

// Constants
constexpr size_t kMinSamplesPerSegment = 100; // Minimum samples for processing
constexpr size_t kMaxSamplesPerSegment = 1024 * 1024; // 1M samples max
constexpr float kMinSampleRateHz = 8000.0f;
constexpr float kMaxSampleRateHz = 192000.0f;
constexpr float kMinValidDurationMs = 10.0f; // 10ms minimum

InputValidation validatePointer(const void* ptr, const char* name) noexcept {
    InputValidation validation;

    if (!ptr) {
        validation.addError(
            ProcessingError::NullPointer,
            std::string(name) + " is null"
        );
    }

    return validation;
}

InputValidation validateSampleCount(size_t num_samples) noexcept {
    InputValidation validation;

    if (num_samples == 0) {
        validation.addError(
            ProcessingError::InvalidSampleCount,
            "num_samples is zero"
        );
        return validation;
    }

    if (num_samples < kMinSamplesPerSegment) {
        validation.addWarning(
            ProcessingError::AudioTooShort,
            "num_samples (" + std::to_string(num_samples) +
            ") below recommended minimum (" + std::to_string(kMinSamplesPerSegment) + ")"
        );
    }

    if (num_samples > kMaxSamplesPerSegment) {
        validation.addError(
            ProcessingError::InvalidSampleCount,
            "num_samples (" + std::to_string(num_samples) +
            ") exceeds maximum (" + std::to_string(kMaxSamplesPerSegment) + ")"
        );
    }

    return validation;
}

InputValidation validateSampleRate(float sample_rate_hz) noexcept {
    InputValidation validation;

    if (sample_rate_hz <= 0.0f || std::isnan(sample_rate_hz) || std::isinf(sample_rate_hz)) {
        validation.addError(
            ProcessingError::InvalidSampleRate,
            "sample_rate_hz is invalid: " + std::to_string(sample_rate_hz)
        );
        return validation;
    }

    if (sample_rate_hz < kMinSampleRateHz) {
        validation.addWarning(
            ProcessingError::InvalidSampleRate,
            "sample_rate_hz (" + std::to_string(sample_rate_hz) +
            "Hz) below typical minimum (" + std::to_string(kMinSampleRateHz) + "Hz)"
        );
    }

    if (sample_rate_hz > kMaxSampleRateHz) {
        validation.addWarning(
            ProcessingError::InvalidSampleRate,
            "sample_rate_hz (" + std::to_string(sample_rate_hz) +
            "Hz) above typical maximum (" + std::to_string(kMaxSampleRateHz) + "Hz)"
        );
    }

    return validation;
}

InputValidation validateAudioInput(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) noexcept {
    InputValidation validation;

    // Validate pointer
    auto ptr_validation = validatePointer(audio_samples, "audio_samples");
    if (ptr_validation.hasErrors()) {
        validation.addError(
            ProcessingError::NullPointer,
            ptr_validation.errorMessage()
        );
        return validation; // Can't continue without valid pointer
    }

    // Validate sample count
    auto count_validation = validateSampleCount(num_samples);
    for (const auto& issue : count_validation.issues()) {
        if (issue.is_error) {
            validation.addError(issue.error_code, issue.message);
        } else {
            validation.addWarning(issue.error_code, issue.message);
        }
    }

    // Validate sample rate
    auto rate_validation = validateSampleRate(sample_rate_hz);
    for (const auto& issue : rate_validation.issues()) {
        if (issue.is_error) {
            validation.addError(issue.error_code, issue.message);
        } else {
            validation.addWarning(issue.error_code, issue.message);
        }
    }

    // Validate duration
    if (num_samples > 0 && sample_rate_hz > 0.0f) {
        float duration_ms = (static_cast<float>(num_samples) / sample_rate_hz) * 1000.0f;
        if (duration_ms < kMinValidDurationMs) {
            validation.addError(
                ProcessingError::AudioTooShort,
                "Duration too short: " + std::to_string(duration_ms) +
                "ms < " + std::to_string(kMinValidDurationMs) + "ms"
            );
        }
    }

    return validation;
}

} // namespace prrot
