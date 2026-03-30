#pragma once

/**
 * InputValidation.h - Input Validation Framework
 * ===============================================
 *
 * Provides comprehensive input validation for PRROT operations.
 */

#include "prrot/ProcessingError.h"
#include <string>
#include <vector>
#include <cstddef>

namespace prrot {

/**
 * ValidationIssue - Individual validation issue
 */
struct ValidationIssue {
    ProcessingError error_code;
    std::string message;
    bool is_error; // true for error, false for warning

    ValidationIssue(ProcessingError code, const std::string& msg, bool is_err = true)
        : error_code(code), message(msg), is_error(is_err) {}
};

/**
 * InputValidation - Validation result
 */
class InputValidation {
public:
    InputValidation() = default;

    void addError(ProcessingError code, const std::string& message) {
        issues_.emplace_back(code, message, true);
    }

    void addWarning(ProcessingError code, const std::string& message) {
        issues_.emplace_back(code, message, false);
    }

    bool isValid() const {
        for (const auto& issue : issues_) {
            if (issue.is_error) {
                return false;
            }
        }
        return true;
    }

    bool hasErrors() const {
        for (const auto& issue : issues_) {
            if (issue.is_error) {
                return true;
            }
        }
        return false;
    }

    bool hasWarnings() const {
        for (const auto& issue : issues_) {
            if (!issue.is_error) {
                return true;
            }
        }
        return false;
    }

    const std::vector<ValidationIssue>& issues() const {
        return issues_;
    }

    std::string errorMessage() const {
        std::string msg;
        for (const auto& issue : issues_) {
            if (issue.is_error) {
                if (!msg.empty()) msg += "; ";
                msg += issue.message;
            }
        }
        return msg;
    }

    std::string warningMessage() const {
        std::string msg;
        for (const auto& issue : issues_) {
            if (!issue.is_error) {
                if (!msg.empty()) msg += "; ";
                msg += issue.message;
            }
        }
        return msg;
    }

    std::string fullReport() const {
        std::string report;
        if (hasErrors()) {
            report += "Errors: " + errorMessage() + "\n";
        }
        if (hasWarnings()) {
            report += "Warnings: " + warningMessage();
        }
        return report;
    }

private:
    std::vector<ValidationIssue> issues_;
};

/**
 * Validate audio input parameters
 */
InputValidation validateAudioInput(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) noexcept;

/**
 * Validate sample count
 */
InputValidation validateSampleCount(size_t num_samples) noexcept;

/**
 * Validate sample rate
 */
InputValidation validateSampleRate(float sample_rate_hz) noexcept;

/**
 * Validate pointer (not null)
 */
InputValidation validatePointer(const void* ptr, const char* name) noexcept;

} // namespace prrot
