#pragma once

/**
 * ProcessingError.h - Error Types for PRROT Processing
 * ====================================================
 *
 * Defines error types for PRROT engine processing operations.
 */

#include <string>
#include <cstdint>

namespace prrot {

/**
 * ProcessingError - Error codes for PRROT operations
 */
enum class ProcessingError : uint32_t {
    Success = 0,

    // Input validation errors
    InvalidInput = 1000,
    NullPointer,
    InvalidSampleCount,
    InvalidSampleRate,
    InvalidDuration,
    AudioTooShort,
    AudioTooQuiet,

    // Processing errors
    ProcessingFailed = 2000,
    PhonemeAnalysisFailed,
    PitchTrackingFailed,
    BreathDetectionFailed,
    SegmentationFailed,
    ValidationFailed,

    // Quality errors
    LowQuality = 3000,
    InsufficientConfidence,
    PoorSignalQuality,

    // Memory errors
    MemoryError = 4000,
    BufferOverflow,
    AllocationFailed,

    // Configuration errors
    ConfigurationError = 5000,
    ProfileNotLoaded,
    InvalidConfiguration,

    // Unknown error
    Unknown = 9999
};

/**
 * ErrorInfo - Detailed error information
 */
struct ErrorInfo {
    ProcessingError code;
    std::string message;
    std::string context; // Additional context (function name, etc.)

    ErrorInfo() : code(ProcessingError::Unknown) {}

    ErrorInfo(ProcessingError code, const std::string& message, const std::string& context = "")
        : code(code), message(message), context(context) {}

    std::string toString() const {
        std::string result = "Error " + std::to_string(static_cast<uint32_t>(code));
        if (!message.empty()) {
            result += ": " + message;
        }
        if (!context.empty()) {
            result += " [" + context + "]";
        }
        return result;
    }
};

// Helper functions
inline const char* errorCodeToString(ProcessingError error) {
    switch (error) {
        case ProcessingError::Success: return "Success";
        case ProcessingError::InvalidInput: return "InvalidInput";
        case ProcessingError::NullPointer: return "NullPointer";
        case ProcessingError::InvalidSampleCount: return "InvalidSampleCount";
        case ProcessingError::InvalidSampleRate: return "InvalidSampleRate";
        case ProcessingError::InvalidDuration: return "InvalidDuration";
        case ProcessingError::AudioTooShort: return "AudioTooShort";
        case ProcessingError::AudioTooQuiet: return "AudioTooQuiet";
        case ProcessingError::ProcessingFailed: return "ProcessingFailed";
        case ProcessingError::PhonemeAnalysisFailed: return "PhonemeAnalysisFailed";
        case ProcessingError::PitchTrackingFailed: return "PitchTrackingFailed";
        case ProcessingError::BreathDetectionFailed: return "BreathDetectionFailed";
        case ProcessingError::SegmentationFailed: return "SegmentationFailed";
        case ProcessingError::ValidationFailed: return "ValidationFailed";
        case ProcessingError::LowQuality: return "LowQuality";
        case ProcessingError::InsufficientConfidence: return "InsufficientConfidence";
        case ProcessingError::PoorSignalQuality: return "PoorSignalQuality";
        case ProcessingError::MemoryError: return "MemoryError";
        case ProcessingError::BufferOverflow: return "BufferOverflow";
        case ProcessingError::AllocationFailed: return "AllocationFailed";
        case ProcessingError::ConfigurationError: return "ConfigurationError";
        case ProcessingError::ProfileNotLoaded: return "ProfileNotLoaded";
        case ProcessingError::InvalidConfiguration: return "InvalidConfiguration";
        default: return "Unknown";
    }
}

} // namespace prrot
