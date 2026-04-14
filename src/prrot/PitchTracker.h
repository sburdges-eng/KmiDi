#pragma once

/**
 * PitchTracker.h - Real-Time Pitch Detection
 * ==========================================
 *
 * PRROT/PARROT Voice-Instrument Compiler - Tier C (Embedded Core)
 *
 * RT-safe pitch tracking using autocorrelation and FFT-based methods.
 * Provides robust fundamental frequency (F0) detection with confidence scoring.
 */

#include <array>
#include <cstdint>
#include <cmath>
#include <vector>

// Forward declaration
namespace prrot {
    struct PitchTarget;
}

namespace prrot {

// Pitch detection constants
constexpr size_t kMaxPitchAnalysisWindow = 4096;
constexpr float kMinPitchHz = 50.0f;   // ~MIDI 24 (very low)
constexpr float kMaxPitchHz = 2000.0f; // ~MIDI 108 (very high)
constexpr float kDefaultPitchHz = 440.0f; // A4 (MIDI 69)

/**
 * PitchTracker - RT-safe pitch detection
 *
 * Uses autocorrelation for robust F0 detection with fallback to FFT-based methods.
 * Provides confidence scoring based on signal quality.
 */
class PitchTracker {
public:
    PitchTracker();
    ~PitchTracker() = default;

    /**
     * Pitch detection result
     */
    struct PitchResult {
        float frequency_hz = 0.0f;      // Detected fundamental frequency
        int midi_note = 60;             // MIDI note number (0-127)
        float cents_offset = 0.0f;      // Fine pitch offset in cents (-50 to +50)
        float confidence = 0.0f;        // Confidence [0.0, 1.0]
        bool is_valid = false;           // Whether result is valid
    };

    /**
     * Track pitch in audio segment
     * RT-Safe: Uses pre-allocated buffers only
     *
     * @param audio_samples Input audio samples
     * @param num_samples Number of samples
     * @param sample_rate_hz Sample rate in Hz
     * @return Pitch detection result
     */
    PitchResult trackPitch(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    /**
     * Track pitch for multiple segments (for time-varying pitch)
     * Returns pitch targets at regular intervals
     *
     * @param audio_samples Input audio samples
     * @param num_samples Number of samples
     * @param sample_rate_hz Sample rate in Hz
     * @param interval_ms Interval between pitch targets (milliseconds)
     * @return Vector of pitch targets
     */
    std::vector<PitchTarget> trackPitchSequence(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz,
        float interval_ms = 50.0f
    ) const ;

private:
    // Pre-allocated buffers for RT-safety
    mutable std::array<float, kMaxPitchAnalysisWindow> autocorr_buffer_;
    mutable std::array<float, kMaxPitchAnalysisWindow> windowed_buffer_;

    /**
     * Autocorrelation-based pitch detection
     * Most robust method for clean signals
     */
    float autocorrelationPitch(
        const float* samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    /**
     * FFT-based pitch detection (fallback)
     * Better for noisy signals
     */
    float fftPitch(
        const float* samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    /**
     * Convert frequency to MIDI note
     */
    int frequencyToMidi(float frequency_hz) const noexcept;

    /**
     * Convert MIDI note to frequency
     */
    float midiToFrequency(int midi_note) const noexcept;

    /**
     * Calculate cents offset from frequency
     */
    float frequencyToCents(float frequency_hz, int midi_note) const noexcept;

    /**
     * Calculate confidence based on signal quality
     */
    float calculateConfidence(
        const float* samples,
        size_t num_samples,
        float detected_frequency,
        float sample_rate_hz
    ) const noexcept;

    /**
     * Apply window function (Hann window)
     */
    void applyWindow(
        const float* input,
        float* output,
        size_t num_samples
    ) const noexcept;

    /**
     * Normalize autocorrelation result
     */
    void normalizeAutocorrelation(
        float* autocorr,
        size_t num_samples
    ) const noexcept;

    /**
     * Validate pitch result
     */
    bool validatePitchResult(
        const PitchResult& result,
        float sample_rate_hz
    ) const noexcept;
};

} // namespace prrot
