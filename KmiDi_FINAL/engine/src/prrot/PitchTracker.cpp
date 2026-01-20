#include "prrot/PitchTracker.h"
#include "prrot/PhonemeControlData.h"  // For PitchTarget
#include "prrot/InputValidation.h"
#include "prrot/SpectralAnalyzer.h"
#include "penta/common/RTLogger.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <limits>
#include <array>

namespace prrot {

PitchTracker::PitchTracker() {
    autocorr_buffer_.fill(0.0f);
    windowed_buffer_.fill(0.0f);
}

PitchTracker::PitchResult PitchTracker::trackPitch(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    PitchResult result;
    result.is_valid = false;
    result.confidence = 0.0f;

    // Validate inputs
    auto validation = validateAudioInput(audio_samples, num_samples, sample_rate_hz);
    if (validation.hasErrors()) {
        penta::getLogger().logRT(penta::LogLevel::Error,
            ("PitchTracker::trackPitch: Input validation failed: " +
             validation.errorMessage()).c_str());
        return result;
    }

    // Check minimum samples for pitch detection (at least 2 periods of lowest pitch)
    size_t min_samples = static_cast<size_t>((2.0f / kMinPitchHz) * sample_rate_hz);
    if (num_samples < min_samples) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            ("PitchTracker::trackPitch: Too few samples (" +
             std::to_string(num_samples) + " < " + std::to_string(min_samples) + ")").c_str());
        return result;
    }

    // Clamp to max buffer size
    size_t samples_to_process = std::min(num_samples, kMaxPitchAnalysisWindow);

    // Check for silence
    float rms = 0.0f;
    for (size_t i = 0; i < samples_to_process; ++i) {
        rms += audio_samples[i] * audio_samples[i];
    }
    rms = std::sqrt(rms / samples_to_process);
    if (rms < 0.001f) {
        penta::getLogger().logRT(penta::LogLevel::Debug,
            "PitchTracker::trackPitch: Silence detected");
        return result;
    }

    // Try autocorrelation first (most robust for clean signals)
    float pitch_hz = autocorrelationPitch(audio_samples, samples_to_process, sample_rate_hz);

    // If autocorrelation fails or low confidence, try FFT method
    if (pitch_hz <= 0.0f || pitch_hz < kMinPitchHz || pitch_hz > kMaxPitchHz) {
        pitch_hz = fftPitch(audio_samples, samples_to_process, sample_rate_hz);
    }

    // Validate result
    if (pitch_hz > 0.0f && pitch_hz >= kMinPitchHz && pitch_hz <= kMaxPitchHz) {
        result.frequency_hz = pitch_hz;
        result.midi_note = frequencyToMidi(pitch_hz);
        result.cents_offset = frequencyToCents(pitch_hz, result.midi_note);
        result.confidence = calculateConfidence(audio_samples, samples_to_process, pitch_hz, sample_rate_hz);
        result.is_valid = result.confidence > 0.3f; // Minimum confidence threshold

        // Validate pitch result
        if (!validatePitchResult(result, sample_rate_hz)) {
            penta::getLogger().logRT(penta::LogLevel::Warning,
                "PitchTracker::trackPitch: Pitch result validation failed");
            result.is_valid = false;
            result.confidence = 0.0f;
        }
    } else {
        penta::getLogger().logRT(penta::LogLevel::Debug,
            ("PitchTracker::trackPitch: Detected pitch out of range: " +
             std::to_string(pitch_hz) + " Hz").c_str());
    }

    return result;
}

std::vector<PitchTarget> PitchTracker::trackPitchSequence(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz,
    float interval_ms
) const noexcept {
    std::vector<PitchTarget> targets;

    if (!audio_samples || num_samples == 0 || sample_rate_hz <= 0.0f || interval_ms <= 0.0f) {
        return targets;
    }

    size_t interval_samples = static_cast<size_t>((interval_ms / 1000.0f) * sample_rate_hz);
    interval_samples = std::max(interval_samples, static_cast<size_t>(sample_rate_hz * 0.01f)); // Minimum 10ms

    for (size_t offset = 0; offset < num_samples; offset += interval_samples) {
        size_t segment_samples = std::min(interval_samples, num_samples - offset);

        // Need minimum samples for pitch detection (at least 2 periods of lowest pitch)
        size_t min_samples = static_cast<size_t>((2.0f / kMinPitchHz) * sample_rate_hz);
        if (segment_samples < min_samples) {
            segment_samples = std::min(min_samples, num_samples - offset);
        }

        if (segment_samples < min_samples) {
            break; // Not enough samples remaining
        }

        auto pitch_result = trackPitch(audio_samples + offset, segment_samples, sample_rate_hz);

        if (pitch_result.is_valid) {
            PitchTarget target;
            target.time_ms = (static_cast<float>(offset) / sample_rate_hz) * 1000.0f;
            target.midi_note = pitch_result.midi_note;
            target.cents_offset = pitch_result.cents_offset;
            target.confidence = pitch_result.confidence;
            targets.push_back(target);
        }
    }

    return targets;
}

float PitchTracker::autocorrelationPitch(
    const float* samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    if (!samples || num_samples < 2) {
        return 0.0f;
    }

    // Apply window to reduce edge effects
    applyWindow(samples, windowed_buffer_.data(), num_samples);

    // Autocorrelation: find period by finding peak in autocorrelation
    size_t max_lag = std::min(num_samples / 2, kMaxPitchAnalysisWindow / 2);

    // Minimum lag corresponds to maximum pitch
    size_t min_lag = static_cast<size_t>(sample_rate_hz / kMaxPitchHz);
    // Maximum lag corresponds to minimum pitch
    size_t max_lag_pitch = static_cast<size_t>(sample_rate_hz / kMinPitchHz);
    max_lag = std::min(max_lag, max_lag_pitch);

    if (min_lag >= max_lag || max_lag >= num_samples) {
        return 0.0f;
    }

    float max_corr = 0.0f;
    size_t best_lag = 0;

    // Compute autocorrelation for lags in pitch range
    for (size_t lag = min_lag; lag < max_lag; ++lag) {
        float corr = 0.0f;
        size_t valid_samples = num_samples - lag;

        for (size_t i = 0; i < valid_samples; ++i) {
            corr += windowed_buffer_[i] * windowed_buffer_[i + lag];
        }

        // Normalize by number of samples
        if (valid_samples > 0) {
            corr /= static_cast<float>(valid_samples);
        }

        // Look for peak (strongest correlation)
        if (corr > max_corr) {
            max_corr = corr;
            best_lag = lag;
        }
    }

    // Convert lag to frequency
    if (best_lag > 0 && max_corr > 0.1f) { // Minimum correlation threshold
        float period_samples = static_cast<float>(best_lag);
        float frequency_hz = sample_rate_hz / period_samples;
        return frequency_hz;
    }

    return 0.0f;
}

float PitchTracker::fftPitch(
    const float* samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    if (!samples || num_samples < 2) {
        return 0.0f;
    }

    // FFT-based pitch detection using cepstral analysis
    // This method is better for noisy signals than autocorrelation

    // Use SpectralAnalyzer for FFT computation
    SpectralAnalyzer analyzer;

    // Compute magnitude spectrum
    std::array<float, 1024> magnitude_spectrum;
    analyzer.computeMagnitudeSpectrum(samples, num_samples, magnitude_spectrum.data(), 1024);

    // Find fundamental frequency from magnitude spectrum
    // Look for peaks in the expected pitch range
    float bin_frequency = sample_rate_hz / (2.0f * 1024.0f);
    size_t min_bin = static_cast<size_t>(kMinPitchHz / bin_frequency);
    size_t max_bin = static_cast<size_t>(std::min(kMaxPitchHz / bin_frequency, 1023.0f));

    // Find peak in pitch range
    float max_magnitude = 0.0f;
    size_t peak_bin = min_bin;

    for (size_t i = min_bin + 1; i < max_bin && i < 1023; ++i) {
        // Check for local maximum
        if (magnitude_spectrum[i] > magnitude_spectrum[i - 1] &&
            magnitude_spectrum[i] > magnitude_spectrum[i + 1] &&
            magnitude_spectrum[i] > max_magnitude) {
            max_magnitude = magnitude_spectrum[i];
            peak_bin = i;
        }
    }

    if (max_magnitude > 0.01f) {  // Minimum threshold
        // Parabolic interpolation for sub-bin accuracy
        float y1 = magnitude_spectrum[peak_bin - 1];
        float y2 = magnitude_spectrum[peak_bin];
        float y3 = magnitude_spectrum[peak_bin + 1];

        float denom = 2.0f * (y1 - 2.0f * y2 + y3);
        float offset = 0.0f;
        if (std::abs(denom) > 1e-6f) {
            offset = (y1 - y3) / denom;
        }

        float interpolated_bin = static_cast<float>(peak_bin) + offset;
        float frequency = interpolated_bin * bin_frequency;

        // Validate and clamp
        if (frequency >= kMinPitchHz && frequency <= kMaxPitchHz) {
            return frequency;
        }
    }

    // Fallback to autocorrelation if FFT method fails
    return autocorrelationPitch(samples, num_samples, sample_rate_hz);
}

int PitchTracker::frequencyToMidi(float frequency_hz) const noexcept {
    if (frequency_hz <= 0.0f) {
        return 60; // Default to middle C
    }

    // MIDI note = 12 * log2(frequency / 440.0) + 69
    // A4 (440 Hz) = MIDI 69
    float midi_float = 12.0f * std::log2(frequency_hz / 440.0f) + 69.0f;
    int midi_note = static_cast<int>(std::round(midi_float));

    // Clamp to valid MIDI range
    return std::clamp(midi_note, 0, 127);
}

float PitchTracker::midiToFrequency(int midi_note) const noexcept {
    // Clamp to valid MIDI range
    midi_note = std::clamp(midi_note, 0, 127);

    // Frequency = 440.0 * 2^((midi_note - 69) / 12)
    return 440.0f * std::pow(2.0f, (static_cast<float>(midi_note) - 69.0f) / 12.0f);
}

float PitchTracker::frequencyToCents(float frequency_hz, int midi_note) const noexcept {
    if (frequency_hz <= 0.0f) {
        return 0.0f;
    }

    float expected_freq = midiToFrequency(midi_note);
    if (expected_freq <= 0.0f) {
        return 0.0f;
    }

    // Cents = 1200 * log2(frequency / expected_frequency)
    constexpr float kEpsilon = 1e-6f;
    if (expected_freq < kEpsilon) {
        return 0.0f;
    }
    float ratio = frequency_hz / expected_freq;
    if (ratio < kEpsilon) {
        return -50.0f;  // Clamp to minimum
    }
    float cents = 1200.0f * std::log2(ratio);

    // Clamp to reasonable range
    return std::clamp(cents, -50.0f, 50.0f);
}

float PitchTracker::calculateConfidence(
    const float* samples,
    size_t num_samples,
    float detected_frequency,
    float sample_rate_hz
) const noexcept {
    if (!samples || num_samples == 0 || detected_frequency <= 0.0f) {
        return 0.0f;
    }

    // Calculate RMS for signal level
    float rms = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        rms += samples[i] * samples[i];
    }
    rms = std::sqrt(rms / num_samples);

    // Factor 1: Signal level (higher = more confident)
    float level_confidence = std::min(rms * 2.0f, 1.0f);
    if (rms < 0.001f) {
        level_confidence = 0.0f; // Too quiet
    }

    // Factor 2: Frequency validity (within expected range)
    float range_confidence = 1.0f;
    if (detected_frequency < kMinPitchHz || detected_frequency > kMaxPitchHz) {
        range_confidence = 0.1f; // Very low confidence for out-of-range
    }

    // Factor 3: Harmonicity (check if signal is periodic)
    // Estimate by checking autocorrelation at detected period
    float period_samples = sample_rate_hz / detected_frequency;
    size_t period_int = static_cast<size_t>(std::round(period_samples));
    float harmonicity = 0.0f;
    if (period_int > 0 && period_int < num_samples / 2) {
        float corr = 0.0f;
        size_t valid_samples = num_samples - period_int;
        for (size_t i = 0; i < valid_samples; ++i) {
            corr += samples[i] * samples[i + period_int];
        }
        if (valid_samples > 0) {
            corr /= static_cast<float>(valid_samples);
            // Normalize by RMS
            if (rms > 0.0f) {
                harmonicity = std::abs(corr) / (rms * rms);
                harmonicity = std::clamp(harmonicity, 0.0f, 1.0f);
            }
        }
    }

    // Factor 4: Temporal stability (check consistency over time)
    // Simplified: check if RMS is consistent
    float stability = 1.0f;
    if (num_samples > 100) {
        // Check RMS in first and second half
        float rms1 = 0.0f, rms2 = 0.0f;
        size_t half = num_samples / 2;
        for (size_t i = 0; i < half; ++i) {
            rms1 += samples[i] * samples[i];
        }
        rms1 = std::sqrt(rms1 / half);
        for (size_t i = half; i < num_samples; ++i) {
            rms2 += samples[i] * samples[i];
        }
        rms2 = std::sqrt(rms2 / (num_samples - half));

        if (rms1 > 0.0f && rms2 > 0.0f) {
            float ratio = std::min(rms1, rms2) / std::max(rms1, rms2);
            stability = ratio; // Closer to 1.0 = more stable
        }
    }

    // Weighted combination
    float confidence = (
        level_confidence * 0.3f +
        range_confidence * 0.2f +
        harmonicity * 0.3f +
        stability * 0.2f
    );

    return std::clamp(confidence, 0.0f, 1.0f);
}

bool PitchTracker::validatePitchResult(
    const PitchResult& result,
    float sample_rate_hz
) const noexcept {
    // Check frequency is in valid range
    if (result.frequency_hz < kMinPitchHz || result.frequency_hz > kMaxPitchHz) {
        return false;
    }

    // Check MIDI note is valid
    if (result.midi_note < 0 || result.midi_note > 127) {
        return false;
    }

    // Check cents offset is reasonable
    if (std::abs(result.cents_offset) > 50.0f) {
        return false; // Too far from MIDI note
    }

    // Check confidence is valid
    if (result.confidence < 0.0f || result.confidence > 1.0f) {
        return false;
    }

    // Check frequency matches MIDI note (within tolerance)
    float expected_freq = midiToFrequency(result.midi_note);
    if (expected_freq <= 0.0f) {
        return false;
    }

    float freq_diff = std::abs(result.frequency_hz - expected_freq);
    float cents_diff = 1200.0f * std::log2(result.frequency_hz / expected_freq);

    // Cents offset should match calculated difference (within 5 cents tolerance)
    if (std::abs(cents_diff - result.cents_offset) > 5.0f) {
        return false; // Mismatch
    }

    return true;
}

void PitchTracker::applyWindow(
    const float* input,
    float* output,
    size_t num_samples
) const noexcept {
    if (!input || !output || num_samples == 0) {
        return;
    }

    // Hann window: w(n) = 0.5 * (1 - cos(2πn / (N-1)))
    for (size_t i = 0; i < num_samples; ++i) {
        constexpr float PI = 3.14159265358979323846f;
        float window_value = 0.5f * (1.0f - std::cos(2.0f * PI * static_cast<float>(i) / static_cast<float>(num_samples - 1)));
        output[i] = input[i] * window_value;
    }
}

void PitchTracker::normalizeAutocorrelation(
    float* autocorr,
    size_t num_samples
) const noexcept {
    if (!autocorr || num_samples == 0) {
        return;
    }

    // Find maximum for normalization
    float max_val = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        max_val = std::max(max_val, std::abs(autocorr[i]));
    }

    // Normalize
    if (max_val > 0.0f) {
        for (size_t i = 0; i < num_samples; ++i) {
            autocorr[i] /= max_val;
        }
    }
}

} // namespace prrot
