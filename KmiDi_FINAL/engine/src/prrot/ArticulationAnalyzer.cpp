#include "prrot/ArticulationAnalyzer.h"
#include <algorithm>
#include <cmath>

namespace prrot {

ArticulationAnalyzer::ArticulationAnalyzer() {
    analysis_buffer_.fill(0.0f);
    energy_buffer_.fill(0.0f);
    derivative_buffer_.fill(0.0f);
}

ArticulationAnalyzer::ArticulationClass ArticulationAnalyzer::classifyArticulation(
    PhonemeType phoneme
) const noexcept {
    if (isVowel(phoneme)) {
        return ArticulationClass::Vowel;
    } else if (isConsonant(phoneme)) {
        return ArticulationClass::Consonant;
    }
    return ArticulationClass::Unknown;
}

ArticulationAnalyzer::OnsetInfo ArticulationAnalyzer::detectOnset(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    OnsetInfo info;

    if (!audio_samples || num_samples == 0 || sample_rate_hz <= 0.0f) {
        return info;
    }

    size_t samples_to_analyze = std::min(num_samples, kMaxAnalysisWindow);

    // Copy to analysis buffer
    std::copy(audio_samples, audio_samples + samples_to_analyze, analysis_buffer_.begin());

    // Compute energy envelope
    size_t window_size = static_cast<size_t>(sample_rate_hz * 0.005f); // 5ms windows
    size_t num_windows = samples_to_analyze / window_size;

    for (size_t i = 0; i < num_windows && i < kMaxAnalysisWindow; ++i) {
        float energy = 0.0f;
        size_t start = i * window_size;
        size_t end = std::min(start + window_size, samples_to_analyze);

        for (size_t j = start; j < end; ++j) {
            energy += analysis_buffer_[j] * analysis_buffer_[j];
        }
        energy = std::sqrt(energy / (end - start));
        energy_buffer_[i] = energy;
    }

    // Compute derivative to find sharpest increase
    computeDerivative(energy_buffer_.data(), num_windows, derivative_buffer_.data());

    // Find peak in derivative (sharpest onset)
    float max_derivative = 0.0f;
    size_t peak_index = 0;

    for (size_t i = 0; i < num_windows; ++i) {
        if (derivative_buffer_[i] > max_derivative) {
            max_derivative = derivative_buffer_[i];
            peak_index = i;
        }
    }

    // Convert back to sample index
    info.sample_index = peak_index * window_size;
    info.time_ms = (static_cast<float>(info.sample_index) / sample_rate_hz) * 1000.0f;
    info.sharpness = std::min(max_derivative * 10.0f, 1.0f); // Normalize
    info.confidence = (max_derivative > 0.01f) ? 0.8f : 0.0f;

    return info;
}

ArticulationAnalyzer::OffsetInfo ArticulationAnalyzer::detectOffset(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    OffsetInfo info;

    if (!audio_samples || num_samples == 0 || sample_rate_hz <= 0.0f) {
        return info;
    }

    size_t samples_to_analyze = std::min(num_samples, kMaxAnalysisWindow);
    std::copy(audio_samples, audio_samples + samples_to_analyze, analysis_buffer_.begin());

    // Compute energy envelope (same as onset detection)
    size_t window_size = static_cast<size_t>(sample_rate_hz * 0.005f);
    size_t num_windows = samples_to_analyze / window_size;

    for (size_t i = 0; i < num_windows && i < kMaxAnalysisWindow; ++i) {
        float energy = 0.0f;
        size_t start = i * window_size;
        size_t end = std::min(start + window_size, samples_to_analyze);

        for (size_t j = start; j < end; ++j) {
            energy += analysis_buffer_[j] * analysis_buffer_[j];
        }
        energy = std::sqrt(energy / (end - start));
        energy_buffer_[i] = energy;
    }

    // Find where energy drops below threshold (working backwards)
    float energy_threshold = findPeak(energy_buffer_.data(), num_windows) * 0.3f;

    size_t offset_window = num_windows;
    for (size_t i = num_windows; i > 0; --i) {
        if (energy_buffer_[i - 1] > energy_threshold) {
            offset_window = i;
            break;
        }
    }

    info.sample_index = offset_window * window_size;
    info.time_ms = (static_cast<float>(info.sample_index) / sample_rate_hz) * 1000.0f;

    // Compute smoothness (how gradual the offset is)
    if (offset_window > 0 && offset_window < num_windows) {
        float decay_rate = (energy_buffer_[offset_window - 1] - energy_buffer_[offset_window]) /
                          (energy_buffer_[offset_window - 1] + 1e-6f);
        info.smoothness = std::max(0.0f, 1.0f - decay_rate * 10.0f);
    } else {
        info.smoothness = 0.5f;
    }

    info.confidence = 0.8f;

    return info;
}

ArticulationAnalyzer::ArticulationShape ArticulationAnalyzer::analyzeArticulationShape(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz,
    PhonemeType phoneme
) const noexcept {
    ArticulationShape shape;

    if (!audio_samples || num_samples == 0) {
        return shape;
    }

    size_t samples_to_analyze = std::min(num_samples, kMaxAnalysisWindow);
    std::copy(audio_samples, audio_samples + samples_to_analyze, analysis_buffer_.begin());

    // Detect onset and offset
    OnsetInfo onset = detectOnset(audio_samples, samples_to_analyze, sample_rate_hz);
    OffsetInfo offset = detectOffset(audio_samples, samples_to_analyze, sample_rate_hz);

    shape.attack_time_ms = onset.time_ms;
    shape.release_time_ms = (offset.time_ms > onset.time_ms) ?
                           (offset.time_ms - onset.time_ms) : 10.0f;

    // Find sustain level (peak energy between onset and offset)
    size_t onset_sample = static_cast<size_t>((onset.time_ms / 1000.0f) * sample_rate_hz);
    size_t offset_sample = std::min(static_cast<size_t>((offset.time_ms / 1000.0f) * sample_rate_hz),
                                   samples_to_analyze);

    if (offset_sample > onset_sample) {
        float max_energy = 0.0f;
        for (size_t i = onset_sample; i < offset_sample; ++i) {
            float energy = std::abs(analysis_buffer_[i]);
            max_energy = std::max(max_energy, energy);
        }
        shape.sustain_level = max_energy;
    } else {
        shape.sustain_level = 1.0f;
    }

    // Determine curve shape based on phoneme type
    if (isVowel(phoneme)) {
        shape.attack_curve = 0.3f;  // Gradual attack for vowels
        shape.release_curve = 0.7f; // Smooth release
    } else if (isConsonant(phoneme)) {
        shape.attack_curve = 0.8f;  // Sharp attack for consonants
        shape.release_curve = 0.5f; // Moderate release
    }

    return shape;
}

float ArticulationAnalyzer::computeEnergy(const float* samples, size_t num_samples) const noexcept {
    float energy = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        energy += samples[i] * samples[i];
    }
    return std::sqrt(energy / num_samples);
}

void ArticulationAnalyzer::computeDerivative(const float* samples, size_t num_samples, float* derivative) const noexcept {
    if (num_samples < 2) {
        return;
    }

    for (size_t i = 1; i < num_samples; ++i) {
        derivative[i - 1] = samples[i] - samples[i - 1];
    }
}

float ArticulationAnalyzer::findPeak(const float* samples, size_t num_samples) const noexcept {
    if (num_samples == 0) {
        return 0.0f;
    }

    float peak = std::abs(samples[0]);
    for (size_t i = 1; i < num_samples; ++i) {
        peak = std::max(peak, std::abs(samples[i]));
    }
    return peak;
}

} // namespace prrot
