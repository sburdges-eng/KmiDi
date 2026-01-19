#include "prrot/BreathDetector.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace prrot {

BreathDetector::BreathDetector() {
    analysis_buffer_.fill(0.0f);
}

BreathDetector::BreathMarker BreathDetector::detectBreath(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    BreathMarker marker;

    if (!audio_samples || num_samples == 0 || sample_rate_hz <= 0.0f) {
        return marker;
    }

    size_t samples_to_analyze = std::min(num_samples, kMaxBreathAnalysisWindow);
    std::copy(audio_samples, audio_samples + samples_to_analyze, analysis_buffer_.begin());

    // Breath detection: look for high-frequency, low-energy content
    float total_energy = computeEnergy(analysis_buffer_.data(), samples_to_analyze);
    float hf_energy = computeHighFrequencyEnergy(analysis_buffer_.data(), samples_to_analyze, sample_rate_hz);

    // Breath has high HF content relative to total energy
    float hf_ratio = (total_energy > 1e-6f) ? (hf_energy / total_energy) : 0.0f;

    // Detect breath regions (typically at phrase boundaries, low energy)
    float energy_threshold = 0.01f;
    float breath_threshold = 0.3f; // HF ratio threshold

    size_t window_size = static_cast<size_t>(sample_rate_hz * 0.01f); // 10ms windows
    size_t num_windows = samples_to_analyze / window_size;

    float max_breath_ratio = 0.0f;
    size_t breath_window = 0;

    for (size_t i = 0; i < num_windows; ++i) {
        size_t start = i * window_size;
        size_t end = std::min(start + window_size, samples_to_analyze);

        float window_energy = computeEnergy(analysis_buffer_.data() + start, end - start);
        float window_hf = computeHighFrequencyEnergy(analysis_buffer_.data() + start, end - start, sample_rate_hz);
        float window_hf_ratio = (window_energy > 1e-6f) ? (window_hf / window_energy) : 0.0f;

        // Breath candidate: low energy + high HF ratio
        if (window_energy < energy_threshold && window_hf_ratio > breath_threshold) {
            if (window_hf_ratio > max_breath_ratio) {
                max_breath_ratio = window_hf_ratio;
                breath_window = i;
            }
        }
    }

    if (max_breath_ratio > breath_threshold) {
        marker.sample_index = breath_window * window_size;
        marker.time_ms = (static_cast<float>(marker.sample_index) / sample_rate_hz) * 1000.0f;
        marker.intensity = std::min(max_breath_ratio, 1.0f);
        marker.duration_ms = 200.0f; // Typical breath duration
        marker.confidence = max_breath_ratio;
    }

    return marker;
}

float BreathDetector::estimateNoiseFloor(
    const float* audio_samples,
    size_t num_samples
) const noexcept {
    if (!audio_samples || num_samples == 0) {
        return 0.0f;
    }

    size_t samples_to_analyze = std::min(num_samples, kMaxBreathAnalysisWindow);

    // Estimate noise floor as minimum energy in low-energy regions
    float min_energy = std::numeric_limits<float>::max();
    size_t window_size = 256; // ~6ms at 44.1kHz
    size_t num_windows = samples_to_analyze / window_size;

    for (size_t i = 0; i < num_windows; ++i) {
        size_t start = i * window_size;
        size_t end = std::min(start + window_size, samples_to_analyze);

        float window_energy = computeEnergy(audio_samples + start, end - start);
        min_energy = std::min(min_energy, window_energy);
    }

    // Convert to dB
    if (min_energy > 1e-10f) {
        return 20.0f * std::log10(min_energy);
    }
    return -100.0f; // Very quiet
}

float BreathDetector::estimateBreathWeight(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    if (!audio_samples || num_samples == 0) {
        return 0.0f;
    }

    float total_energy = computeEnergy(audio_samples, num_samples);
    float hf_energy = computeHighFrequencyEnergy(audio_samples, num_samples, sample_rate_hz);

    if (total_energy > 1e-6f) {
        return std::min(hf_energy / total_energy, 1.0f);
    }
    return 0.0f;
}

float BreathDetector::computeEnergy(const float* samples, size_t num_samples) const noexcept {
    float energy = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        energy += samples[i] * samples[i];
    }
    return std::sqrt(energy / num_samples);
}

float BreathDetector::computeHighFrequencyEnergy(
    const float* samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    // Simple high-pass filter approximation
    // In production, would use proper filter

    float cutoff_hz = 3000.0f; // 3kHz high-pass
    float alpha = 1.0f / (1.0f + 2.0f * 3.14159f * cutoff_hz / sample_rate_hz);

    float filtered_energy = 0.0f;
    float prev_sample = 0.0f;

    for (size_t i = 0; i < num_samples; ++i) {
        float filtered = alpha * (prev_sample + samples[i] - prev_sample);
        filtered_energy += filtered * filtered;
        prev_sample = filtered;
    }

    return std::sqrt(filtered_energy / num_samples);
}

} // namespace prrot
