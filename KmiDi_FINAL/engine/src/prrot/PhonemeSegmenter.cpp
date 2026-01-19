#include "prrot/PhonemeSegmenter.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace prrot {

PhonemeSegmenter::PhonemeSegmenter() {
    // Initialize all buffers to zero
    analysis_buffer_.fill(0.0f);
    magnitude_spectrum_.fill(0.0f);
    power_spectrum_.fill(0.0f);
}

void PhonemeSegmenter::initialize(penta::RTMemoryPool* memory_pool) {
    memory_pool_ = memory_pool;

    // Allocate FFT buffers from memory pool if available
    // Note: In RT-safe context, these should be pre-allocated at startup
    // For now, we use stack-allocated arrays, but in production these
    // should come from a pre-allocated pool
}

PhonemeSegmenter::SegmentResult PhonemeSegmenter::segment(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) noexcept {
    SegmentResult result;
    result.valid = false;
    result.confidence = 0.0f;

    if (!audio_samples || num_samples == 0 || sample_rate_hz <= 0.0f) {
        return result;
    }

    // Clamp to max buffer size
    size_t samples_to_process = std::min(num_samples, kMaxSegmentBufferSize);

    // Copy to analysis buffer (RT-safe: no allocation)
    std::memcpy(analysis_buffer_.data(), audio_samples, samples_to_process * sizeof(float));

    // Simple energy-based segmentation
    // This is a placeholder - actual implementation would use more sophisticated methods
    float energy_threshold = 0.01f; // Adaptive threshold would be better

    std::vector<float> energies;
    size_t frame_size = static_cast<size_t>(sample_rate_hz * 0.01f); // 10ms frames

    for (size_t i = 0; i < samples_to_process; i += frame_size) {
        float energy = 0.0f;
        size_t end = std::min(i + frame_size, samples_to_process);

        for (size_t j = i; j < end; ++j) {
            energy += analysis_buffer_[j] * analysis_buffer_[j];
        }
        energy = std::sqrt(energy / (end - i));
        energies.push_back(energy);
    }

    // Detect boundaries where energy crosses threshold
    bool in_phoneme = false;
    for (size_t i = 0; i < energies.size(); ++i) {
        if (!in_phoneme && energies[i] > energy_threshold) {
            // Onset detected
            float time_ms = (static_cast<float>(i * frame_size) / sample_rate_hz) * 1000.0f;
            result.boundaries_ms.push_back(time_ms);
            in_phoneme = true;
        } else if (in_phoneme && energies[i] < energy_threshold * 0.5f) {
            // Offset detected
            float time_ms = (static_cast<float>(i * frame_size) / sample_rate_hz) * 1000.0f;
            result.boundaries_ms.push_back(time_ms);
            in_phoneme = false;
        }
    }

    // Classify segments (simplified - would need more sophisticated analysis)
    for (size_t i = 0; i < result.boundaries_ms.size() - 1; ++i) {
        size_t start_sample = static_cast<size_t>((result.boundaries_ms[i] / 1000.0f) * sample_rate_hz);
        size_t end_sample = static_cast<size_t>((result.boundaries_ms[i + 1] / 1000.0f) * sample_rate_hz);
        size_t segment_samples = end_sample - start_sample;

        if (start_sample < samples_to_process && segment_samples > 0) {
            PhonemeType phoneme = classifyPhoneme(
                analysis_buffer_.data() + start_sample,
                std::min(segment_samples, samples_to_process - start_sample),
                sample_rate_hz
            );
            result.phonemes.push_back(phoneme);
        }
    }

    result.valid = !result.boundaries_ms.empty() && result.boundaries_ms.size() > 1;
    result.confidence = result.valid ? 0.7f : 0.0f; // Placeholder confidence

    return result;
}

PhonemeType PhonemeSegmenter::classifyPhoneme(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) noexcept {
    if (!audio_samples || num_samples == 0) {
        return PhonemeType::UNKNOWN;
    }

    // Compute spectral features
    size_t fft_size = std::min(static_cast<size_t>(kFFTSize), num_samples);

    // Simple magnitude spectrum computation (simplified)
    // In production, would use proper FFT
    float spectral_centroid = computeSpectralCentroid(magnitude_spectrum_.data(), fft_size / 2);
    float spectral_rolloff = computeSpectralRolloff(magnitude_spectrum_.data(), fft_size / 2);
    float zero_crossing_rate = computeZeroCrossingRate(audio_samples, num_samples);

    // Compute energy
    float energy = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        energy += audio_samples[i] * audio_samples[i];
    }
    energy = std::sqrt(energy / num_samples);

    return classifyByFeatures(spectral_centroid, spectral_rolloff, zero_crossing_rate, energy);
}

PhonemeSegmenter::OnsetOffset PhonemeSegmenter::detectOnsetOffset(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) noexcept {
    OnsetOffset result;

    if (!audio_samples || num_samples == 0) {
        return result;
    }

    // Detect onset
    result.onset_sample = static_cast<size_t>(detectEnergyOnset(audio_samples, num_samples));
    result.onset_time_ms = (static_cast<float>(result.onset_sample) / sample_rate_hz) * 1000.0f;

    // Detect offset
    result.offset_sample = static_cast<size_t>(detectEnergyOffset(audio_samples, num_samples));
    result.offset_time_ms = (static_cast<float>(result.offset_sample) / sample_rate_hz) * 1000.0f;

    result.confidence = 0.8f; // Placeholder

    return result;
}

float PhonemeSegmenter::detectEnergyOnset(const float* samples, size_t num_samples) const noexcept {
    // Simple energy-based onset detection
    float threshold = 0.01f;
    size_t window_size = 128; // ~3ms at 44.1kHz

    for (size_t i = 0; i < num_samples - window_size; i += window_size) {
        float energy = 0.0f;
        for (size_t j = i; j < i + window_size && j < num_samples; ++j) {
            energy += samples[j] * samples[j];
        }
        energy = std::sqrt(energy / window_size);

        if (energy > threshold) {
            return static_cast<float>(i);
        }
    }

    return 0.0f;
}

float PhonemeSegmenter::detectEnergyOffset(const float* samples, size_t num_samples) const noexcept {
    // Detect offset by finding where energy drops below threshold
    float threshold = 0.005f;
    size_t window_size = 128;

    for (size_t i = num_samples; i > window_size; i -= window_size) {
        float energy = 0.0f;
        size_t start = (i > window_size) ? (i - window_size) : 0;
        for (size_t j = start; j < i && j < num_samples; ++j) {
            energy += samples[j] * samples[j];
        }
        energy = std::sqrt(energy / (i - start));

        if (energy > threshold) {
            return static_cast<float>(i);
        }
    }

    return static_cast<float>(num_samples);
}

float PhonemeSegmenter::computeSpectralCentroid(const float* magnitude, size_t num_bins) const noexcept {
    float weighted_sum = 0.0f;
    float magnitude_sum = 0.0f;

    for (size_t i = 0; i < num_bins; ++i) {
        weighted_sum += static_cast<float>(i) * magnitude[i];
        magnitude_sum += magnitude[i];
    }

    if (magnitude_sum > 1e-6f) {
        return weighted_sum / magnitude_sum;
    }
    return 0.0f;
}

float PhonemeSegmenter::computeSpectralRolloff(const float* magnitude, size_t num_bins, float percentile) const noexcept {
    float total_energy = 0.0f;
    for (size_t i = 0; i < num_bins; ++i) {
        total_energy += magnitude[i] * magnitude[i];
    }

    float threshold = total_energy * percentile;
    float cumulative_energy = 0.0f;

    for (size_t i = 0; i < num_bins; ++i) {
        cumulative_energy += magnitude[i] * magnitude[i];
        if (cumulative_energy >= threshold) {
            return static_cast<float>(i) / static_cast<float>(num_bins);
        }
    }

    return 1.0f;
}

float PhonemeSegmenter::computeZeroCrossingRate(const float* samples, size_t num_samples) const noexcept {
    if (num_samples < 2) {
        return 0.0f;
    }

    size_t crossings = 0;
    for (size_t i = 1; i < num_samples; ++i) {
        if ((samples[i - 1] >= 0.0f && samples[i] < 0.0f) ||
            (samples[i - 1] < 0.0f && samples[i] >= 0.0f)) {
            ++crossings;
        }
    }

    return static_cast<float>(crossings) / static_cast<float>(num_samples);
}

PhonemeType PhonemeSegmenter::classifyByFeatures(
    float spectral_centroid,
    float spectral_rolloff,
    float zero_crossing_rate,
    float energy
) const noexcept {
    // Simple rule-based classification
    // In production, would use more sophisticated methods or tiny ML model

    if (isVowelLike(spectral_centroid, spectral_rolloff)) {
        // Classify vowel type based on spectral features
        if (spectral_centroid < 0.2f) {
            return PhonemeType::AH; // Low-frequency vowel
        } else if (spectral_centroid > 0.6f) {
            return PhonemeType::IY; // High-frequency vowel
        } else {
            return PhonemeType::EH; // Mid-frequency vowel
        }
    } else if (isConsonantLike(zero_crossing_rate, energy)) {
        // Classify consonant type
        if (zero_crossing_rate > 0.3f) {
            return PhonemeType::S; // Fricative
        } else if (energy > 0.1f) {
            return PhonemeType::B; // Stop
        } else {
            return PhonemeType::M; // Nasal
        }
    }

    return PhonemeType::UNKNOWN;
}

bool PhonemeSegmenter::isVowelLike(float spectral_centroid, float spectral_rolloff) const noexcept {
    // Vowels typically have lower spectral centroid and higher rolloff
    return spectral_centroid < 0.4f && spectral_rolloff > 0.5f;
}

bool PhonemeSegmenter::isConsonantLike(float zero_crossing_rate, float energy) const noexcept {
    // Consonants typically have higher zero-crossing rate or lower energy
    return zero_crossing_rate > 0.15f || energy < 0.05f;
}

void PhonemeSegmenter::computeFFT(const float* input, float* real, float* imag, size_t size) noexcept {
    // Placeholder for FFT implementation
    // In production, would use optimized FFT (e.g., KissFFT, FFTW with pre-allocated buffers)
    // For RT safety, FFT buffers must be pre-allocated

    // Simplified: just copy input to real, zero imag
    for (size_t i = 0; i < size; ++i) {
        real[i] = input[i];
        imag[i] = 0.0f;
    }
}

void PhonemeSegmenter::computeMagnitudeSpectrum(const float* real, const float* imag, float* magnitude, size_t size) noexcept {
    for (size_t i = 0; i < size; ++i) {
        magnitude[i] = std::sqrt(real[i] * real[i] + imag[i] * imag[i]);
    }
}

} // namespace prrot
