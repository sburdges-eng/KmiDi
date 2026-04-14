#include "prrot/PhonemeSegmenter.h"
#include "prrot/InputValidation.h"
#include "penta/common/RTLogger.h"
#include <juce_dsp/juce_dsp.h>  // JUCE FFT (juce::dsp::FFT)
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

namespace prrot {

// PIMPL wrapper for JUCE FFT to avoid exposing JUCE in header
struct PhonemeSegmenter::FFTImpl {
    juce::dsp::FFT fft;  // JUCE FFT

    FFTImpl() : fft(11) {}  // 2^11 = 2048, order = log2(size)
};

PhonemeSegmenter::PhonemeSegmenter() {
    // Initialize all buffers to zero
    analysis_buffer_.fill(0.0f);
    magnitude_spectrum_.fill(0.0f);
    power_spectrum_.fill(0.0f);
    fft_real_.fill(0.0f);
    fft_imag_.fill(0.0f);

    // Initialize JUCE FFT via PIMPL
    fft_ = std::make_unique<FFTImpl>();
}

PhonemeSegmenter::~PhonemeSegmenter() = default;  // Destructor needed for PIMPL

void PhonemeSegmenter::initialize(penta::RTMemoryPool* memory_pool) {
    memory_pool_ = memory_pool;

    // Memory pool is recommended for RT-safety, but we can work without it
    // using stack-allocated buffers
    if (!memory_pool_) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "PhonemeSegmenter::initialize: No memory pool provided, using stack buffers");
    }

    // Allocate FFT buffers from memory pool if available
    // Note: In RT-safe context, these should be pre-allocated at startup
    // For now, we use stack-allocated arrays, but in production these
    // should come from a pre-allocated pool
}

PhonemeSegmenter::SegmentResult PhonemeSegmenter::segment(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
)  {
    SegmentResult result;
    result.valid = false;
    result.confidence = 0.0f;

    // Validate inputs
    auto validation = validateAudioInput(audio_samples, num_samples, sample_rate_hz);
    if (validation.hasErrors()) {
        penta::getLogger().logRT(penta::LogLevel::Error,
            ("PhonemeSegmenter::segment: Input validation failed: " +
             validation.errorMessage()).c_str());
        return result;
    }

    // Clamp to max buffer size with bounds checking
    if (num_samples > kMaxSegmentBufferSize) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            ("PhonemeSegmenter::segment: Clamping samples from " +
             std::to_string(num_samples) + " to " +
             std::to_string(kMaxSegmentBufferSize)).c_str());
    }
    size_t samples_to_process = std::min(num_samples, kMaxSegmentBufferSize);

    // Copy to analysis buffer (RT-safe: no allocation)
    std::memcpy(analysis_buffer_.data(), audio_samples, samples_to_process * sizeof(float));

    // Multi-feature segmentation with adaptive threshold
    // Calculate adaptive threshold based on audio characteristics
    float energy_threshold = computeAdaptiveEnergyThreshold(
        analysis_buffer_.data(), samples_to_process
    );

    std::vector<float> energies;
    size_t frame_size = static_cast<size_t>(sample_rate_hz * 0.01f); // 10ms frames

    for (size_t i = 0; i < samples_to_process; i += frame_size) {
        float energy = 0.0f;
        size_t end = std::min(i + frame_size, samples_to_process);

        for (size_t j = i; j < end; ++j) {
            energy += analysis_buffer_[j] * analysis_buffer_[j];
        }
        size_t window_samples = end - i;
        if (window_samples > 0) {
            energy = std::sqrt(energy / static_cast<float>(window_samples));
        } else {
            energy = 0.0f;
        }
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
    if (result.boundaries_ms.size() > 1) {
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
    }

    result.valid = !result.boundaries_ms.empty() && result.boundaries_ms.size() > 1;

    // Calculate real confidence based on segmentation quality
    if (result.valid) {
        result.confidence = calculateSegmentationConfidence(
            result, energies, samples_to_process, sample_rate_hz
        );
    } else {
        result.confidence = 0.0f;
    }

    // Validate segmentation results
    if (result.valid && !validateSegmentation(result, sample_rate_hz)) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "PhonemeSegmenter::segment: Segmentation validation failed");
        result.valid = false;
        result.confidence = 0.0f;
    }

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

    // Compute spectral features using real FFT
    size_t fft_size = std::min(static_cast<size_t>(kPhonemeFFTSize), num_samples);

    // Copy input to FFT buffer (zero-pad if needed)
    std::memset(fft_real_.data(), 0, kPhonemeFFTSize * sizeof(float));
    std::copy(audio_samples, audio_samples + fft_size, fft_real_.begin());
    fft_imag_.fill(0.0f);

    // Compute FFT
    computeFFT(fft_real_.data(), fft_real_.data(), fft_imag_.data(), fft_size);

    // Compute magnitude spectrum from FFT results
    computeMagnitudeSpectrum(fft_real_.data(), fft_imag_.data(), magnitude_spectrum_.data(), fft_size);

    // Extract spectral features from magnitude spectrum
    size_t num_bins = fft_size / 2;
    float spectral_centroid = computeSpectralCentroid(magnitude_spectrum_.data(), num_bins);
    float spectral_rolloff = computeSpectralRolloff(magnitude_spectrum_.data(), num_bins);
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

    // Calculate confidence based on onset/offset clarity
    float onset_clarity = 0.0f;
    float offset_clarity = 0.0f;

    // Helper lambda to compute RMS energy
    auto computeEnergy = [](const float* samples, size_t count) -> float {
        if (!samples || count == 0) return 0.0f;
        float sum_sq = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            sum_sq += samples[i] * samples[i];
        }
        return std::sqrt(sum_sq / count);
    };

    if (result.onset_sample > 0 && result.onset_sample < num_samples) {
        // Check energy change at onset
        float pre_onset_energy = computeEnergy(audio_samples, result.onset_sample);
        size_t post_samples = num_samples - result.onset_sample;
        float post_onset_energy = computeEnergy(
            audio_samples + result.onset_sample,
            post_samples
        );
        if (pre_onset_energy > 0.0f) {
            onset_clarity = std::clamp(post_onset_energy / pre_onset_energy, 0.0f, 1.0f);
        }
    }

    if (result.offset_sample > 0 && result.offset_sample < num_samples) {
        // Check energy change at offset
        float pre_offset_energy = computeEnergy(audio_samples, result.offset_sample);
        size_t post_samples = num_samples - result.offset_sample;
        float post_offset_energy = computeEnergy(
            audio_samples + result.offset_sample,
            post_samples
        );
        if (pre_offset_energy > 0.0f) {
            offset_clarity = std::clamp(pre_offset_energy / (post_offset_energy + 1e-6f), 0.0f, 1.0f);
        }
    }

    result.confidence = (onset_clarity + offset_clarity) / 2.0f;

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
        size_t window_samples = i - start;
        if (window_samples > 0) {
            energy = std::sqrt(energy / static_cast<float>(window_samples));
        } else {
            energy = 0.0f;
        }

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

    constexpr float kEpsilon = 1e-6f;
    if (magnitude_sum > kEpsilon) {
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

    if (num_samples > 0) {
        return static_cast<float>(crossings) / static_cast<float>(num_samples);
    }
    return 0.0f;
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

float PhonemeSegmenter::computeAdaptiveEnergyThreshold(
    const float* samples,
    size_t num_samples
) const noexcept {
    if (!samples || num_samples == 0) {
        return 0.01f; // Default threshold
    }

    // Calculate RMS energy
    float rms = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        rms += samples[i] * samples[i];
    }
    rms = std::sqrt(rms / static_cast<float>(num_samples));

    // Estimate noise floor (use minimum energy in small windows)
    float min_energy = rms;
    size_t window_size = std::max(static_cast<size_t>(1), std::min(num_samples / 10, static_cast<size_t>(100)));
    for (size_t i = 0; i <= num_samples - window_size; i += window_size) {
        float window_energy = 0.0f;
        for (size_t j = i; j < i + window_size; ++j) {
            window_energy += analysis_buffer_[j] * analysis_buffer_[j];
        }
        window_energy = std::sqrt(window_energy / static_cast<float>(window_size));
        min_energy = std::min(min_energy, window_energy);
    }

    // Adaptive threshold: noise floor + 3dB
    float threshold = min_energy * 1.414f; // 3dB = sqrt(2)

    // Clamp to reasonable range
    return std::clamp(threshold, 0.001f, 0.1f);
}

float PhonemeSegmenter::calculateSegmentationConfidence(
    const SegmentResult& result,
    const std::vector<float>& energies,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    if (!result.valid || result.boundaries_ms.size() < 2) {
        return 0.0f;
    }

    float confidence = 1.0f;

    // Factor 1: Boundary clarity (how clear are the transitions)
    float boundary_clarity = 1.0f;
    for (size_t i = 0; i < result.boundaries_ms.size(); ++i) {
        size_t boundary_frame = static_cast<size_t>(
            (result.boundaries_ms[i] / 1000.0f) * sample_rate_hz / (sample_rate_hz * 0.01f)
        );
        if (boundary_frame < energies.size()) {
            // Check if energy change is significant at boundary
            float energy_change = 0.0f;
            if (boundary_frame > 0 && energies.size() > 1 && boundary_frame < energies.size() - 1) {
                energy_change = std::abs(energies[boundary_frame] - energies[boundary_frame - 1]);
            }
            // Low energy change = low clarity
            if (energy_change < 0.01f) {
                boundary_clarity *= 0.8f;
            }
        }
    }
    confidence *= boundary_clarity;

    // Factor 2: Segment duration consistency
    float min_duration = std::numeric_limits<float>::max();
    float max_duration = 0.0f;
    if (result.boundaries_ms.size() > 1) {
        for (size_t i = 0; i < result.boundaries_ms.size() - 1; ++i) {
            float duration = result.boundaries_ms[i + 1] - result.boundaries_ms[i];
            min_duration = std::min(min_duration, duration);
            max_duration = std::max(max_duration, duration);
        }
    }
    if (max_duration > 0.0f) {
        float duration_ratio = min_duration / max_duration;
        // Very inconsistent durations reduce confidence
        if (duration_ratio < 0.3f) {
            confidence *= 0.7f;
        }
    }

    // Factor 3: Number of segments (too many or too few reduces confidence)
    size_t num_segments = result.phonemes.size();
    if (num_segments == 0) {
        confidence = 0.0f;
    } else if (num_segments > 50) {
        confidence *= 0.8f; // Too many segments
    } else if (num_segments < 3) {
        confidence *= 0.6f; // Too few segments
    }

    return std::clamp(confidence, 0.0f, 1.0f);
}

bool PhonemeSegmenter::validateSegmentation(
    const SegmentResult& result,
    float sample_rate_hz
) const noexcept {
    if (!result.valid) {
        return false;
    }

    constexpr float kMinPhonemeDurationMs = 10.0f; // Minimum 10ms per phoneme
    constexpr float kMaxPhonemeDurationMs = 2000.0f; // Maximum 2s per phoneme
    constexpr size_t kMaxPhonemes = 256;

    // Check boundaries are in order
    if (result.boundaries_ms.size() > 1) {
        for (size_t i = 0; i < result.boundaries_ms.size() - 1; ++i) {
            if (result.boundaries_ms[i] >= result.boundaries_ms[i + 1]) {
                return false; // Boundaries out of order
            }
        }
    }

    // Check segment durations
    if (result.boundaries_ms.size() > 1) {
        for (size_t i = 0; i < result.boundaries_ms.size() - 1; ++i) {
            float duration = result.boundaries_ms[i + 1] - result.boundaries_ms[i];
            if (duration < kMinPhonemeDurationMs || duration > kMaxPhonemeDurationMs) {
                return false; // Invalid duration
            }
        }
    }

    // Check number of phonemes
    if (result.phonemes.size() > kMaxPhonemes) {
        return false; // Too many phonemes
    }

    // Check phoneme count matches boundary count
    if (result.boundaries_ms.size() < 2) {
        return false; // Need at least 2 boundaries
    }

    // Should have one less phoneme than boundaries (boundaries define segments)
    if (result.phonemes.size() != result.boundaries_ms.size() - 1) {
        return false; // Mismatch
    }

    return true;
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
    // Validate inputs
    if (!real || !imag || size == 0 || !fft_) {
        if (!real || !imag) {
            penta::getLogger().logRT(penta::LogLevel::Error,
                "PhonemeSegmenter::computeFFT: Null pointer argument");
        }
        return;
    }

    // Copy input to real buffer if it's not already there
    if (input && input != real) {
        std::memset(real, 0, kPhonemeFFTSize * sizeof(float));
        std::copy(input, input + std::min(size, kPhonemeFFTSize), real);
    } else if (!input) {
        // If no input provided, assume data is already in 'real' and just zero-pad the rest
        if (size < kPhonemeFFTSize) {
            std::memset(real + size, 0, (kPhonemeFFTSize - size) * sizeof(float));
        }
    }
    // If input == real, data is already in place (but we should still zero-pad the rest if needed)
    else if (input == real && size < kPhonemeFFTSize) {
        std::memset(real + size, 0, (kPhonemeFFTSize - size) * sizeof(float));
    }

    // Always ensure imag is zeroed for real-only forward transform
    std::memset(imag, 0, kPhonemeFFTSize * sizeof(float));

    // JUCE FFT uses interleaved complex format for real-only input
    // Allocate buffer: size must be 2 * getSize() for real-only forward transform
    // First half contains input, second half will contain output
    std::vector<float> fft_data(kPhonemeFFTSize * 2, 0.0f);

    // Copy real input to first half of buffer
    std::copy(real, real + kPhonemeFFTSize, fft_data.data());

    // Perform in-place forward FFT on real data
    // This is optimized for real-only input (no imaginary part)
    fft_->fft.performRealOnlyForwardTransform(fft_data.data(), false);  // false = calculate all frequencies

    // Extract real and imaginary parts from interleaved format
    // For real input, output is symmetric - we only need first half + DC + Nyquist
    size_t output_size = kPhonemeFFTSize / 2 + 1;
    for (size_t i = 0; i < output_size; ++i) {
        real[i] = fft_data[i * 2];      // real part
        imag[i] = fft_data[i * 2 + 1];  // imaginary part
    }

    // Zero out remaining bins (mirror of lower half for real input)
    if (output_size < kPhonemeFFTSize) {
        std::memset(real + output_size, 0, (kPhonemeFFTSize - output_size) * sizeof(float));
        std::memset(imag + output_size, 0, (kPhonemeFFTSize - output_size) * sizeof(float));
    }
}

void PhonemeSegmenter::computeMagnitudeSpectrum(const float* real, const float* imag, float* magnitude, size_t size) noexcept {
    for (size_t i = 0; i < size; ++i) {
        magnitude[i] = std::sqrt(real[i] * real[i] + imag[i] * imag[i]);
    }
}

} // namespace prrot
