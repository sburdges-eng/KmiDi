#include "prrot/SpectralAnalyzer.h"
#include "prrot/InputValidation.h"
#include "penta/common/RTLogger.h"
#include <juce_dsp/juce_dsp.h>  // JUCE FFT (juce::dsp::FFT)
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <memory>

namespace prrot {

// PIMPL wrapper for JUCE FFT to avoid exposing JUCE in header
struct SpectralAnalyzer::FFTImpl {
    juce::dsp::FFT fft;  // JUCE FFT is not a template, just FFT

    FFTImpl() : fft(11) {}  // 2^11 = 2048, order = log2(size)
};

SpectralAnalyzer::SpectralAnalyzer() {
    fft_real_.fill(0.0f);
    fft_imag_.fill(0.0f);
    magnitude_buffer_.fill(0.0f);
    fft_scratch_.assign(kFFTSize * 2, 0.0f);

    // Initialize JUCE FFT via PIMPL
    fft_ = std::make_unique<FFTImpl>();
}

SpectralAnalyzer::~SpectralAnalyzer() = default;  // Destructor needed for PIMPL

void SpectralAnalyzer::computeMagnitudeSpectrum(
    const float* audio_samples,
    size_t num_samples,
    float* magnitude_output,
    size_t num_bins
) const noexcept {
    // Validate inputs
    if (!audio_samples) {
        penta::getLogger().logRT(penta::LogLevel::Error,
            "SpectralAnalyzer::computeMagnitudeSpectrum: audio_samples is null");
        return;
    }

    if (!magnitude_output) {
        penta::getLogger().logRT(penta::LogLevel::Error,
            "SpectralAnalyzer::computeMagnitudeSpectrum: magnitude_output is null");
        return;
    }

    if (num_samples == 0) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "SpectralAnalyzer::computeMagnitudeSpectrum: num_samples is zero");
        return;
    }

    if (num_bins == 0) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "SpectralAnalyzer::computeMagnitudeSpectrum: num_bins is zero");
        return;
    }

    size_t fft_size = std::min(kFFTSize, num_samples);
    size_t output_bins = std::min(num_bins, kMaxSpectralBins);

    // Copy input to FFT buffer (zero-pad if needed)
    std::memset(fft_real_.data(), 0, kFFTSize * sizeof(float));
    std::copy(audio_samples, audio_samples + fft_size, fft_real_.begin());
    fft_imag_.fill(0.0f);

    // Compute FFT
    computeFFT(fft_real_.data(), fft_real_.data(), fft_imag_.data(), fft_size);

    // Compute magnitude spectrum
    for (size_t i = 0; i < output_bins; ++i) {
        float real = fft_real_[i];
        float imag = fft_imag_[i];
        magnitude_output[i] = std::sqrt(real * real + imag * imag);
    }
}

SpectralAnalyzer::FormantFrequencies SpectralAnalyzer::extractFormants(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) const noexcept {
    FormantFrequencies formants;

    // Validate inputs
    auto validation = validateAudioInput(audio_samples, num_samples, sample_rate_hz);
    if (validation.hasErrors()) {
        penta::getLogger().logRT(penta::LogLevel::Error,
            ("SpectralAnalyzer::extractFormants: Input validation failed: " +
             validation.errorMessage()).c_str());
        return formants;
    }

    // Compute magnitude spectrum
    computeMagnitudeSpectrum(audio_samples, num_samples, magnitude_buffer_.data(), kMaxSpectralBins);

    // Enhanced formant detection with frequency-specific ranges and parabolic interpolation
    float bin_frequency = sample_rate_hz / (2.0f * kMaxSpectralBins);

    // Formant frequency ranges (typical for human voice)
    constexpr float F1_MIN = 200.0f;   // Hz
    constexpr float F1_MAX = 1200.0f;
    constexpr float F2_MIN = 600.0f;
    constexpr float F2_MAX = 3000.0f;
    constexpr float F3_MIN = 1500.0f;
    constexpr float F3_MAX = 4000.0f;

    // Convert to bin indices
    size_t f1_min_bin = static_cast<size_t>(F1_MIN / bin_frequency);
    size_t f1_max_bin = static_cast<size_t>(std::min(F1_MAX / bin_frequency, static_cast<float>(kMaxSpectralBins - 1)));
    size_t f2_min_bin = static_cast<size_t>(F2_MIN / bin_frequency);
    size_t f2_max_bin = static_cast<size_t>(std::min(F2_MAX / bin_frequency, static_cast<float>(kMaxSpectralBins - 1)));
    size_t f3_min_bin = static_cast<size_t>(F3_MIN / bin_frequency);
    size_t f3_max_bin = static_cast<size_t>(std::min(F3_MAX / bin_frequency, static_cast<float>(kMaxSpectralBins - 1)));

    // Helper lambda for parabolic interpolation to get sub-bin accuracy
    auto parabolicInterpolation = [this](size_t peak_bin) -> float {
        if (peak_bin == 0 || peak_bin >= kMaxSpectralBins - 1) {
            return static_cast<float>(peak_bin);
        }

        float y1 = magnitude_buffer_[peak_bin - 1];
        float y2 = magnitude_buffer_[peak_bin];
        float y3 = magnitude_buffer_[peak_bin + 1];

        // Parabolic interpolation: find peak location
        float denom = 2.0f * (y1 - 2.0f * y2 + y3);
        if (std::abs(denom) < 1e-6f) {
            return static_cast<float>(peak_bin);
        }

        float offset = (y1 - y3) / denom;
        return static_cast<float>(peak_bin) + offset;
    };

    // Helper lambda to find peak in a range with parabolic interpolation
    auto findPeakInRange = [&](size_t min_bin, size_t max_bin) -> std::pair<float, float> {
        float max_magnitude = 0.0f;
        size_t peak_bin = min_bin;

        // Find local maximum
        for (size_t i = min_bin + 1; i < max_bin && i < kMaxSpectralBins - 1; ++i) {
            // Check if it's a local peak
            if (magnitude_buffer_[i] > magnitude_buffer_[i - 1] &&
                magnitude_buffer_[i] > magnitude_buffer_[i + 1] &&
                magnitude_buffer_[i] > max_magnitude) {
                max_magnitude = magnitude_buffer_[i];
                peak_bin = i;
            }
        }

        if (max_magnitude > 0.01f) {  // Minimum threshold
            float interpolated_bin = parabolicInterpolation(peak_bin);
            float frequency = interpolated_bin * bin_frequency;
            return {frequency, max_magnitude};
        }

        return {0.0f, 0.0f};
    };

    // Find formants in their respective frequency ranges
    auto f1_result = findPeakInRange(f1_min_bin, f1_max_bin);
    if (f1_result.first > 0.0f) {
        formants.f1_hz = std::clamp(f1_result.first, F1_MIN, F1_MAX);
    }

    auto f2_result = findPeakInRange(f2_min_bin, f2_max_bin);
    if (f2_result.first > 0.0f) {
        formants.f2_hz = std::clamp(f2_result.first, F2_MIN, F2_MAX);
    }

    auto f3_result = findPeakInRange(f3_min_bin, f3_max_bin);
    if (f3_result.first > 0.0f) {
        formants.f3_hz = std::clamp(f3_result.first, F3_MIN, F3_MAX);
    }

    // Improved bandwidth estimation using 3dB down points
    auto estimateBandwidth = [&](float formant_freq, size_t formant_bin) -> float {
        if (formant_freq <= 0.0f || formant_bin >= kMaxSpectralBins) {
            return 0.0f;
        }

        float peak_magnitude = magnitude_buffer_[formant_bin];
        float threshold = peak_magnitude * 0.707f;  // 3dB down (sqrt(2)/2)

        // Find left and right 3dB points
        size_t left_bin = formant_bin;
        while (left_bin > 0 && magnitude_buffer_[left_bin] > threshold) {
            --left_bin;
        }

        size_t right_bin = formant_bin;
        while (right_bin < kMaxSpectralBins - 1 && magnitude_buffer_[right_bin] > threshold) {
            ++right_bin;
        }

        float bandwidth_hz = (static_cast<float>(right_bin - left_bin)) * bin_frequency;
        return std::clamp(bandwidth_hz, 10.0f, 500.0f);
    };

    // Estimate bandwidths
    if (formants.f1_hz > 0.0f) {
        size_t f1_bin = static_cast<size_t>(formants.f1_hz / bin_frequency);
        formants.bandwidth_f1 = estimateBandwidth(formants.f1_hz, f1_bin);
    }
    if (formants.f2_hz > 0.0f) {
        size_t f2_bin = static_cast<size_t>(formants.f2_hz / bin_frequency);
        formants.bandwidth_f2 = estimateBandwidth(formants.f2_hz, f2_bin);
    }
    if (formants.f3_hz > 0.0f) {
        size_t f3_bin = static_cast<size_t>(formants.f3_hz / bin_frequency);
        formants.bandwidth_f3 = estimateBandwidth(formants.f3_hz, f3_bin);
    }

    return formants;
}

float SpectralAnalyzer::computeSpectralCentroid(
    const float* magnitude_spectrum,
    size_t num_bins,
    float sample_rate_hz
) const noexcept {
    if (!magnitude_spectrum || num_bins == 0) {
        return 0.0f;
    }

    constexpr float kEpsilon = 1e-6f;
    if (num_bins == 0 || sample_rate_hz < kEpsilon) {
        return 0.0f;
    }
    float bin_frequency = sample_rate_hz / (2.0f * static_cast<float>(num_bins));
    float weighted_sum = 0.0f;
    float magnitude_sum = 0.0f;

    for (size_t i = 0; i < num_bins; ++i) {
        float frequency = static_cast<float>(i) * bin_frequency;
        float magnitude = magnitude_spectrum[i];
        weighted_sum += frequency * magnitude;
        magnitude_sum += magnitude;
    }

    if (magnitude_sum > 1e-6f) {
        return weighted_sum / magnitude_sum;
    }
    return 0.0f;
}

float SpectralAnalyzer::computeSpectralRolloff(
    const float* magnitude_spectrum,
    size_t num_bins,
    float percentile
) const noexcept {
    if (!magnitude_spectrum || num_bins == 0) {
        return 0.0f;
    }

    constexpr float kEpsilon = 1e-6f;
    if (num_bins == 0) {
        return 0.0f;
    }

    float total_energy = 0.0f;
    for (size_t i = 0; i < num_bins; ++i) {
        total_energy += magnitude_spectrum[i] * magnitude_spectrum[i];
    }

    if (total_energy < kEpsilon) {
        return 0.0f;
    }

    float threshold = total_energy * percentile;
    float cumulative_energy = 0.0f;

    for (size_t i = 0; i < num_bins; ++i) {
        cumulative_energy += magnitude_spectrum[i] * magnitude_spectrum[i];
        if (cumulative_energy >= threshold) {
            return static_cast<float>(i) / static_cast<float>(num_bins);
        }
    }

    return 1.0f;
}

float SpectralAnalyzer::computeSpectralFlux(
    const float* current_magnitude,
    const float* previous_magnitude,
    size_t num_bins
) const noexcept {
    if (!current_magnitude || !previous_magnitude || num_bins == 0) {
        return 0.0f;
    }

    float flux = 0.0f;
    for (size_t i = 0; i < num_bins; ++i) {
        float diff = current_magnitude[i] - previous_magnitude[i];
        if (diff > 0.0f) {
            flux += diff * diff;
        }
    }

    return std::sqrt(flux);
}

void SpectralAnalyzer::computeFFT(const float* input, float* real, float* imag, size_t size) const noexcept {
    if (!real || !imag || size == 0 || !fft_) {
        return;
    }

    // Copy input to real buffer if it's not already there
    if (input && input != real) {
        std::memset(real, 0, kFFTSize * sizeof(float));
        std::copy(input, input + std::min(size, kFFTSize), real);
    } else if (!input) {
        // If no input provided, assume data is already in 'real' and just zero-pad the rest
        if (size < kFFTSize) {
            std::memset(real + size, 0, (kFFTSize - size) * sizeof(float));
        }
    }
    // If input == real, data is already in place (but we should still zero-pad the rest if needed)
    else if (input == real && size < kFFTSize) {
        std::memset(real + size, 0, (kFFTSize - size) * sizeof(float));
    }

    // Always ensure imag is zeroed for real-only forward transform
    std::memset(imag, 0, kFFTSize * sizeof(float));

    // JUCE FFT uses interleaved complex format for real-only input
    // Use pre-allocated scratch buffer (resized once in constructor) to avoid
    // per-call heap allocation on the hot path and keep noexcept honest.
    std::fill(fft_scratch_.begin(), fft_scratch_.end(), 0.0f);
    std::vector<float>& fft_data = fft_scratch_;

    // Copy real input to first half of buffer
    std::copy(real, real + kFFTSize, fft_data.data());

    // Perform in-place forward FFT on real data
    // This is optimized for real-only input (no imaginary part)
    fft_->fft.performRealOnlyForwardTransform(fft_data.data(), false);  // false = calculate all frequencies

    // Extract real and imaginary parts from interleaved format
    // For real input, output is symmetric - we only need first half + DC + Nyquist
    size_t output_size = kFFTSize / 2 + 1;
    for (size_t i = 0; i < output_size; ++i) {
        real[i] = fft_data[i * 2];      // real part
        imag[i] = fft_data[i * 2 + 1];  // imaginary part
    }

    // Zero out remaining bins (mirror of lower half for real input)
    if (output_size < kFFTSize) {
        std::memset(real + output_size, 0, (kFFTSize - output_size) * sizeof(float));
        std::memset(imag + output_size, 0, (kFFTSize - output_size) * sizeof(float));
    }
}

} // namespace prrot
