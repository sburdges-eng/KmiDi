#include "prrot/SpectralAnalyzer.h"
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
    if (!audio_samples || !magnitude_output || num_samples == 0 || num_bins == 0) {
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

    if (!audio_samples || num_samples == 0 || sample_rate_hz <= 0.0f) {
        return formants;
    }

    // Compute magnitude spectrum
    computeMagnitudeSpectrum(audio_samples, num_samples, magnitude_buffer_.data(), kMaxSpectralBins);

    // Find peaks in magnitude spectrum (simplified formant detection)
    // In production, would use more sophisticated methods (LPC, cepstral analysis, etc.)
    float bin_frequency = sample_rate_hz / (2.0f * kMaxSpectralBins);

    // Find first three peaks (formants)
    std::vector<std::pair<float, size_t>> peaks; // (magnitude, bin_index)

    for (size_t i = 1; i < kMaxSpectralBins - 1; ++i) {
        if (magnitude_buffer_[i] > magnitude_buffer_[i - 1] &&
            magnitude_buffer_[i] > magnitude_buffer_[i + 1] &&
            magnitude_buffer_[i] > 0.1f) { // Threshold
            peaks.push_back({magnitude_buffer_[i], i});
        }
    }

    // Sort by magnitude and take top 3
    std::sort(peaks.begin(), peaks.end(),
              [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
                  return a.first > b.first;
              });

    if (peaks.size() > 0) {
        formants.f1_hz = peaks[0].second * bin_frequency;
    }
    if (peaks.size() > 1) {
        formants.f2_hz = peaks[1].second * bin_frequency;
    }
    if (peaks.size() > 2) {
        formants.f3_hz = peaks[2].second * bin_frequency;
    }

    // Estimate bandwidths (simplified)
    formants.bandwidth_f1 = formants.f1_hz * 0.1f;
    formants.bandwidth_f2 = formants.f2_hz * 0.1f;
    formants.bandwidth_f3 = formants.f3_hz * 0.1f;

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

    float bin_frequency = sample_rate_hz / (2.0f * num_bins);
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

    float total_energy = 0.0f;
    for (size_t i = 0; i < num_bins; ++i) {
        total_energy += magnitude_spectrum[i] * magnitude_spectrum[i];
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
    if (!input || !real || !imag || size == 0 || size > kFFTSize || !fft_) {
        return;
    }

    // Copy input to real buffer (zero-pad if needed)
    std::memset(real, 0, kFFTSize * sizeof(float));
    std::copy(input, input + size, real);
    std::memset(imag, 0, kFFTSize * sizeof(float));

    // JUCE FFT uses interleaved complex format for real-only input
    // Allocate buffer: size must be 2 * getSize() for real-only forward transform
    // First half contains input, second half will contain output
    std::vector<float> fft_data(kFFTSize * 2, 0.0f);

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
