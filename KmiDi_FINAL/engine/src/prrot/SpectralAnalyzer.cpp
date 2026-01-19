#include "prrot/SpectralAnalyzer.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace prrot {

SpectralAnalyzer::SpectralAnalyzer() {
    fft_real_.fill(0.0f);
    fft_imag_.fill(0.0f);
    magnitude_buffer_.fill(0.0f);
}

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
    // Simplified FFT placeholder
    // In production, would use optimized FFT (KissFFT, FFTW with pre-allocated buffers)
    // For now, just copy input to real part and zero imag part
    if (real != input) {
        std::copy(input, input + size, real);
    }
    std::fill(imag, imag + size, 0.0f);
}

} // namespace prrot
