#pragma once

/**
 * SpectralAnalyzer.h - Pitch-Independent Spectral Analysis
 * ========================================================
 *
 * PRROT/PARROT Voice-Instrument Compiler - Tier C (Embedded Core)
 *
 * RT-safe spectral analysis using FFT with pre-allocated buffers.
 * Analyzes spectral shape independently of pitch.
 * Uses JUCE's optimized FFT for real-time performance.
 */

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace prrot {

// FFT size constants - using inline constexpr (C++17) to allow multiple definitions
inline constexpr size_t kFFTSize = 2048;
inline constexpr size_t kMaxSpectralBins = kFFTSize / 2;

/**
 * SpectralAnalyzer - RT-safe spectral analysis
 */
class SpectralAnalyzer {
public:
    SpectralAnalyzer();
    ~SpectralAnalyzer();  // Must be defined in .cpp for PIMPL

    // Compute magnitude spectrum
    void computeMagnitudeSpectrum(
        const float* audio_samples,
        size_t num_samples,
        float* magnitude_output,
        size_t num_bins
    ) const noexcept;

    // Extract formant frequencies (pitch-independent)
    struct FormantFrequencies {
        float f1_hz = 0.0f;  // First formant
        float f2_hz = 0.0f;  // Second formant
        float f3_hz = 0.0f;  // Third formant
        float bandwidth_f1 = 0.0f;
        float bandwidth_f2 = 0.0f;
        float bandwidth_f3 = 0.0f;
    };

    FormantFrequencies extractFormants(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    ) const noexcept;

    // Compute spectral centroid (brightness)
    float computeSpectralCentroid(
        const float* magnitude_spectrum,
        size_t num_bins,
        float sample_rate_hz
    ) const noexcept;

    // Compute spectral rolloff
    float computeSpectralRolloff(
        const float* magnitude_spectrum,
        size_t num_bins,
        float percentile = 0.85f
    ) const noexcept;

    // Compute spectral flux (rate of spectral change)
    float computeSpectralFlux(
        const float* current_magnitude,
        const float* previous_magnitude,
        size_t num_bins
    ) const noexcept;

private:
    // Pre-allocated FFT buffers
    mutable std::array<float, kFFTSize> fft_real_;
    mutable std::array<float, kFFTSize> fft_imag_;
    mutable std::array<float, kMaxSpectralBins> magnitude_buffer_;
    // Scratch buffer for JUCE interleaved FFT input/output (resized once in constructor)
    mutable std::vector<float> fft_scratch_;

    // JUCE FFT instance (optimized, RT-safe)
    // Using PIMPL pattern to avoid exposing JUCE headers in public API
    // Forward declaration - full definition in .cpp
    struct FFTImpl;
    std::unique_ptr<FFTImpl> fft_;

    // Compute FFT using JUCE's optimized implementation
    void computeFFT(const float* input, float* real, float* imag, size_t size) const noexcept;
};

} // namespace prrot
