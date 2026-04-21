#include "ml/MelSpectrogram.h"
#include "penta/common/SIMDKernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace penta::ml {

namespace {

constexpr float kPI = 3.14159265358979323846f;
constexpr float kLogFloor = 1e-10f;

// Hz to mel (HTK formula)
float hzToMel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

// Mel to Hz
float melToHz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

} // anonymous namespace

MelSpectrogram::MelSpectrogram()
    : window_(kNFft)
    , windowed_(kNFft)
    , fftReal_(kNFft)
    , fftImag_(kNFft)
    , magnitudes_(kNFft / 2 + 1)
    , melFilterbank_(kNMels * (kNFft / 2 + 1), 0.0f)
    , melEnergies_(kNMels)
{
    // Hann window
    for (size_t i = 0; i < kNFft; ++i) {
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * kPI * static_cast<float>(i) /
                                               static_cast<float>(kNFft)));
    }
    buildMelFilterbank();
}

void MelSpectrogram::buildMelFilterbank() {
    const size_t nBins = kNFft / 2 + 1;
    const float fMax = static_cast<float>(kSampleRate) / 2.0f;
    const float melMin = hzToMel(0.0f);
    const float melMax = hzToMel(fMax);

    // kNMels + 2 equally spaced points in mel space
    std::vector<float> melPoints(kNMels + 2);
    for (size_t i = 0; i < kNMels + 2; ++i) {
        float mel = melMin + (melMax - melMin) * static_cast<float>(i) /
                    static_cast<float>(kNMels + 1);
        melPoints[i] = melToHz(mel);
    }

    // Convert Hz points to FFT bin frequencies
    std::vector<float> binFreqs(nBins);
    for (size_t i = 0; i < nBins; ++i) {
        binFreqs[i] = static_cast<float>(i) * static_cast<float>(kSampleRate) /
                      static_cast<float>(kNFft);
    }

    // Build triangular filters
    for (size_t m = 0; m < kNMels; ++m) {
        float fLeft   = melPoints[m];
        float fCenter = melPoints[m + 1];
        float fRight  = melPoints[m + 2];

        for (size_t k = 0; k < nBins; ++k) {
            float f = binFreqs[k];
            float weight = 0.0f;

            if (f >= fLeft && f <= fCenter && fCenter > fLeft) {
                weight = (f - fLeft) / (fCenter - fLeft);
            } else if (f > fCenter && f <= fRight && fRight > fCenter) {
                weight = (fRight - f) / (fRight - fCenter);
            }

            melFilterbank_[m * nBins + k] = weight;
        }
    }
}

void MelSpectrogram::applyWindow(const float* frame, float* windowed) {
    for (size_t i = 0; i < kNFft; ++i) {
        windowed[i] = frame[i] * window_[i];
    }
}

void MelSpectrogram::computeFFT(const float* windowed, float* magnitudes) {
    // Simple DFT for correctness. For production, replace with vDSP_fft or pffft.
    // Only compute kNFft/2+1 bins (real input symmetry).
    const size_t nBins = kNFft / 2 + 1;

    for (size_t k = 0; k < nBins; ++k) {
        float re = 0.0f;
        float im = 0.0f;
        const float phase_step = -2.0f * kPI * static_cast<float>(k) / static_cast<float>(kNFft);

        for (size_t n = 0; n < kNFft; ++n) {
            float phase = phase_step * static_cast<float>(n);
            re += windowed[n] * std::cos(phase);
            im += windowed[n] * std::sin(phase);
        }
        // Power spectrum
        magnitudes[k] = re * re + im * im;
    }
}

bool MelSpectrogram::compute(const float* samples, size_t count, float* output) {
    if (count < kRequiredSamples || !samples || !output) {
        return false;
    }

    const size_t nBins = kNFft / 2 + 1;

    for (size_t frame = 0; frame < kNFrames; ++frame) {
        const float* frameStart = samples + frame * kHopLength;

        applyWindow(frameStart, windowed_.data());
        computeFFT(windowed_.data(), magnitudes_.data());

        // Apply mel filterbank — inner dot product accelerated via SIMD.
        for (size_t m = 0; m < kNMels; ++m) {
            const float* filter = &melFilterbank_[m * nBins];
            float energy = penta::simd::dot_product_f32(filter, magnitudes_.data(), nBins);
            // Log-mel
            output[m * kNFrames + frame] = std::log(std::max(energy, kLogFloor));
        }
    }

    return true;
}

} // namespace penta::ml
