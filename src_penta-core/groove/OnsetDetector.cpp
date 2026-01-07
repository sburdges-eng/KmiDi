#include "penta/groove/OnsetDetector.h"
#include "penta/common/SIMDKernels.h"
#include <juce_dsp/juce_dsp.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace penta::groove
{

    OnsetDetector::OnsetDetector(const Config &config)
        : config_(config), onsetDetected_(false), onsetStrength_(0.0f), onsetPosition_(0), lastOnsetPosition_(0), sampleCounter_(0)
    {
        // Initialize JUCE FFT (requires power-of-2 size)
        int fftOrder = static_cast<int>(std::log2(config_.fftSize));
        fft_ = std::make_unique<juce::dsp::FFT>(fftOrder);
        
        // Pre-allocate buffers
        // JUCE FFT uses interleaved complex format: [real0, imag0, real1, imag1, ...]
        fftBuffer_.resize(config_.fftSize * 2);  // Real + imag for each sample
        windowedBuffer_.resize(config_.fftSize);  // Pre-allocated buffer for windowed input (RT-safe)
        spectrum_.resize(config_.fftSize / 2 + 1);  // Magnitude spectrum
        prevSpectrum_.resize(config_.fftSize / 2 + 1);
        fluxHistory_.resize(100);

        // Initialize Hann window for spectral analysis
        window_.resize(config_.fftSize);
        for (size_t i = 0; i < config_.fftSize; ++i)
        {
            window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (config_.fftSize - 1)));
        }
    }

    OnsetDetector::~OnsetDetector() = default;

    void OnsetDetector::process(const float *buffer, size_t frames) noexcept
    {
        onsetDetected_ = false;

        // Process in hop-size chunks
        for (size_t i = 0; i + config_.hopSize <= frames; i += config_.hopSize)
        {
            computeSpectralFlux(buffer + i, config_.hopSize);
            detectPeaks();
            sampleCounter_ += config_.hopSize;
        }
    }

    void OnsetDetector::setThreshold(float threshold) noexcept
    {
        config_.threshold = threshold;
    }

    void OnsetDetector::reset() noexcept
    {
        onsetDetected_ = false;
        onsetStrength_ = 0.0f;
        onsetPosition_ = 0;
        lastOnsetPosition_ = 0;
        sampleCounter_ = 0;
        std::fill(prevSpectrum_.begin(), prevSpectrum_.end(), 0.0f);
        std::fill(fluxHistory_.begin(), fluxHistory_.end(), 0.0f);
    }

    void OnsetDetector::computeSpectralFlux(const float *buffer, size_t frames) noexcept
    {
        // FFT-based spectral flux detection using juce::dsp::FFT
        
        const size_t fftSize = config_.fftSize;
        const size_t numBins = spectrum_.size();
        
        // Zero-pad or truncate input to FFT size (RT-safe: using pre-allocated buffer)
        std::fill(windowedBuffer_.begin(), windowedBuffer_.end(), 0.0f);
        std::fill(fftBuffer_.begin(), fftBuffer_.end(), 0.0f);
        
        size_t copySize = std::min(frames, fftSize);
        
        // Copy input to pre-allocated windowed buffer
        for (size_t i = 0; i < copySize; ++i)
        {
            windowedBuffer_[i] = buffer[i];
        }
        
        // Apply window using SIMD-optimized kernel (in-place)
        SIMDKernels::applyWindow(windowedBuffer_.data(), window_.data(), fftSize);
        
        // Copy windowed buffer to FFT buffer (interleaved complex format)
        for (size_t i = 0; i < fftSize; ++i)
        {
            fftBuffer_[i * 2] = windowedBuffer_[i];      // Real part
            fftBuffer_[i * 2 + 1] = 0.0f;               // Imaginary part (zero for real input)
        }
        
        // Perform FFT (real-to-complex)
        fft_->performRealOnlyForwardTransform(fftBuffer_.data(), false);
        
        // Extract magnitude spectrum from FFT output
        // JUCE FFT output format: [DC, real1, imag1, real2, imag2, ..., Nyquist]
        
        // DC component (bin 0)
        spectrum_[0] = std::abs(fftBuffer_[0]);
        
        // Positive frequencies (bins 1 to Nyquist-1)
        for (size_t i = 1; i < numBins - 1; ++i)
        {
            float real = fftBuffer_[i * 2];
            float imag = fftBuffer_[i * 2 + 1];
            spectrum_[i] = std::sqrt(real * real + imag * imag);
        }
        
        // Nyquist frequency (last bin)
        if (numBins > 0)
        {
            spectrum_[numBins - 1] = std::abs(fftBuffer_[1]);
        }
        
        // Compute spectral flux using SIMD-optimized kernel
        float flux = SIMDKernels::spectralFlux(spectrum_.data(), prevSpectrum_.data(), numBins);
        
        // Normalize flux by number of bins
        if (numBins > 0)
        {
            flux /= static_cast<float>(numBins);
        }
        
        // Update flux history (rolling buffer)
        std::rotate(fluxHistory_.begin(), fluxHistory_.begin() + 1, fluxHistory_.end());
        fluxHistory_.back() = flux;
        
        // Store current spectrum for next frame
        prevSpectrum_ = spectrum_;
    }

    void OnsetDetector::detectPeaks() noexcept
    {
        if (fluxHistory_.size() < 3)
        {
            return;
        }

        // Get current flux value
        float currentFlux = fluxHistory_.back();

        // Calculate adaptive threshold from recent history
        float mean = 0.0f;
        size_t historyWindow = std::min(size_t(20), fluxHistory_.size() - 1);
        for (size_t i = fluxHistory_.size() - historyWindow; i < fluxHistory_.size() - 1; ++i)
        {
            mean += fluxHistory_[i];
        }
        mean /= historyWindow;

        float adaptiveThreshold = mean + config_.threshold;

        // Check if current flux is a peak above threshold
        if (currentFlux > adaptiveThreshold)
        {
            // Check previous and next values to confirm it's a local maximum
            float prevFlux = fluxHistory_[fluxHistory_.size() - 2];

            if (currentFlux > prevFlux)
            {
                // Check minimum time between onsets
                uint64_t timeSinceLastOnset = sampleCounter_ - lastOnsetPosition_;
                uint64_t minSamples = static_cast<uint64_t>(
                    config_.minTimeBetweenOnsets * config_.sampleRate);

                if (timeSinceLastOnset >= minSamples)
                {
                    onsetDetected_ = true;
                    onsetStrength_ = std::min(1.0f, currentFlux / (adaptiveThreshold + 0.1f));
                    onsetPosition_ = sampleCounter_;
                    lastOnsetPosition_ = sampleCounter_;
                }
            }
        }
    }

} // namespace penta::groove
