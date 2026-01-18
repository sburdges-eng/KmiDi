#pragma once

#include "penta/common/RTTypes.h"
#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <memory>

namespace penta::groove
{

    /**
     * Real-time onset detection using spectral flux
     * Optimized for low-latency performance
     */
    class OnsetDetector
    {
    public:
        struct Config
        {
            double sampleRate;
            size_t fftSize;
            size_t hopSize;
            float threshold;
            float minTimeBetweenOnsets; // seconds

            Config()
                : sampleRate(kDefaultSampleRate), fftSize(2048), hopSize(512),
                  // Threshold is used as an offset above the recent mean flux.
                  // FFT-based spectral flux is normalized, so practical values are in the ~0.0-0.05 range.
                  threshold(0.01f), minTimeBetweenOnsets(0.05f)
            {
            }
        };

        explicit OnsetDetector(const Config &config = Config{});
        ~OnsetDetector();

        // RT-safe: Process audio and detect onsets
        void process(const float *buffer, size_t frames) noexcept;

        // RT-safe: Check if onset detected in last process call
        bool hasOnset() const noexcept { return onsetDetected_; }

        // RT-safe: Get onset strength (0.0-1.0)
        float getOnsetStrength() const noexcept { return onsetStrength_; }

        // RT-safe: Get position of last onset (samples since last reset)
        uint64_t getOnsetPosition() const noexcept { return onsetPosition_; }

        // Configuration
        void setThreshold(float threshold) noexcept;
        void reset() noexcept;

    private:
        void computeSpectralFlux(const float *buffer, size_t frames) noexcept;
        void detectPeaks() noexcept;

        Config config_;

        std::unique_ptr<juce::dsp::FFT> fft_;
        std::vector<float> window_;
        std::vector<float> windowedBuffer_;  // Pre-allocated buffer for windowed input (RT-safe)
        std::vector<float> fftBuffer_;  // Interleaved complex format for JUCE FFT
        std::vector<float> spectrum_;
        std::vector<float> prevSpectrum_;
        std::vector<float> fluxHistory_;

        bool onsetDetected_;
        float onsetStrength_;
        uint64_t onsetPosition_;
        uint64_t lastOnsetPosition_;
        uint64_t sampleCounter_;
        size_t fluxHistoryIndex_ = 0;
        size_t fluxHistoryCount_ = 0;
        float lastFlux_ = 0.0f;
    };

} // namespace penta::groove
