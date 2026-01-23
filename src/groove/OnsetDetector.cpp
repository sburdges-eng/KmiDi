#include "penta/groove/OnsetDetector.h"
#include <algorithm>
#include <cmath>

namespace {
size_t nextPowerOfTwo(size_t value) {
    if (value < 2) {
        return 2;
    }
    size_t power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

bool isPowerOfTwo(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}
}  // namespace

namespace penta::groove {

OnsetDetector::OnsetDetector(const Config& config)
    : config_(config)
    , onsetDetected_(false)
    , onsetStrength_(0.0f)
    , onsetPosition_(0)
    , lastOnsetPosition_(0)
    , sampleCounter_(0)
{
    if (config_.fftSize > 0 && !isPowerOfTwo(config_.fftSize)) {
        config_.fftSize = nextPowerOfTwo(config_.fftSize);
    }

    // Pre-allocate buffers
    window_.resize(config_.fftSize);
    windowedBuffer_.resize(config_.fftSize);
    fftBuffer_.resize(config_.fftSize * 2);
    spectrum_.resize(config_.fftSize / 2 + 1);
    prevSpectrum_.resize(config_.fftSize / 2 + 1);
    fluxHistory_.resize(100);

    // Initialize Hann window
    const double denom = static_cast<double>(config_.fftSize - 1);
    for (size_t i = 0; i < config_.fftSize; ++i) {
        window_[i] = 0.5f - 0.5f * std::cos((2.0 * juce::MathConstants<double>::pi * i) / denom);
    }

    // Setup FFT (requires power-of-two size)
    size_t fftSize = config_.fftSize;
    int order = 0;
    while ((1u << order) < fftSize) {
        ++order;
    }
    fft_ = std::make_unique<juce::dsp::FFT>(order);
}

OnsetDetector::~OnsetDetector() = default;

void OnsetDetector::process(const float* buffer, size_t frames) noexcept {
    if (!buffer || frames == 0 || config_.fftSize == 0) {
        onsetDetected_ = false;
        return;
    }

    onsetDetected_ = false;
    computeSpectralFlux(buffer, frames);
    detectPeaks();

    sampleCounter_ += frames;
}

void OnsetDetector::setThreshold(float threshold) noexcept {
    config_.threshold = threshold;
}

void OnsetDetector::reset() noexcept {
    onsetDetected_ = false;
    onsetStrength_ = 0.0f;
    onsetPosition_ = 0;
    lastOnsetPosition_ = 0;
    sampleCounter_ = 0;
    std::fill(prevSpectrum_.begin(), prevSpectrum_.end(), 0.0f);
    std::fill(fluxHistory_.begin(), fluxHistory_.end(), 0.0f);
    fluxHistoryIndex_ = 0;
    fluxHistoryCount_ = 0;
    lastFlux_ = 0.0f;
}

void OnsetDetector::computeSpectralFlux(const float* buffer, size_t frames) noexcept {
    if (!fft_) {
        return;
    }

    const size_t fftSize = config_.fftSize;
    const float* readPtr = buffer;

    if (frames >= fftSize) {
        readPtr = buffer + (frames - fftSize);
        for (size_t i = 0; i < fftSize; ++i) {
            windowedBuffer_[i] = readPtr[i] * window_[i];
        }
    } else {
        const size_t pad = fftSize - frames;
        std::fill(windowedBuffer_.begin(), windowedBuffer_.begin() + pad, 0.0f);
        for (size_t i = 0; i < frames; ++i) {
            windowedBuffer_[pad + i] = buffer[i] * window_[pad + i];
        }
    }

    std::fill(fftBuffer_.begin(), fftBuffer_.end(), 0.0f);
    for (size_t i = 0; i < fftSize; ++i) {
        fftBuffer_[i] = windowedBuffer_[i];
    }

    fft_->performRealOnlyForwardTransform(fftBuffer_.data());

    float flux = 0.0f;
    const size_t bins = spectrum_.size();
    for (size_t i = 0; i < bins; ++i) {
        const float re = fftBuffer_[2 * i];
        const float im = fftBuffer_[2 * i + 1];
        const float mag = std::sqrt(re * re + im * im);
        spectrum_[i] = mag;
        const float diff = mag - prevSpectrum_[i];
        if (diff > 0.0f) {
            flux += diff;
        }
        prevSpectrum_[i] = mag;
    }

    lastFlux_ = flux / static_cast<float>(bins);
    fluxHistory_[fluxHistoryIndex_] = lastFlux_;
    fluxHistoryIndex_ = (fluxHistoryIndex_ + 1) % fluxHistory_.size();
    fluxHistoryCount_ = std::min(fluxHistoryCount_ + 1, fluxHistory_.size());
}

void OnsetDetector::detectPeaks() noexcept {
    if (fluxHistoryCount_ == 0) {
        onsetDetected_ = false;
        onsetStrength_ = 0.0f;
        return;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < fluxHistoryCount_; ++i) {
        sum += fluxHistory_[i];
    }
    const float mean = sum / static_cast<float>(fluxHistoryCount_);
    const float threshold = mean + config_.threshold;

    const uint64_t minSamplesBetween = static_cast<uint64_t>(
        config_.minTimeBetweenOnsets * config_.sampleRate);

    if (lastFlux_ > threshold &&
        (sampleCounter_ >= lastOnsetPosition_ + minSamplesBetween)) {
        onsetDetected_ = true;
        onsetPosition_ = sampleCounter_;
        lastOnsetPosition_ = onsetPosition_;
        onsetStrength_ = juce::jlimit(
            0.0f,
            1.0f,
            (lastFlux_ - threshold) / (threshold + 1.0e-6f));
    } else {
        onsetDetected_ = false;
        onsetStrength_ = 0.0f;
    }
}

} // namespace penta::groove
