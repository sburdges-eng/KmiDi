#pragma once

#include "penta/kernels/Constants.h"
#include <array>
#include <cmath>
#include <algorithm>

namespace penta::kernels {

// ============================================================================
// Grain Window (Hann, Tukey, Gaussian)
// ============================================================================

class GrainWindow {
public:
    enum class Shape { Hann, Tukey, Gaussian };

    void setShape(Shape s) noexcept { shape_ = s; }
    void setTukeyAlpha(float a) noexcept { tukeyAlpha_ = std::clamp(a, 0.0f, 1.0f); }

    float apply(float phase) const noexcept {
        float t = std::clamp(phase, 0.0f, 1.0f);
        switch (shape_) {
            case Shape::Hann:
                return 0.5f * (1.0f - std::cos(kTwoPi * t));
            case Shape::Tukey: {
                if (t < tukeyAlpha_ / 2.0f)
                    return 0.5f * (1.0f + std::cos(kTwoPi / tukeyAlpha_ * (t - tukeyAlpha_ / 2.0f)));
                if (t > 1.0f - tukeyAlpha_ / 2.0f)
                    return 0.5f * (1.0f + std::cos(kTwoPi / tukeyAlpha_ * (t - 1.0f + tukeyAlpha_ / 2.0f)));
                return 1.0f;
            }
            case Shape::Gaussian: {
                float sigma = 0.4f;
                float x = (t - 0.5f) / sigma;
                return std::exp(-0.5f * x * x);
            }
        }
        return 1.0f;
    }

private:
    Shape shape_ = Shape::Hann;
    float tukeyAlpha_ = 0.5f;
};

// ============================================================================
// Grain Scheduler
// ============================================================================

class GrainScheduler {
public:
    struct GrainParams {
        size_t startSample = 0;
        size_t lengthSamples = 2048;
        float pitch = 1.0f;
        float amplitude = 1.0f;
    };

    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setDensity(float grainsPerSec) noexcept {
        interval_ = static_cast<size_t>(static_cast<float>(sr_) / std::max(grainsPerSec, 0.1f));
    }
    void setGrainLengthMs(float ms) noexcept {
        grainLength_ = static_cast<size_t>(static_cast<float>(sr_) * ms * 0.001f);
    }

    bool shouldFire() noexcept {
        if (++counter_ >= interval_) {
            counter_ = 0;
            return true;
        }
        return false;
    }

    GrainParams nextGrain(size_t sourcePos) const noexcept {
        return { sourcePos, grainLength_, 1.0f, 1.0f };
    }

    void reset() noexcept { counter_ = 0; }

private:
    uint32_t sr_ = kDefaultSampleRate;
    size_t interval_ = 4800;
    size_t grainLength_ = 2048;
    size_t counter_ = 0;
};

// ============================================================================
// Granular Engine (max 32 concurrent grains)
// ============================================================================

class GranularEngine {
public:
    static constexpr size_t kMaxGrains = 32;

    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; scheduler_.setSampleRate(sr); }
    void setSource(const float* data, size_t length) noexcept { source_ = data; sourceLen_ = length; }
    void setDensity(float gps) noexcept { scheduler_.setDensity(gps); }
    void setGrainLengthMs(float ms) noexcept { scheduler_.setGrainLengthMs(ms); }
    void setPosition(float normalized) noexcept { position_ = std::clamp(normalized, 0.0f, 1.0f); }

    float process() noexcept {
        if (!source_ || sourceLen_ == 0) return 0.0f;

        if (scheduler_.shouldFire()) {
            size_t pos = static_cast<size_t>(position_ * static_cast<float>(sourceLen_));
            auto params = scheduler_.nextGrain(pos);
            spawnGrain(params);
        }

        float sum = 0.0f;
        for (size_t i = 0; i < kMaxGrains; ++i) {
            if (!grains_[i].active) continue;
            auto& g = grains_[i];
            float phase = static_cast<float>(g.pos) / static_cast<float>(g.length);
            float win = window_.apply(phase);
            size_t srcIdx = g.start + g.pos;
            if (srcIdx < sourceLen_)
                sum += source_[srcIdx] * win * g.amplitude;
            if (++g.pos >= g.length) g.active = false;
        }
        return sum;
    }

    void reset() noexcept {
        for (auto& g : grains_) g.active = false;
        scheduler_.reset();
    }

private:
    struct Grain {
        size_t start = 0, length = 0, pos = 0;
        float amplitude = 1.0f;
        bool active = false;
    };

    void spawnGrain(const GrainScheduler::GrainParams& p) noexcept {
        for (auto& g : grains_) {
            if (!g.active) {
                g = { p.startSample, p.lengthSamples, 0, p.amplitude, true };
                return;
            }
        }
    }

    uint32_t sr_ = kDefaultSampleRate;
    const float* source_ = nullptr;
    size_t sourceLen_ = 0;
    float position_ = 0.0f;
    GrainScheduler scheduler_;
    GrainWindow window_;
    std::array<Grain, kMaxGrains> grains_{};
};

// ============================================================================
// Spectral Resynthesis (stub — requires FFT infrastructure)
// ============================================================================

class SpectralResynthesis {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setFFTSize(size_t size) noexcept { fftSize_ = size; }

    // Stub: returns 0. Full implementation requires FFT + phase accumulation.
    float process() noexcept { return 0.0f; }
    void reset() noexcept {}

private:
    uint32_t sr_ = kDefaultSampleRate;
    size_t fftSize_ = 2048;
};

} // namespace penta::kernels
