#pragma once

#include "penta/kernels/Constants.h"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace penta::kernels {

// ============================================================================
// Crossfade (linear and equal-power)
// ============================================================================

class Crossfade {
public:
    enum class Law { Linear, EqualPower };

    void setLaw(Law law) noexcept { law_ = law; }

    void process(const float* a, const float* b, float* out,
                 float mix, size_t frames) noexcept {
        mix = std::clamp(mix, 0.0f, 1.0f);
        float gA, gB;
        if (law_ == Law::EqualPower) {
            gA = std::cos(mix * kHalfPi);
            gB = std::sin(mix * kHalfPi);
        } else {
            gA = 1.0f - mix;
            gB = mix;
        }
        for (size_t i = 0; i < frames; ++i)
            out[i] = a[i] * gA + b[i] * gB;
    }

private:
    Law law_ = Law::Linear;
};

// ============================================================================
// Invert Polarity
// ============================================================================

class InvertPolarity {
public:
    void process(const float* in, float* out, size_t frames) noexcept {
        for (size_t i = 0; i < frames; ++i)
            out[i] = -in[i];
    }
};

// ============================================================================
// Volume Ramp (linear / exponential smoothing)
// ============================================================================

class VolumeRamp {
public:
    enum class Shape { Linear, Exponential };

    void setShape(Shape s) noexcept { shape_ = s; }
    void setSampleRate(uint32_t sr) noexcept { sampleRate_ = sr; recalc(); }
    void setRampTimeMs(float ms) noexcept { rampMs_ = ms; recalc(); }
    void setTarget(float target) noexcept { target_ = target; }

    void process(float* buf, size_t frames) noexcept {
        for (size_t i = 0; i < frames; ++i) {
            if (shape_ == Shape::Exponential) {
                current_ += (target_ - current_) * coeff_;
            } else {
                float step = (target_ - current_) * coeff_;
                current_ += step;
            }
            buf[i] *= current_;
        }
    }

    void reset() noexcept { current_ = 0.0f; }

private:
    void recalc() noexcept {
        if (sampleRate_ > 0 && rampMs_ > 0.0f) {
            float samples = sampleRate_ * rampMs_ * 0.001f;
            coeff_ = 1.0f / std::max(samples, 1.0f);
        }
    }

    Shape shape_ = Shape::Linear;
    uint32_t sampleRate_ = kDefaultSampleRate;
    float rampMs_ = 10.0f;
    float coeff_ = 0.002f;
    float current_ = 0.0f;
    float target_ = 1.0f;
};

} // namespace penta::kernels
