#pragma once

#include "penta/kernels/Constants.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <random>

namespace penta::kernels {

// ============================================================================
// Bandlimited Oscillator (polyBLEP)
// ============================================================================

class BandlimitedOsc {
public:
    enum class Shape { Saw, Square, Triangle };

    void setShape(Shape s) noexcept { shape_ = s; }
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setFrequency(float hz) noexcept { inc_ = hz / static_cast<float>(sr_); }

    float process() noexcept {
        float out = 0.0f;
        float t = phase_;

        switch (shape_) {
            case Shape::Saw:
                out = 2.0f * t - 1.0f;
                out -= polyBlep(t);
                break;
            case Shape::Square:
                out = (t < 0.5f) ? 1.0f : -1.0f;
                out += polyBlep(t);
                out -= polyBlep(std::fmod(t + 0.5f, 1.0f));
                break;
            case Shape::Triangle:
                out = (t < 0.5f) ? 1.0f : -1.0f;
                out += polyBlep(t);
                out -= polyBlep(std::fmod(t + 0.5f, 1.0f));
                triState_ += 4.0f * inc_ * out;
                out = triState_;
                break;
        }

        phase_ += inc_;
        if (phase_ >= 1.0f) phase_ -= 1.0f;
        return out;
    }

    void reset() noexcept { phase_ = 0.0f; triState_ = 0.0f; }

private:
    float polyBlep(float t) const noexcept {
        if (t < inc_) {
            t /= inc_;
            return t + t - t * t - 1.0f;
        }
        if (t > 1.0f - inc_) {
            t = (t - 1.0f) / inc_;
            return t * t + t + t + 1.0f;
        }
        return 0.0f;
    }

    Shape shape_ = Shape::Saw;
    uint32_t sr_ = kDefaultSampleRate;
    float inc_ = 0.0f;
    float phase_ = 0.0f;
    float triState_ = 0.0f;
};

// ============================================================================
// Phase Distortion Oscillator
// ============================================================================

class PhaseDistortionOsc {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setFrequency(float hz) noexcept { inc_ = hz / static_cast<float>(sr_); }
    void setDistortion(float d) noexcept { dist_ = std::clamp(d, 0.0f, 0.99f); }

    float process() noexcept {
        float dp;
        if (phase_ < 0.5f) {
            dp = phase_ / (1.0f - dist_);
        } else {
            dp = 0.5f + (phase_ - 0.5f) / (1.0f + dist_);
        }
        dp = std::clamp(dp, 0.0f, 1.0f);
        float out = std::sin(kTwoPi * dp);

        phase_ += inc_;
        if (phase_ >= 1.0f) phase_ -= 1.0f;
        return out;
    }

    void reset() noexcept { phase_ = 0.0f; }

private:
    uint32_t sr_ = kDefaultSampleRate;
    float inc_ = 0.0f;
    float phase_ = 0.0f;
    float dist_ = 0.0f;
};

// ============================================================================
// FM Operator (sine carrier with modulation input)
// ============================================================================

class FMOperator {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setFrequency(float hz) noexcept { inc_ = hz / static_cast<float>(sr_); }
    void setModIndex(float idx) noexcept { modIndex_ = idx; }

    float process(float modInput = 0.0f) noexcept {
        float out = std::sin(kTwoPi * (phase_ + modIndex_ * modInput));
        phase_ += inc_;
        if (phase_ >= 1.0f) phase_ -= 1.0f;
        return out;
    }

    void reset() noexcept { phase_ = 0.0f; }

private:
    uint32_t sr_ = kDefaultSampleRate;
    float inc_ = 0.0f;
    float phase_ = 0.0f;
    float modIndex_ = 1.0f;
};

// ============================================================================
// Noise Generator (white / pink / brown)
// ============================================================================

class NoiseGenerator {
public:
    enum class Color { White, Pink, Brown };

    void setColor(Color c) noexcept { color_ = c; }

    float process() noexcept {
        float white = dist_(rng_);

        switch (color_) {
            case Color::White:
                return white;
            case Color::Pink:
                b0_ = 0.99886f * b0_ + white * 0.0555179f;
                b1_ = 0.99332f * b1_ + white * 0.0750759f;
                b2_ = 0.96900f * b2_ + white * 0.1538520f;
                b3_ = 0.86650f * b3_ + white * 0.3104856f;
                b4_ = 0.55000f * b4_ + white * 0.5329522f;
                b5_ = -0.7616f * b5_ - white * 0.0168980f;
                return (b0_ + b1_ + b2_ + b3_ + b4_ + b5_ + b6_ + white * 0.5362f) * 0.11f;
            case Color::Brown:
                brown_ = (brown_ + 0.02f * white) / 1.02f;
                return brown_ * 3.5f;
        }
        return white;
    }

    void reset() noexcept {
        b0_ = b1_ = b2_ = b3_ = b4_ = b5_ = b6_ = brown_ = 0.0f;
    }

private:
    Color color_ = Color::White;
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};
    float b0_ = 0, b1_ = 0, b2_ = 0, b3_ = 0, b4_ = 0, b5_ = 0, b6_ = 0;
    float brown_ = 0.0f;
};

} // namespace penta::kernels
