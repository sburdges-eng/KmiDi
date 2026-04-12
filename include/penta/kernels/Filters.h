#pragma once

#include "penta/kernels/Constants.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace penta::kernels {

// ============================================================================
// State Variable Filter (SVF) — Chamberlin topology
// ============================================================================

class SVFilter {
public:
    enum class Mode { Lowpass, Highpass, Bandpass, Notch };

    void setMode(Mode m) noexcept { mode_ = m; }
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; recalc(); }
    void setFrequency(float hz) noexcept { freq_ = hz; recalc(); }
    void setResonance(float q) noexcept { q_ = std::max(q, 0.5f); recalc(); }

    float process(float in) noexcept {
        lp_ += f_ * bp_;
        hp_ = in - lp_ - damp_ * bp_;
        bp_ += f_ * hp_;
        notch_ = hp_ + lp_;

        switch (mode_) {
            case Mode::Lowpass:  return lp_;
            case Mode::Highpass: return hp_;
            case Mode::Bandpass: return bp_;
            case Mode::Notch:    return notch_;
        }
        return lp_;
    }

    void reset() noexcept { lp_ = hp_ = bp_ = notch_ = 0.0f; }

private:
    void recalc() noexcept {
        f_ = 2.0f * std::sin(kPi * freq_ / static_cast<float>(sr_));
        damp_ = 1.0f / q_;
    }

    Mode mode_ = Mode::Lowpass;
    uint32_t sr_ = kDefaultSampleRate;
    float freq_ = 1000.0f;
    float q_ = 0.707f;
    float f_ = 0.0f;
    float damp_ = 0.0f;
    float lp_ = 0.0f, hp_ = 0.0f, bp_ = 0.0f, notch_ = 0.0f;
};

// ============================================================================
// Ladder Filter (Moog-style, 4-pole)
// ============================================================================

class LadderFilter {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; recalc(); }
    void setCutoff(float hz) noexcept { cutoff_ = hz; recalc(); }
    void setResonance(float r) noexcept { resonance_ = std::clamp(r, 0.0f, 1.0f); }

    float process(float in) noexcept {
        float input = in - resonance_ * 4.0f * stage_[3];
        for (int i = 0; i < 4; ++i) {
            float s = (i == 0) ? input : stage_[i - 1];
            stage_[i] += g_ * (std::tanh(s) - std::tanh(stage_[i]));
        }
        return stage_[3];
    }

    void reset() noexcept { stage_.fill(0.0f); }

private:
    void recalc() noexcept {
        g_ = 1.0f - std::exp(-kTwoPi * cutoff_ / static_cast<float>(sr_));
    }

    uint32_t sr_ = kDefaultSampleRate;
    float cutoff_ = 1000.0f;
    float resonance_ = 0.0f;
    float g_ = 0.0f;
    std::array<float, 4> stage_{};
};

// ============================================================================
// FIR Filter (fixed-order, coefficients set externally)
// ============================================================================

class FIRFilter {
public:
    void setCoefficients(const float* coeffs, size_t order) noexcept {
        order_ = std::min(order, kMaxOrder);
        for (size_t i = 0; i < order_; ++i) coeffs_[i] = coeffs[i];
    }

    float process(float in) noexcept {
        buffer_[writePos_] = in;
        float sum = 0.0f;
        size_t idx = writePos_;
        for (size_t i = 0; i < order_; ++i) {
            sum += coeffs_[i] * buffer_[idx];
            idx = (idx == 0) ? kMaxOrder - 1 : idx - 1;
        }
        writePos_ = (writePos_ + 1) % kMaxOrder;
        return sum;
    }

    void reset() noexcept {
        buffer_.fill(0.0f);
        writePos_ = 0;
    }

private:
    static constexpr size_t kMaxOrder = 512;
    std::array<float, kMaxOrder> coeffs_{};
    std::array<float, kMaxOrder> buffer_{};
    size_t order_ = 1;
    size_t writePos_ = 0;
};

// ============================================================================
// DC Blocker (single-pole highpass at ~5 Hz)
// ============================================================================

class DCBlocker {
public:
    void setSampleRate(uint32_t sr) noexcept {
        coeff_ = 1.0f - (kTwoPi * 5.0f / static_cast<float>(sr));
    }

    float process(float in) noexcept {
        float out = in - xPrev_ + coeff_ * yPrev_;
        xPrev_ = in;
        yPrev_ = out;
        return out;
    }

    void reset() noexcept { xPrev_ = yPrev_ = 0.0f; }

private:
    float coeff_ = 0.9993f;
    float xPrev_ = 0.0f;
    float yPrev_ = 0.0f;
};

// ============================================================================
// All-Pass Filter (first order)
// ============================================================================

class AllPassFilter {
public:
    void setCoefficient(float a) noexcept { a_ = a; }

    float process(float in) noexcept {
        float out = a_ * in + xPrev_ - a_ * yPrev_;
        xPrev_ = in;
        yPrev_ = out;
        return out;
    }

    void reset() noexcept { xPrev_ = yPrev_ = 0.0f; }

private:
    float a_ = 0.5f;
    float xPrev_ = 0.0f;
    float yPrev_ = 0.0f;
};

} // namespace penta::kernels
