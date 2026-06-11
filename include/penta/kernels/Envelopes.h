#pragma once

#include "penta/kernels/Constants.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace penta::kernels {

// ============================================================================
// AHDSR Envelope (Attack-Hold-Decay-Sustain-Release)
// ============================================================================

class AHDSREnvelope {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; recalc(); }
    void setAttackMs(float ms) noexcept { attackMs_ = ms; recalc(); }
    void setHoldMs(float ms) noexcept { holdMs_ = ms; recalc(); }
    void setDecayMs(float ms) noexcept { decayMs_ = ms; recalc(); }
    void setSustain(float s) noexcept { sustain_ = std::clamp(s, 0.0f, 1.0f); }
    void setReleaseMs(float ms) noexcept { releaseMs_ = ms; recalc(); }

    void noteOn() noexcept { stage_ = Stage::Attack; }
    void noteOff() noexcept { stage_ = Stage::Release; }

    float process() noexcept {
        switch (stage_) {
            case Stage::Idle:
                return 0.0f;
            case Stage::Attack:
                value_ += attackRate_;
                if (value_ >= 1.0f) { value_ = 1.0f; stage_ = Stage::Hold; holdCounter_ = holdSamples_; }
                return value_;
            case Stage::Hold:
                if (--holdCounter_ <= 0) stage_ = Stage::Decay;
                return 1.0f;
            case Stage::Decay:
                value_ += (sustain_ - value_) * decayCoeff_;
                if (value_ <= sustain_ + 0.0001f) { value_ = sustain_; stage_ = Stage::Sustain; }
                return value_;
            case Stage::Sustain:
                return sustain_;
            case Stage::Release:
                value_ *= releaseCoeff_;
                if (value_ < 0.0001f) { value_ = 0.0f; stage_ = Stage::Idle; }
                return value_;
        }
        return 0.0f;
    }

    void reset() noexcept { value_ = 0.0f; stage_ = Stage::Idle; }

private:
    enum class Stage { Idle, Attack, Hold, Decay, Sustain, Release };

    void recalc() noexcept {
        float srF = static_cast<float>(sr_);
        attackRate_ = (attackMs_ > 0.0f) ? 1.0f / (srF * attackMs_ * 0.001f) : 1.0f;
        holdSamples_ = static_cast<int>(srF * holdMs_ * 0.001f);
        decayCoeff_ = (decayMs_ > 0.0f) ? 1.0f - std::exp(-1.0f / (srF * decayMs_ * 0.001f)) : 1.0f;
        releaseCoeff_ = (releaseMs_ > 0.0f) ? std::exp(-1.0f / (srF * releaseMs_ * 0.001f)) : 0.0f;
    }

    uint32_t sr_ = kDefaultSampleRate;
    float attackMs_ = 10.0f, holdMs_ = 0.0f, decayMs_ = 100.0f;
    float sustain_ = 0.7f, releaseMs_ = 200.0f;
    float attackRate_ = 0.0f, decayCoeff_ = 0.0f, releaseCoeff_ = 0.0f;
    int holdSamples_ = 0, holdCounter_ = 0;
    float value_ = 0.0f;
    Stage stage_ = Stage::Idle;
};

// ============================================================================
// Multi-Stage Envelope (MSEG) — up to 16 breakpoints
// ============================================================================

class MSEGEnvelope {
public:
    static constexpr size_t kMaxPoints = 16;

    struct Point {
        float time;
        float level;
    };

    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }

    void setPoints(const Point* pts, size_t count) noexcept {
        count_ = std::min(count, kMaxPoints);
        for (size_t i = 0; i < count_; ++i) points_[i] = pts[i];
    }

    void trigger() noexcept { segment_ = 0; sampleInSegment_ = 0; value_ = 0.0f; active_ = true; }

    float process() noexcept {
        if (!active_ || segment_ >= count_) return value_;

        float segSamples = points_[segment_].time * static_cast<float>(sr_);
        float target = points_[segment_].level;
        float prev = (segment_ > 0) ? points_[segment_ - 1].level : 0.0f;

        if (segSamples > 0.0f) {
            float t = static_cast<float>(sampleInSegment_) / segSamples;
            value_ = prev + (target - prev) * std::clamp(t, 0.0f, 1.0f);
        } else {
            value_ = target;
        }

        if (++sampleInSegment_ >= static_cast<size_t>(segSamples)) {
            value_ = target;
            sampleInSegment_ = 0;
            if (++segment_ >= count_) active_ = false;
        }
        return value_;
    }

    void reset() noexcept { segment_ = 0; sampleInSegment_ = 0; value_ = 0.0f; active_ = false; }

private:
    uint32_t sr_ = kDefaultSampleRate;
    std::array<Point, kMaxPoints> points_{};
    size_t count_ = 0;
    size_t segment_ = 0;
    size_t sampleInSegment_ = 0;
    float value_ = 0.0f;
    bool active_ = false;
};

// ============================================================================
// Sample and Hold
// ============================================================================

class SampleAndHold {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setRateHz(float hz) noexcept {
        interval_ = static_cast<size_t>(static_cast<float>(sr_) / std::max(hz, 0.1f));
    }

    float process(float in) noexcept {
        if (++counter_ >= interval_) {
            held_ = in;
            counter_ = 0;
        }
        return held_;
    }

    void reset() noexcept { held_ = 0.0f; counter_ = 0; }

private:
    uint32_t sr_ = kDefaultSampleRate;
    size_t interval_ = 4800;
    size_t counter_ = 0;
    float held_ = 0.0f;
};

// ============================================================================
// Slew Limiter
// ============================================================================

class SlewLimiter {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; recalc(); }
    void setRiseMs(float ms) noexcept { riseMs_ = ms; recalc(); }
    void setFallMs(float ms) noexcept { fallMs_ = ms; recalc(); }

    float process(float in) noexcept {
        float diff = in - value_;
        if (diff > riseRate_) diff = riseRate_;
        else if (diff < -fallRate_) diff = -fallRate_;
        value_ += diff;
        return value_;
    }

    void reset() noexcept { value_ = 0.0f; }

private:
    void recalc() noexcept {
        float srF = static_cast<float>(sr_);
        riseRate_ = (riseMs_ > 0.0f) ? 1.0f / (srF * riseMs_ * 0.001f) : 1e6f;
        fallRate_ = (fallMs_ > 0.0f) ? 1.0f / (srF * fallMs_ * 0.001f) : 1e6f;
    }

    uint32_t sr_ = kDefaultSampleRate;
    float riseMs_ = 1.0f, fallMs_ = 1.0f;
    float riseRate_ = 0.0f, fallRate_ = 0.0f;
    float value_ = 0.0f;
};

} // namespace penta::kernels
