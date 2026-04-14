#pragma once

#include "penta/kernels/Constants.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace penta::kernels {

// ============================================================================
// Modulated Delay (chorus / flanger via LFO)
// ============================================================================

class ModulatedDelay {
public:
    void setSampleRate(uint32_t sr) noexcept {
        sr_ = sr;
        buffer_.assign(sr_ * 2, 0.0f); // 2 seconds max
    }
    void setBaseDelayMs(float ms) noexcept { baseDelay_ = ms; }
    void setDepthMs(float ms) noexcept { depth_ = ms; }
    void setRateHz(float hz) noexcept { lfoInc_ = hz / static_cast<float>(sr_); }
    void setFeedback(float fb) noexcept { feedback_ = std::clamp(fb, -0.99f, 0.99f); }
    void setMix(float m) noexcept { mix_ = std::clamp(m, 0.0f, 1.0f); }

    float process(float in) noexcept {
        if (buffer_.empty()) return in;

        float lfo = std::sin(kTwoPi * lfoPhase_);
        lfoPhase_ += lfoInc_;
        if (lfoPhase_ >= 1.0f) lfoPhase_ -= 1.0f;

        float delaySamples = (baseDelay_ + depth_ * lfo) * static_cast<float>(sr_) * 0.001f;
        delaySamples = std::clamp(delaySamples, 1.0f, static_cast<float>(buffer_.size() - 1));

        // Read with linear interpolation
        float readPos = static_cast<float>(writePos_) - delaySamples;
        if (readPos < 0.0f) readPos += static_cast<float>(buffer_.size());
        size_t i0 = static_cast<size_t>(readPos) % buffer_.size();
        size_t i1 = (i0 + 1) % buffer_.size();
        float frac = readPos - std::floor(readPos);
        float delayed = buffer_[i0] * (1.0f - frac) + buffer_[i1] * frac;

        buffer_[writePos_] = in + delayed * feedback_;
        writePos_ = (writePos_ + 1) % buffer_.size();

        return in * (1.0f - mix_) + delayed * mix_;
    }

    void reset() noexcept {
        std::fill(buffer_.begin(), buffer_.end(), 0.0f);
        writePos_ = 0;
        lfoPhase_ = 0.0f;
    }

private:
    uint32_t sr_ = kDefaultSampleRate;
    std::vector<float> buffer_;
    size_t writePos_ = 0;
    float baseDelay_ = 7.0f, depth_ = 3.0f;
    float lfoInc_ = 0.0f, lfoPhase_ = 0.0f;
    float feedback_ = 0.0f, mix_ = 0.5f;
};

// ============================================================================
// Tap Delay (up to 8 taps)
// ============================================================================

class TapDelay {
public:
    static constexpr size_t kMaxTaps = 8;

    struct Tap {
        float delayMs = 0.0f;
        float gain = 1.0f;
        float pan = 0.0f; // -1 left, +1 right (unused in mono)
    };

    void setSampleRate(uint32_t sr) noexcept {
        sr_ = sr;
        buffer_.assign(sr_ * 4, 0.0f); // 4 seconds max
    }
    void setTap(size_t idx, Tap t) noexcept { if (idx < kMaxTaps) taps_[idx] = t; }
    void setNumTaps(size_t n) noexcept { numTaps_ = std::min(n, kMaxTaps); }
    void setFeedback(float fb) noexcept { feedback_ = std::clamp(fb, -0.99f, 0.99f); }

    float process(float in) noexcept {
        if (buffer_.empty()) return in;

        float sum = 0.0f;
        for (size_t t = 0; t < numTaps_; ++t) {
            float delaySamples = taps_[t].delayMs * static_cast<float>(sr_) * 0.001f;
            float readPos = static_cast<float>(writePos_) - delaySamples;
            if (readPos < 0.0f) readPos += static_cast<float>(buffer_.size());
            size_t idx = static_cast<size_t>(readPos) % buffer_.size();
            sum += buffer_[idx] * taps_[t].gain;
        }

        buffer_[writePos_] = in + sum * feedback_;
        writePos_ = (writePos_ + 1) % buffer_.size();
        return in + sum;
    }

    void reset() noexcept {
        std::fill(buffer_.begin(), buffer_.end(), 0.0f);
        writePos_ = 0;
    }

private:
    uint32_t sr_ = kDefaultSampleRate;
    std::vector<float> buffer_;
    size_t writePos_ = 0;
    std::array<Tap, kMaxTaps> taps_{};
    size_t numTaps_ = 0;
    float feedback_ = 0.3f;
};

// ============================================================================
// Feedback Delay Network (4x4 Hadamard matrix)
// ============================================================================

class FeedbackDelayNetwork {
public:
    static constexpr size_t kNumLines = 4;

    void setSampleRate(uint32_t sr) noexcept {
        sr_ = sr;
        // Prime delay lengths for decorrelation
        constexpr float delayMs[kNumLines] = { 29.7f, 37.1f, 41.1f, 43.7f };
        for (size_t i = 0; i < kNumLines; ++i) {
            size_t len = static_cast<size_t>(sr * delayMs[i] * 0.001f);
            lines_[i].assign(len, 0.0f);
            writePos_[i] = 0;
        }
    }
    void setDecay(float seconds) noexcept {
        for (size_t i = 0; i < kNumLines; ++i) {
            if (!lines_[i].empty())
                feedback_[i] = std::pow(0.001f, static_cast<float>(lines_[i].size()) /
                    (seconds * static_cast<float>(sr_)));
        }
    }

    float process(float in) noexcept {
        std::array<float, kNumLines> delayed{};
        for (size_t i = 0; i < kNumLines; ++i) {
            if (lines_[i].empty()) continue;
            delayed[i] = lines_[i][writePos_[i]];
        }

        // Hadamard mixing
        std::array<float, kNumLines> mixed{};
        mixed[0] = ( delayed[0] + delayed[1] + delayed[2] + delayed[3]) * 0.5f;
        mixed[1] = ( delayed[0] - delayed[1] + delayed[2] - delayed[3]) * 0.5f;
        mixed[2] = ( delayed[0] + delayed[1] - delayed[2] - delayed[3]) * 0.5f;
        mixed[3] = ( delayed[0] - delayed[1] - delayed[2] + delayed[3]) * 0.5f;

        float out = 0.0f;
        for (size_t i = 0; i < kNumLines; ++i) {
            if (lines_[i].empty()) continue;
            lines_[i][writePos_[i]] = in * 0.5f + mixed[i] * feedback_[i];
            writePos_[i] = (writePos_[i] + 1) % lines_[i].size();
            out += delayed[i];
        }
        return out * 0.25f;
    }

    void reset() noexcept {
        for (size_t i = 0; i < kNumLines; ++i) {
            std::fill(lines_[i].begin(), lines_[i].end(), 0.0f);
            writePos_[i] = 0;
        }
    }

private:
    uint32_t sr_ = kDefaultSampleRate;
    std::array<std::vector<float>, kNumLines> lines_;
    std::array<size_t, kNumLines> writePos_{};
    std::array<float, kNumLines> feedback_{};
};

} // namespace penta::kernels
