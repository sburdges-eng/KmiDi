#pragma once

#include "penta/kernels/Constants.h"
#include "penta/kernels/Filters.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace penta::kernels {

// ============================================================================
// Schroeder Reverb (4 comb + 2 allpass)
// ============================================================================

class SchroederReverb {
public:
    void setSampleRate(uint32_t sr) noexcept {
        sr_ = sr;
        constexpr float combMs[4] = { 29.7f, 37.1f, 41.1f, 43.7f };
        constexpr float apMs[2]   = { 5.0f, 1.7f };
        for (int i = 0; i < 4; ++i) {
            size_t len = static_cast<size_t>(sr * combMs[i] * 0.001f);
            combBuf_[i].assign(len, 0.0f);
            combPos_[i] = 0;
        }
        for (int i = 0; i < 2; ++i) {
            size_t len = static_cast<size_t>(sr * apMs[i] * 0.001f);
            apBuf_[i].assign(len, 0.0f);
            apPos_[i] = 0;
        }
    }

    void setDecay(float seconds) noexcept {
        for (int i = 0; i < 4; ++i) {
            if (!combBuf_[i].empty())
                combFb_[i] = std::pow(0.001f, static_cast<float>(combBuf_[i].size()) /
                    (seconds * static_cast<float>(sr_)));
        }
    }

    void setMix(float m) noexcept { mix_ = std::clamp(m, 0.0f, 1.0f); }

    float process(float in) noexcept {
        float combSum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            if (combBuf_[i].empty()) continue;
            float delayed = combBuf_[i][combPos_[i]];
            combBuf_[i][combPos_[i]] = in + delayed * combFb_[i];
            combPos_[i] = (combPos_[i] + 1) % combBuf_[i].size();
            combSum += delayed;
        }
        combSum *= 0.25f;

        // Allpass chain
        for (int i = 0; i < 2; ++i) {
            if (apBuf_[i].empty()) continue;
            float delayed = apBuf_[i][apPos_[i]];
            float temp = combSum + delayed * 0.5f;
            apBuf_[i][apPos_[i]] = temp;
            combSum = delayed - temp * 0.5f;
            apPos_[i] = (apPos_[i] + 1) % apBuf_[i].size();
        }

        return in * (1.0f - mix_) + combSum * mix_;
    }

    void reset() noexcept {
        for (auto& b : combBuf_) std::fill(b.begin(), b.end(), 0.0f);
        for (auto& b : apBuf_) std::fill(b.begin(), b.end(), 0.0f);
        combPos_.fill(0); apPos_.fill(0);
    }

private:
    uint32_t sr_ = kDefaultSampleRate;
    std::array<std::vector<float>, 4> combBuf_;
    std::array<size_t, 4> combPos_{};
    std::array<float, 4> combFb_{};
    std::array<std::vector<float>, 2> apBuf_;
    std::array<size_t, 2> apPos_{};
    float mix_ = 0.3f;
};

// ============================================================================
// Plate Reverb (stub — simplified diffusion network)
// ============================================================================

class PlateReverb {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; fdn_.setSampleRate(sr); }
    void setDecay(float seconds) noexcept { fdn_.setDecay(seconds); }
    void setDamping(float d) noexcept { damping_ = std::clamp(d, 0.0f, 1.0f); }
    void setMix(float m) noexcept { mix_ = std::clamp(m, 0.0f, 1.0f); }

    float process(float in) noexcept {
        float wet = fdn_.process(in);
        // Simple damping via one-pole
        dampState_ += (wet - dampState_) * (1.0f - damping_);
        return in * (1.0f - mix_) + dampState_ * mix_;
    }

    void reset() noexcept { fdn_.reset(); dampState_ = 0.0f; }

private:
    uint32_t sr_ = kDefaultSampleRate;
    FeedbackDelayNetwork fdn_;
    float damping_ = 0.5f;
    float mix_ = 0.3f;
    float dampState_ = 0.0f;
};

// ============================================================================
// Partitioned Convolution (stub — interface only)
// ============================================================================

class PartitionedConvolution {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setPartitionSize(size_t size) noexcept { partSize_ = size; }

    // Stub: requires FFT. setIR + process are no-ops.
    void setIR(const float* ir, size_t length) noexcept {
        (void)ir; irLength_ = length;
    }

    float process(float in) noexcept {
        (void)in;
        return 0.0f; // Requires FFT-based overlap-save
    }

    void reset() noexcept {}

private:
    uint32_t sr_ = kDefaultSampleRate;
    size_t partSize_ = 512;
    size_t irLength_ = 0;
};

} // namespace penta::kernels
