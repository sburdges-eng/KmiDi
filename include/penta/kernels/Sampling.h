#pragma once

#include "penta/kernels/Constants.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace penta::kernels {

// ============================================================================
// Sample Player (mono, one-shot / looping)
// ============================================================================

class SamplePlayer {
public:
    void setBuffer(const float* data, size_t length) noexcept { data_ = data; length_ = length; }
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setPlaybackRate(float rate) noexcept { rate_ = rate; }
    void setLoop(bool loop) noexcept { loop_ = loop; }

    void trigger() noexcept { pos_ = 0.0; active_ = true; }
    void stop() noexcept { active_ = false; }

    float process() noexcept {
        if (!active_ || !data_ || length_ == 0) return 0.0f;

        size_t idx = static_cast<size_t>(pos_);
        float frac = static_cast<float>(pos_ - static_cast<double>(idx));
        size_t next = idx + 1;

        if (next >= length_) {
            if (loop_) { next = 0; } else { active_ = false; return 0.0f; }
        }

        float out = data_[idx] * (1.0f - frac) + data_[next] * frac;

        pos_ += static_cast<double>(rate_);
        if (pos_ >= static_cast<double>(length_)) {
            if (loop_) pos_ -= static_cast<double>(length_);
            else active_ = false;
        }
        return out;
    }

    void reset() noexcept { pos_ = 0.0; active_ = false; }

private:
    const float* data_ = nullptr;
    size_t length_ = 0;
    uint32_t sr_ = kDefaultSampleRate;
    float rate_ = 1.0f;
    bool loop_ = false;
    double pos_ = 0.0;
    bool active_ = false;
};

// ============================================================================
// Multi-Sample Map (up to 128 zones, note + velocity keyed)
// ============================================================================

class MultiSampleMap {
public:
    struct Zone {
        uint8_t noteMin = 0, noteMax = 127;
        uint8_t velMin = 0, velMax = 127;
        const float* data = nullptr;
        size_t length = 0;
        float rootNote = 60.0f;
    };

    void addZone(const Zone& z) noexcept {
        if (count_ < kMaxZones) zones_[count_++] = z;
    }

    const Zone* findZone(uint8_t note, uint8_t velocity) const noexcept {
        for (size_t i = 0; i < count_; ++i) {
            const auto& z = zones_[i];
            if (note >= z.noteMin && note <= z.noteMax &&
                velocity >= z.velMin && velocity <= z.velMax)
                return &z;
        }
        return nullptr;
    }

    void reset() noexcept { count_ = 0; }

private:
    static constexpr size_t kMaxZones = 128;
    std::array<Zone, kMaxZones> zones_{};
    size_t count_ = 0;
};

// ============================================================================
// Resampler (linear / cubic interpolation)
// ============================================================================

class Resampler {
public:
    enum class Quality { Linear, Cubic };

    void setQuality(Quality q) noexcept { quality_ = q; }

    float process(const float* buf, size_t len, double pos) const noexcept {
        if (!buf || len == 0) return 0.0f;
        size_t i = static_cast<size_t>(pos);
        float f = static_cast<float>(pos - static_cast<double>(i));

        if (quality_ == Quality::Cubic && i >= 1 && i + 2 < len) {
            float y0 = buf[i - 1], y1 = buf[i], y2 = buf[i + 1], y3 = buf[i + 2];
            float a = y3 - y2 - y0 + y1;
            float b = y0 - y1 - a;
            float c = y2 - y0;
            return ((a * f + b) * f + c) * f + y1;
        }

        size_t next = (i + 1 < len) ? i + 1 : i;
        return buf[i] * (1.0f - f) + buf[next] * f;
    }

private:
    Quality quality_ = Quality::Linear;
};

// ============================================================================
// Time Stretch (simple granular overlap-add stub)
// ============================================================================

class TimeStretch {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setStretchFactor(float factor) noexcept { stretch_ = std::max(factor, 0.1f); }
    void setGrainSizeMs(float ms) noexcept { grainMs_ = ms; }

    // Stub: returns input unmodified. Full implementation requires grain scheduling.
    float process(float in) noexcept {
        (void)stretch_; (void)grainMs_;
        return in;
    }

    void reset() noexcept {}

private:
    uint32_t sr_ = kDefaultSampleRate;
    float stretch_ = 1.0f;
    float grainMs_ = 50.0f;
};

// ============================================================================
// Pitch Shifter (stub — requires overlap-add or phase vocoder)
// ============================================================================

class PitchShifter {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setSemitones(float st) noexcept { semitones_ = st; }

    // Stub: returns input unmodified.
    float process(float in) noexcept {
        (void)semitones_;
        return in;
    }

    void reset() noexcept {}

private:
    uint32_t sr_ = kDefaultSampleRate;
    float semitones_ = 0.0f;
};

} // namespace penta::kernels
