#pragma once

#include "penta/common/Platform.h"
#include <cmath>
#include <cstdint>

namespace penta::kernels {

constexpr float kPi      = 3.14159265358979323846f;
constexpr float kTwoPi   = 2.0f * kPi;
constexpr float kHalfPi  = kPi / 2.0f;
constexpr float kSqrt2   = 1.41421356237309504880f;
constexpr float kInvSqrt2 = 0.70710678118654752440f;
constexpr float kMinDb   = -144.0f;

constexpr uint32_t kDefaultSampleRate = 48000;

inline float linearToDb(float linear) noexcept {
    if (linear <= 0.0f) return kMinDb;
    return 20.0f * std::log10(linear);
}

inline float dbToLinear(float db) noexcept {
    return std::pow(10.0f, db / 20.0f);
}

inline float midiToFreq(float note) noexcept {
    return 440.0f * std::pow(2.0f, (note - 69.0f) / 12.0f);
}

} // namespace penta::kernels
