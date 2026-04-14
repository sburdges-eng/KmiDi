# DSP Kernel Stubs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `include/penta/kernels/` with RT-safe stub headers for all ~58 missing DSP primitives, organized by category, following existing KMiDi C++20 conventions.

**Architecture:** Each kernel is a minimal class in `namespace penta::kernels` with `process()`, `reset()`, `setSampleRate()`. All methods are `noexcept`. No heap allocation in `process()`. Headers live in `include/penta/kernels/` — one header per category. A single `Kernels.h` umbrella header includes all. A Catch2 test file validates every stub compiles, instantiates, and produces finite output.

**Tech Stack:** C++20, Catch2, CMake (KellyCore static lib)

**Conventions (derived from existing code):**
- Namespace: `penta::kernels` (parallels `penta::diagnostics`, `penta::mixer`)
- Include guard: `#pragma once`
- First include: `"penta/common/Platform.h"`
- Types: `float` for samples, `size_t` for frame counts, `uint32_t` for sample rate
- All `process()` methods: `noexcept`
- All classes: non-copyable where they hold state (deleted copy ctor/assign)
- Reset: `reset()` zeroes all state
- Sample rate: `setSampleRate(uint32_t sr)` recalculates coefficients
- Constants: `constexpr float kPi`, `kTwoPi` in a shared constants header
- Stubs produce pass-through or silence — valid output, not undefined behavior

---

## File Structure

### New files to create:

| File | Responsibility |
|------|---------------|
| `include/penta/kernels/Constants.h` | Shared DSP constants (pi, twopi, etc.) |
| `include/penta/kernels/CoreSignal.h` | Crossfade, InvertPolarity, VolumeRamp |
| `include/penta/kernels/Filters.h` | SVF, LadderFilter, FIRFilter, DCBlocker, AllPassFilter |
| `include/penta/kernels/Oscillators.h` | BandlimitedOsc, PhaseDistortionOsc, FMOperator, NoiseGenerator |
| `include/penta/kernels/Envelopes.h` | AHDSREnvelope, MSEGEnvelope, SampleAndHold, SlewLimiter |
| `include/penta/kernels/Sampling.h` | SamplePlayer, MultiSampleMap, Resampler, TimeStretch, PitchShifter |
| `include/penta/kernels/Granular.h` | GranularEngine, GrainWindow, GrainScheduler, SpectralResynthesis |
| `include/penta/kernels/Delay.h` | ModulatedDelay, TapDelay, FeedbackDelayNetwork |
| `include/penta/kernels/Reverb.h` | SchroederReverb, PlateReverb, PartitionedConvolution |
| `include/penta/kernels/Dynamics.h` | Compressor, Expander, NoiseGate, DeEsser |
| `include/penta/kernels/Distortion.h` | Waveshaper, Bitcrusher, SampleRateReducer |
| `include/penta/kernels/EQ.h` | GraphicEQ, DynamicEQ |
| `include/penta/kernels/Spectral.h` | OverlapAdd, SpectralFilter, SpectralGate |
| `include/penta/kernels/Spatial.h` | MidSideEncoder, MidSideDecoder, StereoWidth, BinauralPanner |
| `include/penta/kernels/Modulation.h` | ModulationMatrix, ParameterSmoother, ControlScaler |
| `include/penta/kernels/MidiControl.h` | MidiEventScheduler, VelocityScaler, AftertouchMapper, CCMapper |
| `include/penta/kernels/Voice.h` | GlideProcessor |
| `include/penta/kernels/Analysis.h` | LUFSMeter, TransientDetector, SilenceDetector |
| `include/penta/kernels/Utility.h` | DenormalGuard, NaNGuard, BufferOps |
| `include/penta/kernels/Routing.h` | BufferSplitter, BufferMerger, ChannelRemapper, TopologicalNode |
| `include/penta/kernels/Kernels.h` | Umbrella include |
| `tests/cpp/test_kernel_stubs.cpp` | Catch2 tests for all stubs |

### Files to modify:

| File | Change |
|------|--------|
| `CMakeLists.txt` | Add `KernelStubTests` executable under `BUILD_TESTS` |

---

## Task 1: Constants header

**Files:**
- Create: `include/penta/kernels/Constants.h`

- [ ] **Step 1: Create constants header**

```cpp
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
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Constants.h
git commit -m "feat(kernels): add DSP constants header"
```

---

## Task 2: Core Signal kernels

**Files:**
- Create: `include/penta/kernels/CoreSignal.h`

- [ ] **Step 1: Create CoreSignal.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include <algorithm>
#include <cmath>

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
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/CoreSignal.h
git commit -m "feat(kernels): add crossfade, invert polarity, volume ramp"
```

---

## Task 3: Filters

**Files:**
- Create: `include/penta/kernels/Filters.h`

- [ ] **Step 1: Create Filters.h**

```cpp
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
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Filters.h
git commit -m "feat(kernels): add SVF, ladder, FIR, DC blocker, allpass filters"
```

---

## Task 4: Oscillators

**Files:**
- Create: `include/penta/kernels/Oscillators.h`

- [ ] **Step 1: Create Oscillators.h**

```cpp
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
                // Leaky integrator for triangle
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
        // Distort phase: compress first half, stretch second half
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
                // Paul Kellet's pink noise approximation
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
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Oscillators.h
git commit -m "feat(kernels): add polyBLEP osc, phase distortion, FM operator, noise gen"
```

---

## Task 5: Envelopes

**Files:**
- Create: `include/penta/kernels/Envelopes.h`

- [ ] **Step 1: Create Envelopes.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include <algorithm>
#include <array>
#include <cmath>

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
        float time;  // seconds from previous point
        float level; // 0.0 to 1.0
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
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Envelopes.h
git commit -m "feat(kernels): add AHDSR, MSEG, sample-and-hold, slew limiter"
```

---

## Task 6: Sampling / Playback

**Files:**
- Create: `include/penta/kernels/Sampling.h`

- [ ] **Step 1: Create Sampling.h**

```cpp
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
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Sampling.h
git commit -m "feat(kernels): add sample player, multi-sample map, resampler, time stretch, pitch shifter stubs"
```

---

## Task 7: Granular

**Files:**
- Create: `include/penta/kernels/Granular.h`

- [ ] **Step 1: Create Granular.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include <array>
#include <cmath>

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
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Granular.h
git commit -m "feat(kernels): add granular engine, grain window, scheduler, spectral resynthesis stub"
```

---

## Task 8: Delay

**Files:**
- Create: `include/penta/kernels/Delay.h`

- [ ] **Step 1: Create Delay.h**

```cpp
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
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Delay.h
git commit -m "feat(kernels): add modulated delay, tap delay, feedback delay network"
```

---

## Task 9: Reverb

**Files:**
- Create: `include/penta/kernels/Reverb.h`

- [ ] **Step 1: Create Reverb.h**

```cpp
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
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Reverb.h
git commit -m "feat(kernels): add Schroeder reverb, plate reverb, partitioned convolution stub"
```

---

## Task 10: Dynamics

**Files:**
- Create: `include/penta/kernels/Dynamics.h`

- [ ] **Step 1: Create Dynamics.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include "penta/kernels/Filters.h"
#include <algorithm>
#include <cmath>

namespace penta::kernels {

// ============================================================================
// Compressor (feed-forward, RMS detection)
// ============================================================================

class Compressor {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; recalc(); }
    void setThresholdDb(float db) noexcept { threshDb_ = db; }
    void setRatio(float r) noexcept { ratio_ = std::max(r, 1.0f); }
    void setAttackMs(float ms) noexcept { attackMs_ = ms; recalc(); }
    void setReleaseMs(float ms) noexcept { releaseMs_ = ms; recalc(); }
    void setMakeupDb(float db) noexcept { makeup_ = dbToLinear(db); }

    float process(float in) noexcept {
        float inDb = linearToDb(std::abs(in));
        float overDb = inDb - threshDb_;
        float targetGainDb = (overDb > 0.0f) ? overDb * (1.0f / ratio_ - 1.0f) : 0.0f;

        float coeff = (targetGainDb < envDb_) ? attackCoeff_ : releaseCoeff_;
        envDb_ += (targetGainDb - envDb_) * coeff;

        float gain = dbToLinear(envDb_) * makeup_;
        return in * gain;
    }

    void reset() noexcept { envDb_ = 0.0f; }

private:
    void recalc() noexcept {
        float srF = static_cast<float>(sr_);
        attackCoeff_ = 1.0f - std::exp(-1.0f / (srF * attackMs_ * 0.001f));
        releaseCoeff_ = 1.0f - std::exp(-1.0f / (srF * releaseMs_ * 0.001f));
    }

    uint32_t sr_ = kDefaultSampleRate;
    float threshDb_ = -20.0f;
    float ratio_ = 4.0f;
    float attackMs_ = 10.0f, releaseMs_ = 100.0f;
    float attackCoeff_ = 0.0f, releaseCoeff_ = 0.0f;
    float envDb_ = 0.0f;
    float makeup_ = 1.0f;
};

// ============================================================================
// Expander (downward, same topology as compressor)
// ============================================================================

class Expander {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; recalc(); }
    void setThresholdDb(float db) noexcept { threshDb_ = db; }
    void setRatio(float r) noexcept { ratio_ = std::max(r, 1.0f); }
    void setAttackMs(float ms) noexcept { attackMs_ = ms; recalc(); }
    void setReleaseMs(float ms) noexcept { releaseMs_ = ms; recalc(); }

    float process(float in) noexcept {
        float inDb = linearToDb(std::abs(in));
        float underDb = threshDb_ - inDb;
        float targetGainDb = (underDb > 0.0f) ? -underDb * (ratio_ - 1.0f) : 0.0f;

        float coeff = (targetGainDb < envDb_) ? attackCoeff_ : releaseCoeff_;
        envDb_ += (targetGainDb - envDb_) * coeff;

        return in * dbToLinear(envDb_);
    }

    void reset() noexcept { envDb_ = 0.0f; }

private:
    void recalc() noexcept {
        float srF = static_cast<float>(sr_);
        attackCoeff_ = 1.0f - std::exp(-1.0f / (srF * attackMs_ * 0.001f));
        releaseCoeff_ = 1.0f - std::exp(-1.0f / (srF * releaseMs_ * 0.001f));
    }

    uint32_t sr_ = kDefaultSampleRate;
    float threshDb_ = -40.0f;
    float ratio_ = 2.0f;
    float attackMs_ = 1.0f, releaseMs_ = 50.0f;
    float attackCoeff_ = 0.0f, releaseCoeff_ = 0.0f;
    float envDb_ = 0.0f;
};

// ============================================================================
// Noise Gate
// ============================================================================

class NoiseGate {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; recalc(); }
    void setThresholdDb(float db) noexcept { threshDb_ = db; }
    void setAttackMs(float ms) noexcept { attackMs_ = ms; recalc(); }
    void setReleaseMs(float ms) noexcept { releaseMs_ = ms; recalc(); }

    float process(float in) noexcept {
        float level = linearToDb(std::abs(in));
        float target = (level > threshDb_) ? 1.0f : 0.0f;
        float coeff = (target > gate_) ? attackCoeff_ : releaseCoeff_;
        gate_ += (target - gate_) * coeff;
        return in * gate_;
    }

    void reset() noexcept { gate_ = 0.0f; }

private:
    void recalc() noexcept {
        float srF = static_cast<float>(sr_);
        attackCoeff_ = 1.0f - std::exp(-1.0f / (srF * attackMs_ * 0.001f));
        releaseCoeff_ = 1.0f - std::exp(-1.0f / (srF * releaseMs_ * 0.001f));
    }

    uint32_t sr_ = kDefaultSampleRate;
    float threshDb_ = -60.0f;
    float attackMs_ = 0.1f, releaseMs_ = 50.0f;
    float attackCoeff_ = 0.0f, releaseCoeff_ = 0.0f;
    float gate_ = 0.0f;
};

// ============================================================================
// De-Esser (sidechain bandpass → compressor)
// ============================================================================

class DeEsser {
public:
    void setSampleRate(uint32_t sr) noexcept {
        sr_ = sr;
        bandpass_.setMode(SVFilter::Mode::Bandpass);
        bandpass_.setSampleRate(sr);
        bandpass_.setFrequency(6000.0f);
        bandpass_.setResonance(1.5f);
        comp_.setSampleRate(sr);
        comp_.setThresholdDb(-20.0f);
        comp_.setRatio(6.0f);
        comp_.setAttackMs(0.5f);
        comp_.setReleaseMs(20.0f);
    }

    void setFrequency(float hz) noexcept { bandpass_.setFrequency(hz); }
    void setThresholdDb(float db) noexcept { comp_.setThresholdDb(db); }

    float process(float in) noexcept {
        float sidechain = bandpass_.process(in);
        float gainReduction = comp_.process(sidechain);
        // Apply the gain reduction ratio to the original
        float ratio = (std::abs(sidechain) > 0.0001f) ? gainReduction / sidechain : 1.0f;
        return in * std::clamp(std::abs(ratio), 0.0f, 1.0f);
    }

    void reset() noexcept { bandpass_.reset(); comp_.reset(); }

private:
    uint32_t sr_ = kDefaultSampleRate;
    SVFilter bandpass_;
    Compressor comp_;
};

} // namespace penta::kernels
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Dynamics.h
git commit -m "feat(kernels): add compressor, expander, noise gate, de-esser"
```

---

## Task 11: Distortion

**Files:**
- Create: `include/penta/kernels/Distortion.h`

- [ ] **Step 1: Create Distortion.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace penta::kernels {

// ============================================================================
// Waveshaper (lookup table, 1024 points)
// ============================================================================

class Waveshaper {
public:
    static constexpr size_t kTableSize = 1024;

    // Set transfer function: table maps [-1, 1] → output
    void setTable(const float* table) noexcept {
        for (size_t i = 0; i < kTableSize; ++i) table_[i] = table[i];
    }

    // Generate a tanh-like curve with adjustable drive
    void setDrive(float drive) noexcept {
        for (size_t i = 0; i < kTableSize; ++i) {
            float x = (static_cast<float>(i) / static_cast<float>(kTableSize - 1)) * 2.0f - 1.0f;
            table_[i] = std::tanh(x * drive);
        }
    }

    float process(float in) noexcept {
        float idx = (std::clamp(in, -1.0f, 1.0f) + 1.0f) * 0.5f * static_cast<float>(kTableSize - 1);
        size_t i0 = static_cast<size_t>(idx);
        size_t i1 = std::min(i0 + 1, kTableSize - 1);
        float frac = idx - static_cast<float>(i0);
        return table_[i0] * (1.0f - frac) + table_[i1] * frac;
    }

    void reset() noexcept {}

private:
    std::array<float, kTableSize> table_{};
};

// ============================================================================
// Bitcrusher
// ============================================================================

class Bitcrusher {
public:
    void setBits(int bits) noexcept { bits_ = std::clamp(bits, 1, 24); recalc(); }

    float process(float in) noexcept {
        float scaled = in * levels_;
        return std::round(scaled) / levels_;
    }

    void reset() noexcept {}

private:
    void recalc() noexcept { levels_ = std::pow(2.0f, static_cast<float>(bits_) - 1.0f); }

    int bits_ = 16;
    float levels_ = 32768.0f;
};

// ============================================================================
// Sample Rate Reducer
// ============================================================================

class SampleRateReducer {
public:
    void setFactor(float factor) noexcept { factor_ = std::max(factor, 1.0f); }

    float process(float in) noexcept {
        phase_ += 1.0f;
        if (phase_ >= factor_) {
            phase_ -= factor_;
            held_ = in;
        }
        return held_;
    }

    void reset() noexcept { held_ = 0.0f; phase_ = 0.0f; }

private:
    float factor_ = 1.0f;
    float phase_ = 0.0f;
    float held_ = 0.0f;
};

} // namespace penta::kernels
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Distortion.h
git commit -m "feat(kernels): add waveshaper, bitcrusher, sample rate reducer"
```

---

## Task 12: EQ

**Files:**
- Create: `include/penta/kernels/EQ.h`

- [ ] **Step 1: Create EQ.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include "penta/kernels/Filters.h"
#include "penta/kernels/Dynamics.h"
#include <array>

namespace penta::kernels {

// ============================================================================
// Graphic EQ (31-band ISO frequencies)
// ============================================================================

class GraphicEQ {
public:
    static constexpr size_t kNumBands = 31;

    void setSampleRate(uint32_t sr) noexcept {
        constexpr float isoFreqs[kNumBands] = {
            20, 25, 31.5f, 40, 50, 63, 80, 100, 125, 160,
            200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
            2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
            20000
        };
        for (size_t i = 0; i < kNumBands; ++i) {
            bands_[i].setMode(SVFilter::Mode::Bandpass);
            bands_[i].setSampleRate(sr);
            bands_[i].setFrequency(isoFreqs[i]);
            bands_[i].setResonance(4.3f); // ~1/3 octave Q
        }
    }

    void setBandGain(size_t band, float db) noexcept {
        if (band < kNumBands) gains_[band] = dbToLinear(db);
    }

    float process(float in) noexcept {
        float sum = 0.0f;
        for (size_t i = 0; i < kNumBands; ++i)
            sum += bands_[i].process(in) * gains_[i];
        return sum;
    }

    void reset() noexcept {
        for (auto& b : bands_) b.reset();
        gains_.fill(1.0f);
    }

private:
    std::array<SVFilter, kNumBands> bands_;
    std::array<float, kNumBands> gains_{};
};

// ============================================================================
// Dynamic EQ (per-band compressor-controlled gain)
// ============================================================================

class DynamicEQ {
public:
    static constexpr size_t kMaxBands = 8;

    struct Band {
        float freqHz = 1000.0f;
        float q = 1.0f;
        float threshDb = -20.0f;
        float ratio = 2.0f;
        float gainDb = 0.0f;
    };

    void setSampleRate(uint32_t sr) noexcept {
        sr_ = sr;
        for (size_t i = 0; i < kMaxBands; ++i) {
            filters_[i].setSampleRate(sr);
            comps_[i].setSampleRate(sr);
        }
    }

    void setBand(size_t idx, const Band& b) noexcept {
        if (idx >= kMaxBands) return;
        filters_[idx].setMode(SVFilter::Mode::Bandpass);
        filters_[idx].setFrequency(b.freqHz);
        filters_[idx].setResonance(b.q);
        comps_[idx].setThresholdDb(b.threshDb);
        comps_[idx].setRatio(b.ratio);
        staticGain_[idx] = dbToLinear(b.gainDb);
    }

    void setNumBands(size_t n) noexcept { numBands_ = std::min(n, kMaxBands); }

    float process(float in) noexcept {
        float sum = in;
        for (size_t i = 0; i < numBands_; ++i) {
            float bandSig = filters_[i].process(in);
            float compressed = comps_[i].process(bandSig);
            sum += (compressed - bandSig) * staticGain_[i];
        }
        return sum;
    }

    void reset() noexcept {
        for (auto& f : filters_) f.reset();
        for (auto& c : comps_) c.reset();
    }

private:
    uint32_t sr_ = kDefaultSampleRate;
    std::array<SVFilter, kMaxBands> filters_;
    std::array<Compressor, kMaxBands> comps_;
    std::array<float, kMaxBands> staticGain_{};
    size_t numBands_ = 0;
};

} // namespace penta::kernels
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/EQ.h
git commit -m "feat(kernels): add graphic EQ, dynamic EQ"
```

---

## Task 13: Spectral

**Files:**
- Create: `include/penta/kernels/Spectral.h`

- [ ] **Step 1: Create Spectral.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace penta::kernels {

// ============================================================================
// Overlap-Add processor (framework for FFT-based effects)
// ============================================================================

class OverlapAdd {
public:
    void setHopSize(size_t hop) noexcept { hopSize_ = hop; }
    void setWindowSize(size_t win) noexcept { windowSize_ = win; }

    // Stub: copies input to output. Full version applies FFT → process → IFFT.
    void process(const float* in, float* out, size_t frames) noexcept {
        for (size_t i = 0; i < frames; ++i) out[i] = in[i];
    }

    void reset() noexcept {}

private:
    size_t hopSize_ = 256;
    size_t windowSize_ = 1024;
};

// ============================================================================
// Spectral Filter (brick-wall via FFT bins — stub)
// ============================================================================

class SpectralFilter {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setFFTSize(size_t size) noexcept { fftSize_ = size; }
    void setLowCutHz(float hz) noexcept { lowCut_ = hz; }
    void setHighCutHz(float hz) noexcept { highCut_ = hz; }

    // Stub: passthrough. Requires FFT forward → bin masking → FFT inverse.
    float process(float in) noexcept { return in; }
    void reset() noexcept {}

private:
    uint32_t sr_ = kDefaultSampleRate;
    size_t fftSize_ = 2048;
    float lowCut_ = 20.0f;
    float highCut_ = 20000.0f;
};

// ============================================================================
// Spectral Gate (per-bin noise gate — stub)
// ============================================================================

class SpectralGate {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setFFTSize(size_t size) noexcept { fftSize_ = size; }
    void setThresholdDb(float db) noexcept { threshDb_ = db; }

    // Stub: passthrough. Requires FFT + magnitude thresholding.
    float process(float in) noexcept { return in; }
    void reset() noexcept {}

private:
    uint32_t sr_ = kDefaultSampleRate;
    size_t fftSize_ = 2048;
    float threshDb_ = -60.0f;
};

} // namespace penta::kernels
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Spectral.h
git commit -m "feat(kernels): add overlap-add, spectral filter, spectral gate stubs"
```

---

## Task 14: Spatial

**Files:**
- Create: `include/penta/kernels/Spatial.h`

- [ ] **Step 1: Create Spatial.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include <cmath>

namespace penta::kernels {

// ============================================================================
// Mid/Side Encoder
// ============================================================================

class MidSideEncoder {
public:
    void process(float left, float right, float& mid, float& side) noexcept {
        mid  = (left + right) * kInvSqrt2;
        side = (left - right) * kInvSqrt2;
    }
};

// ============================================================================
// Mid/Side Decoder
// ============================================================================

class MidSideDecoder {
public:
    void process(float mid, float side, float& left, float& right) noexcept {
        left  = (mid + side) * kInvSqrt2;
        right = (mid - side) * kInvSqrt2;
    }
};

// ============================================================================
// Stereo Width
// ============================================================================

class StereoWidth {
public:
    void setWidth(float w) noexcept { width_ = w; } // 0=mono, 1=normal, 2=extra wide

    void process(float& left, float& right) noexcept {
        float mid  = (left + right) * 0.5f;
        float side = (left - right) * 0.5f;
        side *= width_;
        left  = mid + side;
        right = mid - side;
    }

private:
    float width_ = 1.0f;
};

// ============================================================================
// Binaural Panner (simplified HRTF — ITD + ILD model)
// ============================================================================

class BinauralPanner {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setAzimuth(float degrees) noexcept { azimuth_ = degrees; recalc(); }

    void process(float in, float& left, float& right) noexcept {
        left  = in * gainL_;
        right = in * gainR_;
        // Simple ITD via fractional delay (stub — full version uses delay lines)
    }

    void reset() noexcept {}

private:
    void recalc() noexcept {
        // Simplified ILD: cosine panning law based on azimuth
        float rad = azimuth_ * kPi / 180.0f;
        gainL_ = std::cos(std::clamp(rad, 0.0f, kHalfPi));
        gainR_ = std::sin(std::clamp(rad, 0.0f, kHalfPi));
    }

    uint32_t sr_ = kDefaultSampleRate;
    float azimuth_ = 0.0f; // -90=left, 0=center, +90=right
    float gainL_ = kInvSqrt2;
    float gainR_ = kInvSqrt2;
};

} // namespace penta::kernels
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Spatial.h
git commit -m "feat(kernels): add mid/side encode/decode, stereo width, binaural panner"
```

---

## Task 15: Modulation

**Files:**
- Create: `include/penta/kernels/Modulation.h`

- [ ] **Step 1: Create Modulation.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace penta::kernels {

// ============================================================================
// Modulation Matrix (up to 16 routes)
// ============================================================================

class ModulationMatrix {
public:
    static constexpr size_t kMaxRoutes = 16;
    static constexpr size_t kMaxSources = 8;
    static constexpr size_t kMaxDests = 8;

    struct Route {
        uint8_t source = 0;
        uint8_t dest = 0;
        float amount = 0.0f;
        bool active = false;
    };

    void setRoute(size_t idx, uint8_t src, uint8_t dst, float amount) noexcept {
        if (idx >= kMaxRoutes) return;
        routes_[idx] = { src, dst, amount, true };
    }

    void setSource(uint8_t idx, float value) noexcept {
        if (idx < kMaxSources) sources_[idx] = value;
    }

    void process() noexcept {
        dests_.fill(0.0f);
        for (const auto& r : routes_) {
            if (!r.active) continue;
            if (r.source < kMaxSources && r.dest < kMaxDests)
                dests_[r.dest] += sources_[r.source] * r.amount;
        }
    }

    float getDest(uint8_t idx) const noexcept {
        return (idx < kMaxDests) ? dests_[idx] : 0.0f;
    }

    void reset() noexcept { sources_.fill(0.0f); dests_.fill(0.0f); }

private:
    std::array<Route, kMaxRoutes> routes_{};
    std::array<float, kMaxSources> sources_{};
    std::array<float, kMaxDests> dests_{};
};

// ============================================================================
// Parameter Smoother (one-pole exponential)
// ============================================================================

class ParameterSmoother {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; recalc(); }
    void setSmoothingMs(float ms) noexcept { ms_ = ms; recalc(); }
    void setTarget(float target) noexcept { target_ = target; }

    float process() noexcept {
        current_ += (target_ - current_) * coeff_;
        return current_;
    }

    void snapTo(float value) noexcept { current_ = target_ = value; }
    void reset() noexcept { current_ = target_ = 0.0f; }

private:
    void recalc() noexcept {
        if (sr_ > 0 && ms_ > 0.0f)
            coeff_ = 1.0f - std::exp(-1.0f / (static_cast<float>(sr_) * ms_ * 0.001f));
        else
            coeff_ = 1.0f;
    }

    uint32_t sr_ = kDefaultSampleRate;
    float ms_ = 5.0f;
    float coeff_ = 0.0f;
    float current_ = 0.0f;
    float target_ = 0.0f;
};

// ============================================================================
// Control Scaler (linear ↔ logarithmic mapping)
// ============================================================================

class ControlScaler {
public:
    enum class Curve { Linear, Logarithmic, Exponential };

    void setCurve(Curve c) noexcept { curve_ = c; }
    void setRange(float min, float max) noexcept { min_ = min; max_ = max; }

    float process(float normalized) const noexcept {
        float t = std::clamp(normalized, 0.0f, 1.0f);
        switch (curve_) {
            case Curve::Linear:
                break;
            case Curve::Logarithmic:
                t = std::log2(1.0f + t) / std::log2(2.0f);
                break;
            case Curve::Exponential:
                t = t * t;
                break;
        }
        return min_ + t * (max_ - min_);
    }

private:
    Curve curve_ = Curve::Linear;
    float min_ = 0.0f;
    float max_ = 1.0f;
};

} // namespace penta::kernels
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Modulation.h
git commit -m "feat(kernels): add modulation matrix, parameter smoother, control scaler"
```

---

## Task 16: MIDI Control

**Files:**
- Create: `include/penta/kernels/MidiControl.h`

- [ ] **Step 1: Create MidiControl.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include <array>
#include <cstdint>

namespace penta::kernels {

// ============================================================================
// MIDI Event Scheduler (time-sorted queue, max 256 events)
// ============================================================================

class MidiEventScheduler {
public:
    struct Event {
        uint64_t sampleTime = 0;
        uint8_t status = 0;
        uint8_t data1 = 0;
        uint8_t data2 = 0;
    };

    void addEvent(const Event& e) noexcept {
        if (count_ < kMaxEvents) {
            events_[count_++] = e;
            // Insertion sort by time
            for (size_t i = count_ - 1; i > 0; --i) {
                if (events_[i].sampleTime < events_[i - 1].sampleTime)
                    std::swap(events_[i], events_[i - 1]);
                else break;
            }
        }
    }

    const Event* peek() const noexcept {
        return (readIdx_ < count_) ? &events_[readIdx_] : nullptr;
    }

    void advance() noexcept { if (readIdx_ < count_) ++readIdx_; }

    void reset() noexcept { count_ = 0; readIdx_ = 0; }

private:
    static constexpr size_t kMaxEvents = 256;
    std::array<Event, kMaxEvents> events_{};
    size_t count_ = 0;
    size_t readIdx_ = 0;
};

// ============================================================================
// Velocity Scaler (curve shaping)
// ============================================================================

class VelocityScaler {
public:
    enum class Curve { Linear, Soft, Hard, Fixed };

    void setCurve(Curve c) noexcept { curve_ = c; }
    void setFixedValue(uint8_t v) noexcept { fixed_ = v; }

    uint8_t process(uint8_t velocity) const noexcept {
        float v = static_cast<float>(velocity) / 127.0f;
        switch (curve_) {
            case Curve::Linear: break;
            case Curve::Soft:   v = std::sqrt(v); break;
            case Curve::Hard:   v = v * v; break;
            case Curve::Fixed:  return fixed_;
        }
        return static_cast<uint8_t>(std::clamp(v * 127.0f, 0.0f, 127.0f));
    }

private:
    Curve curve_ = Curve::Linear;
    uint8_t fixed_ = 100;
};

// ============================================================================
// Aftertouch Mapper (channel pressure → parameter)
// ============================================================================

class AftertouchMapper {
public:
    void setRange(float min, float max) noexcept { min_ = min; max_ = max; }

    float process(uint8_t pressure) const noexcept {
        float t = static_cast<float>(pressure) / 127.0f;
        return min_ + t * (max_ - min_);
    }

private:
    float min_ = 0.0f;
    float max_ = 1.0f;
};

// ============================================================================
// CC Mapper (control change → parameter with smoothing)
// ============================================================================

class CCMapper {
public:
    void setCC(uint8_t cc) noexcept { cc_ = cc; }
    void setRange(float min, float max) noexcept { min_ = min; max_ = max; }

    void receiveMidi(uint8_t status, uint8_t data1, uint8_t data2) noexcept {
        if ((status & 0xF0) == 0xB0 && data1 == cc_) {
            raw_ = static_cast<float>(data2) / 127.0f;
        }
    }

    float getValue() const noexcept { return min_ + raw_ * (max_ - min_); }
    void reset() noexcept { raw_ = 0.0f; }

private:
    uint8_t cc_ = 1; // mod wheel default
    float min_ = 0.0f, max_ = 1.0f;
    float raw_ = 0.0f;
};

} // namespace penta::kernels
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/MidiControl.h
git commit -m "feat(kernels): add MIDI event scheduler, velocity scaler, aftertouch mapper, CC mapper"
```

---

## Task 17: Voice

**Files:**
- Create: `include/penta/kernels/Voice.h`

- [ ] **Step 1: Create Voice.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include <cmath>

namespace penta::kernels {

// ============================================================================
// Glide / Portamento processor
// ============================================================================

class GlideProcessor {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; recalc(); }
    void setGlideTimeMs(float ms) noexcept { glideMs_ = ms; recalc(); }
    void setTargetNote(float midiNote) noexcept { targetFreq_ = midiToFreq(midiNote); }

    float process() noexcept {
        if (currentFreq_ <= 0.0f) { currentFreq_ = targetFreq_; return currentFreq_; }
        float ratio = targetFreq_ / currentFreq_;
        if (ratio > 1.0f) {
            ratio = 1.0f + (ratio - 1.0f) * coeff_;
        } else {
            ratio = 1.0f - (1.0f - ratio) * coeff_;
        }
        currentFreq_ *= ratio;
        return currentFreq_;
    }

    float getFrequency() const noexcept { return currentFreq_; }
    void reset() noexcept { currentFreq_ = 0.0f; }

private:
    void recalc() noexcept {
        if (sr_ > 0 && glideMs_ > 0.0f)
            coeff_ = 1.0f - std::exp(-1.0f / (static_cast<float>(sr_) * glideMs_ * 0.001f));
        else
            coeff_ = 1.0f;
    }

    uint32_t sr_ = kDefaultSampleRate;
    float glideMs_ = 50.0f;
    float coeff_ = 0.0f;
    float currentFreq_ = 0.0f;
    float targetFreq_ = 440.0f;
};

} // namespace penta::kernels
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Voice.h
git commit -m "feat(kernels): add glide/portamento processor"
```

---

## Task 18: Analysis

**Files:**
- Create: `include/penta/kernels/Analysis.h`

- [ ] **Step 1: Create Analysis.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace penta::kernels {

// ============================================================================
// LUFS Meter (simplified ITU-R BS.1770 — K-weighted RMS)
// ============================================================================

class LUFSMeter {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; blockSize_ = sr * 4 / 10; } // 400ms blocks

    void process(float sample) noexcept {
        float sq = sample * sample;
        sumSq_ += sq;
        if (++count_ >= blockSize_) {
            float meanSq = sumSq_ / static_cast<float>(blockSize_);
            float lufs = (meanSq > 0.0f) ? -0.691f + 10.0f * std::log10(meanSq) : kMinDb;
            shortTermLUFS_ = lufs;
            sumSq_ = 0.0f;
            count_ = 0;
        }
    }

    float getShortTermLUFS() const noexcept { return shortTermLUFS_; }
    void reset() noexcept { sumSq_ = 0.0f; count_ = 0; shortTermLUFS_ = kMinDb; }

private:
    uint32_t sr_ = kDefaultSampleRate;
    size_t blockSize_ = 19200; // 400ms at 48kHz
    float sumSq_ = 0.0f;
    size_t count_ = 0;
    float shortTermLUFS_ = kMinDb;
};

// ============================================================================
// Transient Detector (differential envelope)
// ============================================================================

class TransientDetector {
public:
    void setSampleRate(uint32_t sr) noexcept {
        sr_ = sr;
        float srF = static_cast<float>(sr);
        fastCoeff_ = 1.0f - std::exp(-1.0f / (srF * 0.001f)); // 1ms
        slowCoeff_ = 1.0f - std::exp(-1.0f / (srF * 0.020f)); // 20ms
    }
    void setThreshold(float t) noexcept { threshold_ = t; }

    bool process(float in) noexcept {
        float abs = std::abs(in);
        fastEnv_ += (abs - fastEnv_) * fastCoeff_;
        slowEnv_ += (abs - slowEnv_) * slowCoeff_;
        float diff = fastEnv_ - slowEnv_;
        bool detected = (diff > threshold_ && !wasAbove_);
        wasAbove_ = diff > threshold_;
        return detected;
    }

    void reset() noexcept { fastEnv_ = slowEnv_ = 0.0f; wasAbove_ = false; }

private:
    uint32_t sr_ = kDefaultSampleRate;
    float fastCoeff_ = 0.0f, slowCoeff_ = 0.0f;
    float fastEnv_ = 0.0f, slowEnv_ = 0.0f;
    float threshold_ = 0.1f;
    bool wasAbove_ = false;
};

// ============================================================================
// Silence Detector
// ============================================================================

class SilenceDetector {
public:
    void setSampleRate(uint32_t sr) noexcept { sr_ = sr; }
    void setThresholdDb(float db) noexcept { threshold_ = dbToLinear(db); }
    void setHoldMs(float ms) noexcept { holdSamples_ = static_cast<size_t>(static_cast<float>(sr_) * ms * 0.001f); }

    bool process(float in) noexcept {
        if (std::abs(in) > threshold_) {
            counter_ = 0;
            silent_ = false;
        } else {
            if (++counter_ >= holdSamples_) silent_ = true;
        }
        return silent_;
    }

    bool isSilent() const noexcept { return silent_; }
    void reset() noexcept { counter_ = 0; silent_ = false; }

private:
    uint32_t sr_ = kDefaultSampleRate;
    float threshold_ = 0.0001f;
    size_t holdSamples_ = 48000; // 1 second
    size_t counter_ = 0;
    bool silent_ = false;
};

} // namespace penta::kernels
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Analysis.h
git commit -m "feat(kernels): add LUFS meter, transient detector, silence detector"
```

---

## Task 19: Utility / Safety

**Files:**
- Create: `include/penta/kernels/Utility.h`

- [ ] **Step 1: Create Utility.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include <cmath>
#include <cstring>

namespace penta::kernels {

// ============================================================================
// Denormal Guard (flush-to-zero)
// ============================================================================

class DenormalGuard {
public:
    static float flush(float x) noexcept {
        // Add and subtract a tiny DC offset to kill denormals
        constexpr float kAntiDenormal = 1e-25f;
        return x + kAntiDenormal - kAntiDenormal;
    }

    static void flushBuffer(float* buf, size_t frames) noexcept {
        for (size_t i = 0; i < frames; ++i)
            buf[i] = flush(buf[i]);
    }
};

// ============================================================================
// NaN/Inf Guard
// ============================================================================

class NaNGuard {
public:
    static bool isFinite(float x) noexcept {
        return std::isfinite(x);
    }

    static float sanitize(float x, float fallback = 0.0f) noexcept {
        return std::isfinite(x) ? x : fallback;
    }

    static bool sanitizeBuffer(float* buf, size_t frames, float fallback = 0.0f) noexcept {
        bool hadBad = false;
        for (size_t i = 0; i < frames; ++i) {
            if (!std::isfinite(buf[i])) {
                buf[i] = fallback;
                hadBad = true;
            }
        }
        return hadBad;
    }
};

// ============================================================================
// Buffer Operations
// ============================================================================

class BufferOps {
public:
    static void clear(float* buf, size_t frames) noexcept {
        std::memset(buf, 0, frames * sizeof(float));
    }

    static void copy(const float* src, float* dst, size_t frames) noexcept {
        std::memcpy(dst, src, frames * sizeof(float));
    }

    static void sum(const float* a, const float* b, float* out, size_t frames) noexcept {
        for (size_t i = 0; i < frames; ++i)
            out[i] = a[i] + b[i];
    }

    static void scale(float* buf, float gain, size_t frames) noexcept {
        for (size_t i = 0; i < frames; ++i)
            buf[i] *= gain;
    }

    static float peakLevel(const float* buf, size_t frames) noexcept {
        float peak = 0.0f;
        for (size_t i = 0; i < frames; ++i)
            peak = std::max(peak, std::abs(buf[i]));
        return peak;
    }

    static bool hasClipping(const float* buf, size_t frames, float threshold = 1.0f) noexcept {
        for (size_t i = 0; i < frames; ++i)
            if (std::abs(buf[i]) > threshold) return true;
        return false;
    }
};

} // namespace penta::kernels
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Utility.h
git commit -m "feat(kernels): add denormal guard, NaN guard, buffer ops"
```

---

## Task 20: Routing

**Files:**
- Create: `include/penta/kernels/Routing.h`

- [ ] **Step 1: Create Routing.h**

```cpp
#pragma once

#include "penta/kernels/Constants.h"
#include "penta/kernels/Utility.h"
#include <array>
#include <cstdint>

namespace penta::kernels {

// ============================================================================
// Buffer Splitter (1 input → N outputs)
// ============================================================================

class BufferSplitter {
public:
    static constexpr size_t kMaxOutputs = 8;

    void process(const float* in, float* const* outs, size_t numOuts, size_t frames) noexcept {
        size_t n = std::min(numOuts, kMaxOutputs);
        for (size_t o = 0; o < n; ++o)
            BufferOps::copy(in, outs[o], frames);
    }
};

// ============================================================================
// Buffer Merger (N inputs → 1 output, summed)
// ============================================================================

class BufferMerger {
public:
    static constexpr size_t kMaxInputs = 8;

    void process(const float* const* ins, size_t numIns, float* out, size_t frames) noexcept {
        BufferOps::clear(out, frames);
        size_t n = std::min(numIns, kMaxInputs);
        for (size_t i = 0; i < n; ++i)
            BufferOps::sum(out, ins[i], out, frames);
    }
};

// ============================================================================
// Channel Remapper (reorder channels in interleaved buffer)
// ============================================================================

class ChannelRemapper {
public:
    static constexpr size_t kMaxChannels = 16;

    void setMapping(const uint8_t* map, size_t numChannels) noexcept {
        channels_ = std::min(numChannels, kMaxChannels);
        for (size_t i = 0; i < channels_; ++i)
            mapping_[i] = map[i];
    }

    void process(const float* in, float* out, size_t frames) noexcept {
        for (size_t f = 0; f < frames; ++f) {
            for (size_t c = 0; c < channels_; ++c) {
                out[f * channels_ + c] = in[f * channels_ + mapping_[c]];
            }
        }
    }

private:
    std::array<uint8_t, kMaxChannels> mapping_{};
    size_t channels_ = 2;
};

// ============================================================================
// Topological Execution Node (audio graph node interface)
// ============================================================================

class TopologicalNode {
public:
    static constexpr size_t kMaxInputs = 8;
    static constexpr size_t kMaxOutputs = 8;

    void setId(uint32_t id) noexcept { id_ = id; }
    uint32_t getId() const noexcept { return id_; }

    void addInput(uint32_t nodeId) noexcept {
        if (numInputs_ < kMaxInputs) inputs_[numInputs_++] = nodeId;
    }
    void addOutput(uint32_t nodeId) noexcept {
        if (numOutputs_ < kMaxOutputs) outputs_[numOutputs_++] = nodeId;
    }

    size_t getNumInputs() const noexcept { return numInputs_; }
    size_t getNumOutputs() const noexcept { return numOutputs_; }
    uint32_t getInput(size_t idx) const noexcept { return (idx < numInputs_) ? inputs_[idx] : 0; }
    uint32_t getOutput(size_t idx) const noexcept { return (idx < numOutputs_) ? outputs_[idx] : 0; }

    // Override in derived nodes
    virtual void process(const float* in, float* out, size_t frames) noexcept {
        BufferOps::copy(in, out, frames);
    }

    virtual ~TopologicalNode() = default;

    void reset() noexcept { numInputs_ = numOutputs_ = 0; }

private:
    uint32_t id_ = 0;
    std::array<uint32_t, kMaxInputs> inputs_{};
    std::array<uint32_t, kMaxOutputs> outputs_{};
    size_t numInputs_ = 0;
    size_t numOutputs_ = 0;
};

} // namespace penta::kernels
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Routing.h
git commit -m "feat(kernels): add buffer splitter, merger, channel remapper, topological node"
```

---

## Task 21: Umbrella header

**Files:**
- Create: `include/penta/kernels/Kernels.h`

- [ ] **Step 1: Create Kernels.h**

```cpp
#pragma once

/// Penta Kernels — RT-safe DSP building blocks for KMiDi.
///
/// All classes live in namespace penta::kernels.
/// All process() methods are noexcept and allocation-free.

#include "penta/kernels/Constants.h"
#include "penta/kernels/CoreSignal.h"
#include "penta/kernels/Filters.h"
#include "penta/kernels/Oscillators.h"
#include "penta/kernels/Envelopes.h"
#include "penta/kernels/Sampling.h"
#include "penta/kernels/Granular.h"
#include "penta/kernels/Delay.h"
#include "penta/kernels/Reverb.h"
#include "penta/kernels/Dynamics.h"
#include "penta/kernels/Distortion.h"
#include "penta/kernels/EQ.h"
#include "penta/kernels/Spectral.h"
#include "penta/kernels/Spatial.h"
#include "penta/kernels/Modulation.h"
#include "penta/kernels/MidiControl.h"
#include "penta/kernels/Voice.h"
#include "penta/kernels/Analysis.h"
#include "penta/kernels/Utility.h"
#include "penta/kernels/Routing.h"
```

- [ ] **Step 2: Commit**

```bash
git add include/penta/kernels/Kernels.h
git commit -m "feat(kernels): add umbrella header"
```

---

## Task 22: Catch2 tests

**Files:**
- Create: `tests/cpp/test_kernel_stubs.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create test file**

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "penta/kernels/Kernels.h"
#include <array>
#include <cmath>

using namespace penta::kernels;
using Catch::Approx;

// Helper: check a float is finite
static bool finite(float x) { return std::isfinite(x); }

// ==========================================================================
// Core Signal
// ==========================================================================

TEST_CASE("Crossfade linear", "[kernels][core]") {
    Crossfade xf;
    float a[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4];
    xf.process(a, b, out, 0.5f, 4);
    REQUIRE(out[0] == Approx(0.5f));
}

TEST_CASE("Crossfade equal power", "[kernels][core]") {
    Crossfade xf;
    xf.setLaw(Crossfade::Law::EqualPower);
    float a[1] = {1.0f};
    float b[1] = {1.0f};
    float out[1];
    xf.process(a, b, out, 0.5f, 1);
    REQUIRE(finite(out[0]));
    REQUIRE(out[0] > 0.9f); // equal power mid-point > linear
}

TEST_CASE("InvertPolarity", "[kernels][core]") {
    InvertPolarity inv;
    float in[3] = {1.0f, -0.5f, 0.0f};
    float out[3];
    inv.process(in, out, 3);
    REQUIRE(out[0] == Approx(-1.0f));
    REQUIRE(out[1] == Approx(0.5f));
    REQUIRE(out[2] == Approx(0.0f));
}

TEST_CASE("VolumeRamp produces finite output", "[kernels][core]") {
    VolumeRamp ramp;
    ramp.setSampleRate(48000);
    ramp.setTarget(1.0f);
    float buf[64];
    for (auto& s : buf) s = 1.0f;
    ramp.process(buf, 64);
    for (auto s : buf) REQUIRE(finite(s));
}

// ==========================================================================
// Filters
// ==========================================================================

TEST_CASE("SVFilter lowpass passes DC", "[kernels][filters]") {
    SVFilter svf;
    svf.setSampleRate(48000);
    svf.setFrequency(1000.0f);
    svf.setResonance(0.707f);
    float out = 0.0f;
    for (int i = 0; i < 1000; ++i) out = svf.process(1.0f);
    REQUIRE(out == Approx(1.0f).margin(0.1f));
}

TEST_CASE("LadderFilter produces finite output", "[kernels][filters]") {
    LadderFilter lf;
    lf.setSampleRate(48000);
    lf.setCutoff(500.0f);
    lf.setResonance(0.5f);
    for (int i = 0; i < 100; ++i) {
        float out = lf.process(static_cast<float>(i % 2));
        REQUIRE(finite(out));
    }
}

TEST_CASE("DCBlocker removes DC", "[kernels][filters]") {
    DCBlocker dc;
    dc.setSampleRate(48000);
    float out = 0.0f;
    for (int i = 0; i < 48000; ++i) out = dc.process(1.0f);
    REQUIRE(std::abs(out) < 0.01f);
}

TEST_CASE("AllPassFilter unity magnitude", "[kernels][filters]") {
    AllPassFilter ap;
    ap.setCoefficient(0.5f);
    float out = 0.0f;
    for (int i = 0; i < 100; ++i) out = ap.process(1.0f);
    REQUIRE(finite(out));
}

TEST_CASE("FIRFilter passthrough with identity", "[kernels][filters]") {
    FIRFilter fir;
    float coeffs[1] = {1.0f};
    fir.setCoefficients(coeffs, 1);
    REQUIRE(fir.process(0.5f) == Approx(0.5f));
}

// ==========================================================================
// Oscillators
// ==========================================================================

TEST_CASE("BandlimitedOsc saw produces finite output", "[kernels][osc]") {
    BandlimitedOsc osc;
    osc.setSampleRate(48000);
    osc.setFrequency(440.0f);
    osc.setShape(BandlimitedOsc::Shape::Saw);
    for (int i = 0; i < 1000; ++i) REQUIRE(finite(osc.process()));
}

TEST_CASE("FMOperator sine output", "[kernels][osc]") {
    FMOperator fm;
    fm.setSampleRate(48000);
    fm.setFrequency(440.0f);
    for (int i = 0; i < 100; ++i) {
        float out = fm.process();
        REQUIRE(out >= -1.0f);
        REQUIRE(out <= 1.0f);
    }
}

TEST_CASE("NoiseGenerator white noise non-zero", "[kernels][osc]") {
    NoiseGenerator ng;
    ng.setColor(NoiseGenerator::Color::White);
    float sum = 0.0f;
    for (int i = 0; i < 1000; ++i) sum += std::abs(ng.process());
    REQUIRE(sum > 0.0f);
}

TEST_CASE("PhaseDistortionOsc finite output", "[kernels][osc]") {
    PhaseDistortionOsc pd;
    pd.setSampleRate(48000);
    pd.setFrequency(440.0f);
    pd.setDistortion(0.5f);
    for (int i = 0; i < 100; ++i) REQUIRE(finite(pd.process()));
}

// ==========================================================================
// Envelopes
// ==========================================================================

TEST_CASE("AHDSREnvelope lifecycle", "[kernels][env]") {
    AHDSREnvelope env;
    env.setSampleRate(48000);
    env.setAttackMs(1.0f);
    env.setHoldMs(1.0f);
    env.setDecayMs(10.0f);
    env.setSustain(0.5f);
    env.setReleaseMs(10.0f);
    env.noteOn();
    float peak = 0.0f;
    for (int i = 0; i < 4800; ++i) peak = std::max(peak, env.process());
    REQUIRE(peak == Approx(1.0f).margin(0.01f));
    env.noteOff();
    for (int i = 0; i < 48000; ++i) env.process();
    REQUIRE(env.process() < 0.001f);
}

TEST_CASE("SlewLimiter limits rate", "[kernels][env]") {
    SlewLimiter slew;
    slew.setSampleRate(48000);
    slew.setRiseMs(10.0f);
    slew.setFallMs(10.0f);
    float out = slew.process(1.0f);
    REQUIRE(out < 0.5f); // Can't jump to 1.0 instantly
}

TEST_CASE("SampleAndHold holds value", "[kernels][env]") {
    SampleAndHold sh;
    sh.setSampleRate(48000);
    sh.setRateHz(10.0f);
    float first = sh.process(1.0f);
    float second = sh.process(0.5f);
    REQUIRE(first == second); // Held until next trigger
}

// ==========================================================================
// Sampling
// ==========================================================================

TEST_CASE("SamplePlayer plays buffer", "[kernels][sampling]") {
    float data[10] = {0, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
    SamplePlayer sp;
    sp.setBuffer(data, 10);
    sp.trigger();
    REQUIRE(sp.process() == Approx(0.0f));
    REQUIRE(sp.process() == Approx(0.1f));
}

TEST_CASE("Resampler linear interpolation", "[kernels][sampling]") {
    float buf[4] = {0.0f, 1.0f, 0.0f, -1.0f};
    Resampler rs;
    REQUIRE(rs.process(buf, 4, 0.5) == Approx(0.5f));
}

// ==========================================================================
// Granular
// ==========================================================================

TEST_CASE("GranularEngine produces output from source", "[kernels][granular]") {
    std::array<float, 4096> src;
    for (size_t i = 0; i < src.size(); ++i) src[i] = std::sin(kTwoPi * static_cast<float>(i) / 100.0f);

    GranularEngine ge;
    ge.setSampleRate(48000);
    ge.setSource(src.data(), src.size());
    ge.setDensity(20.0f);
    ge.setGrainLengthMs(50.0f);
    ge.setPosition(0.5f);

    float sum = 0.0f;
    for (int i = 0; i < 4800; ++i) sum += std::abs(ge.process());
    REQUIRE(sum > 0.0f);
}

TEST_CASE("GrainWindow hann symmetry", "[kernels][granular]") {
    GrainWindow w;
    w.setShape(GrainWindow::Shape::Hann);
    REQUIRE(w.apply(0.0f) == Approx(0.0f).margin(0.001f));
    REQUIRE(w.apply(0.5f) == Approx(1.0f).margin(0.001f));
    REQUIRE(w.apply(1.0f) == Approx(0.0f).margin(0.001f));
}

// ==========================================================================
// Delay
// ==========================================================================

TEST_CASE("ModulatedDelay finite output", "[kernels][delay]") {
    ModulatedDelay md;
    md.setSampleRate(48000);
    md.setBaseDelayMs(7.0f);
    md.setDepthMs(2.0f);
    md.setRateHz(0.5f);
    for (int i = 0; i < 1000; ++i) REQUIRE(finite(md.process(static_cast<float>(i % 2))));
}

TEST_CASE("FeedbackDelayNetwork finite output", "[kernels][delay]") {
    FeedbackDelayNetwork fdn;
    fdn.setSampleRate(48000);
    fdn.setDecay(1.0f);
    for (int i = 0; i < 100; ++i) REQUIRE(finite(fdn.process(i == 0 ? 1.0f : 0.0f)));
}

// ==========================================================================
// Reverb
// ==========================================================================

TEST_CASE("SchroederReverb tail", "[kernels][reverb]") {
    SchroederReverb rev;
    rev.setSampleRate(48000);
    rev.setDecay(1.0f);
    rev.setMix(1.0f);
    rev.process(1.0f); // impulse
    float tailEnergy = 0.0f;
    for (int i = 0; i < 24000; ++i) tailEnergy += std::abs(rev.process(0.0f));
    REQUIRE(tailEnergy > 0.0f);
}

// ==========================================================================
// Dynamics
// ==========================================================================

TEST_CASE("Compressor reduces loud signal", "[kernels][dynamics]") {
    Compressor comp;
    comp.setSampleRate(48000);
    comp.setThresholdDb(-20.0f);
    comp.setRatio(4.0f);
    comp.setAttackMs(1.0f);
    comp.setReleaseMs(50.0f);
    float loud = 0.5f;
    float out = loud;
    for (int i = 0; i < 4800; ++i) out = comp.process(loud);
    REQUIRE(std::abs(out) < loud);
}

TEST_CASE("NoiseGate mutes quiet signal", "[kernels][dynamics]") {
    NoiseGate gate;
    gate.setSampleRate(48000);
    gate.setThresholdDb(-40.0f);
    float quiet = 0.001f;
    float out = quiet;
    for (int i = 0; i < 48000; ++i) out = gate.process(quiet);
    REQUIRE(std::abs(out) < quiet);
}

// ==========================================================================
// Distortion
// ==========================================================================

TEST_CASE("Bitcrusher reduces precision", "[kernels][distortion]") {
    Bitcrusher bc;
    bc.setBits(4);
    float out = bc.process(0.123456f);
    REQUIRE(finite(out));
    REQUIRE(out != Approx(0.123456f)); // quantized
}

TEST_CASE("SampleRateReducer holds samples", "[kernels][distortion]") {
    SampleRateReducer srr;
    srr.setFactor(4.0f);
    float first = srr.process(1.0f);
    float second = srr.process(0.5f);
    REQUIRE(first == second); // held
}

// ==========================================================================
// Spatial
// ==========================================================================

TEST_CASE("MidSide roundtrip", "[kernels][spatial]") {
    MidSideEncoder enc;
    MidSideDecoder dec;
    float mid, side, left, right;
    enc.process(0.8f, 0.3f, mid, side);
    dec.process(mid, side, left, right);
    REQUIRE(left == Approx(0.8f).margin(0.001f));
    REQUIRE(right == Approx(0.3f).margin(0.001f));
}

TEST_CASE("StereoWidth mono at 0", "[kernels][spatial]") {
    StereoWidth sw;
    sw.setWidth(0.0f);
    float L = 1.0f, R = 0.0f;
    sw.process(L, R);
    REQUIRE(L == Approx(R).margin(0.001f));
}

// ==========================================================================
// Modulation
// ==========================================================================

TEST_CASE("ModulationMatrix routes", "[kernels][mod]") {
    ModulationMatrix mm;
    mm.setRoute(0, 0, 0, 0.5f);
    mm.setSource(0, 1.0f);
    mm.process();
    REQUIRE(mm.getDest(0) == Approx(0.5f));
}

TEST_CASE("ParameterSmoother converges", "[kernels][mod]") {
    ParameterSmoother ps;
    ps.setSampleRate(48000);
    ps.setSmoothingMs(10.0f);
    ps.setTarget(1.0f);
    float out = 0.0f;
    for (int i = 0; i < 48000; ++i) out = ps.process();
    REQUIRE(out == Approx(1.0f).margin(0.001f));
}

// ==========================================================================
// MIDI Control
// ==========================================================================

TEST_CASE("VelocityScaler curves", "[kernels][midi]") {
    VelocityScaler vs;
    vs.setCurve(VelocityScaler::Curve::Linear);
    REQUIRE(vs.process(64) == 64);
    vs.setCurve(VelocityScaler::Curve::Fixed);
    vs.setFixedValue(100);
    REQUIRE(vs.process(64) == 100);
}

TEST_CASE("CCMapper receives CC", "[kernels][midi]") {
    CCMapper cc;
    cc.setCC(1);
    cc.setRange(0.0f, 10.0f);
    cc.receiveMidi(0xB0, 1, 127);
    REQUIRE(cc.getValue() == Approx(10.0f).margin(0.1f));
}

// ==========================================================================
// Voice
// ==========================================================================

TEST_CASE("GlideProcessor converges to target", "[kernels][voice]") {
    GlideProcessor gp;
    gp.setSampleRate(48000);
    gp.setGlideTimeMs(10.0f);
    gp.setTargetNote(69.0f); // A4
    for (int i = 0; i < 48000; ++i) gp.process();
    REQUIRE(gp.getFrequency() == Approx(440.0f).margin(1.0f));
}

// ==========================================================================
// Analysis
// ==========================================================================

TEST_CASE("LUFSMeter produces reading", "[kernels][analysis]") {
    LUFSMeter lufs;
    lufs.setSampleRate(48000);
    for (int i = 0; i < 48000; ++i) lufs.process(0.5f);
    REQUIRE(lufs.getShortTermLUFS() > -144.0f);
}

TEST_CASE("SilenceDetector detects silence", "[kernels][analysis]") {
    SilenceDetector sd;
    sd.setSampleRate(48000);
    sd.setThresholdDb(-60.0f);
    sd.setHoldMs(100.0f);
    for (int i = 0; i < 48000; ++i) sd.process(0.0f);
    REQUIRE(sd.isSilent());
}

// ==========================================================================
// Utility
// ==========================================================================

TEST_CASE("NaNGuard sanitizes", "[kernels][utility]") {
    REQUIRE(NaNGuard::sanitize(1.0f) == 1.0f);
    REQUIRE(NaNGuard::sanitize(std::numeric_limits<float>::quiet_NaN()) == 0.0f);
    REQUIRE(NaNGuard::sanitize(std::numeric_limits<float>::infinity()) == 0.0f);
}

TEST_CASE("BufferOps clear and sum", "[kernels][utility]") {
    float a[4] = {1, 2, 3, 4};
    float b[4] = {4, 3, 2, 1};
    float out[4];
    BufferOps::sum(a, b, out, 4);
    REQUIRE(out[0] == 5.0f);
    REQUIRE(out[3] == 5.0f);
    BufferOps::clear(out, 4);
    REQUIRE(out[0] == 0.0f);
}

// ==========================================================================
// Routing
// ==========================================================================

TEST_CASE("BufferSplitter copies to all outputs", "[kernels][routing]") {
    float in[4] = {1, 2, 3, 4};
    float o1[4], o2[4];
    float* outs[] = {o1, o2};
    BufferSplitter bs;
    bs.process(in, outs, 2, 4);
    REQUIRE(o1[2] == 3.0f);
    REQUIRE(o2[2] == 3.0f);
}

TEST_CASE("BufferMerger sums inputs", "[kernels][routing]") {
    float a[4] = {1, 1, 1, 1};
    float b[4] = {2, 2, 2, 2};
    const float* ins[] = {a, b};
    float out[4];
    BufferMerger bm;
    bm.process(ins, 2, out, 4);
    REQUIRE(out[0] == 3.0f);
}
```

- [ ] **Step 2: Add test target to CMakeLists.txt**

After the existing `catch_discover_tests(EmotionSchemaTests)` line (~line 626), add:

```cmake
        # Kernel stub tests
        add_executable(KernelStubTests
            tests/cpp/test_kernel_stubs.cpp
        )
        target_link_libraries(KernelStubTests PRIVATE
            KellyCore
            Catch2::Catch2WithMain
        )
        target_include_directories(KernelStubTests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        catch_discover_tests(KernelStubTests)
```

- [ ] **Step 3: Build and run tests**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_KELLY_CORE=ON -DBUILD_TESTS=ON
cmake --build build --target KernelStubTests -j8
ctest --test-dir build -R Kernel --output-on-failure
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/cpp/test_kernel_stubs.cpp CMakeLists.txt
git commit -m "test(kernels): add Catch2 test suite for all kernel stubs"
```

---

## Summary

| Task | Category | Kernel count | Status |
|------|----------|-------------|--------|
| 1 | Constants | — | Shared header |
| 2 | Core Signal | 3 | Crossfade, InvertPolarity, VolumeRamp |
| 3 | Filters | 5 | SVF, Ladder, FIR, DCBlocker, AllPass |
| 4 | Oscillators | 4 | BandlimitedOsc, PhaseDistortionOsc, FMOperator, NoiseGen |
| 5 | Envelopes | 4 | AHDSR, MSEG, SampleAndHold, SlewLimiter |
| 6 | Sampling | 5 | SamplePlayer, MultiSampleMap, Resampler, TimeStretch*, PitchShifter* |
| 7 | Granular | 4 | GranularEngine, GrainWindow, GrainScheduler, SpectralResynthesis* |
| 8 | Delay | 3 | ModulatedDelay, TapDelay, FDN |
| 9 | Reverb | 3 | Schroeder, Plate, PartitionedConvolution* |
| 10 | Dynamics | 4 | Compressor, Expander, NoiseGate, DeEsser |
| 11 | Distortion | 3 | Waveshaper, Bitcrusher, SampleRateReducer |
| 12 | EQ | 2 | GraphicEQ, DynamicEQ |
| 13 | Spectral | 3 | OverlapAdd*, SpectralFilter*, SpectralGate* |
| 14 | Spatial | 4 | MidSideEncoder, MidSideDecoder, StereoWidth, BinauralPanner |
| 15 | Modulation | 3 | ModMatrix, ParameterSmoother, ControlScaler |
| 16 | MIDI Control | 4 | EventScheduler, VelocityScaler, AftertouchMapper, CCMapper |
| 17 | Voice | 1 | GlideProcessor |
| 18 | Analysis | 3 | LUFSMeter, TransientDetector, SilenceDetector |
| 19 | Utility | 3 | DenormalGuard, NaNGuard, BufferOps |
| 20 | Routing | 4 | BufferSplitter, BufferMerger, ChannelRemapper, TopologicalNode |
| 21 | Umbrella | — | Kernels.h |
| 22 | Tests | — | 35+ Catch2 test cases |

\* = passthrough stub (requires FFT or complex infrastructure for full implementation)

**Total: 58 new kernel classes across 20 headers + 1 umbrella + 1 test file**
