#pragma once

#include "penta/common/RTTypes.h"

#include <array>
#include <atomic>
#include <cstdint>

namespace penta {

// Maximum number of emotion dimensions (VAD + intensity + confidence + discrete_id)
constexpr size_t kEmotionDims = 6;
// Maximum track parameters exposed to the bridge
constexpr size_t kMaxTrackParams = 16;

/**
 * RTState — Snapshot of the real-time engine state.
 *
 * Published by the audio thread (writer), consumed by the Python bridge (reader).
 * All fields are atomic for lock-free access. No heap allocations.
 *
 * Layout mirrors IntentFrame's EmotionState + TimingInfo + MusicalIntent
 * so the bridge can reconstruct the full picture without locking.
 */
// Verify atomics are lock-free on this platform (RT safety requirement)
static_assert(std::atomic<double>::is_always_lock_free,
              "double atomics must be lock-free for RT safety");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "uint64_t atomics must be lock-free for RT safety");
static_assert(std::atomic<float>::is_always_lock_free,
              "float atomics must be lock-free for RT safety");

struct RTState {
    // --- Timing ---
    std::atomic<double>   bpm{120.0};
    std::atomic<uint64_t> samplePosition{0};
    std::atomic<uint64_t> barStart{0};        // sample pos of current bar
    std::atomic<uint32_t> bar{0};             // current bar number
    std::atomic<uint32_t> beat{0};            // current beat within bar
    std::atomic<uint32_t> numerator{4};
    std::atomic<uint32_t> denominator{4};
    std::atomic<bool>     playing{false};

    // --- Emotion (VAD model) ---
    std::atomic<float>    valence{0.0f};      // [-1.0, 1.0]
    std::atomic<float>    arousal{0.5f};       // [0.0, 1.0]
    std::atomic<float>    dominance{0.5f};     // [0.0, 1.0]
    std::atomic<int16_t>  discreteEmotionId{-1}; // -1 = unused
    std::atomic<float>    emotionIntensity{0.0f};
    std::atomic<float>    emotionConfidence{0.0f};

    // --- Musical intent biases ---
    std::atomic<float>    tempoBias{0.0f};         // [-1.0, 1.0]
    std::atomic<float>    rhythmicDensity{0.5f};   // [0.0, 1.0]
    std::atomic<float>    grooveStrength{0.5f};     // [0.0, 1.0]
    std::atomic<float>    harmonicTension{0.5f};    // [0.0, 1.0]
    std::atomic<float>    harmonicMotion{0.5f};     // [0.0, 1.0]
    std::atomic<float>    melodicActivity{0.5f};    // [0.0, 1.0]
    std::atomic<float>    textureDensity{0.5f};     // [0.0, 1.0]
    std::atomic<float>    dynamicRange{0.5f};       // [0.0, 1.0]

    // --- Track parameters (generic float array for extensibility) ---
    std::array<std::atomic<float>, kMaxTrackParams> trackParams{};

    // --- Sequence counter (incremented each audio callback) ---
    std::atomic<uint64_t> sequence{0};

    RTState() {
        for (auto& p : trackParams) p.store(0.0f, std::memory_order_relaxed);
    }

    // Non-copyable (atomics)
    RTState(const RTState&) = delete;
    RTState& operator=(const RTState&) = delete;
};

/**
 * RTParameterUpdate — A single parameter change request.
 *
 * Pushed by the Python bridge (writer), consumed by the audio thread (reader).
 * POD type suitable for readerwriterqueue.
 */
struct RTParameterUpdate {
    enum class Target : uint8_t {
        BPM,
        Emotion,         // valence, arousal, dominance, intensity
        MusicalIntent,   // any MusicalIntent field
        TrackParam,      // indexed track parameter
        Transport        // play/stop/seek
    };

    Target   target;
    uint8_t  paramIndex;  // which sub-parameter or track index
    float    value;
    uint64_t timestamp;   // sample position or 0 for immediate

    RTParameterUpdate()
        : target(Target::BPM), paramIndex(0), value(0.0f), timestamp(0) {}

    RTParameterUpdate(Target t, uint8_t idx, float v, uint64_t ts = 0)
        : target(t), paramIndex(idx), value(v), timestamp(ts) {}
};

} // namespace penta
