#pragma once

#include "penta/common/RTTypes.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

// Cache-line sizing: Apple Silicon uses 128-byte cache lines (two 64-byte
// pairs treated as a single coherence unit by the prefetcher), Intel/x86_64
// uses 64-byte lines. If libc++ exposes the hardware interference size, use
// it; otherwise fall back to the per-arch default. This matters because RTState
// is read by the bridge on a non-audio thread at high frequency — if the
// readers sample atomics on the same cache line that an audio-thread writer
// just touched, every read snapshot triggers a full coherence miss.
#if defined(__cpp_lib_hardware_interference_size)
  #include <new>
  inline constexpr std::size_t kCacheLine = std::hardware_destructive_interference_size;
#elif defined(__aarch64__) && defined(__APPLE__)
  inline constexpr std::size_t kCacheLine = 128;  // Apple M-series
#else
  inline constexpr std::size_t kCacheLine = 64;
#endif

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
 * Publishing protocol (seqlock):
 *   Writers that update more than one field must bracket the stores with
 *   begin_publish() / end_publish(). Readers call rt_state_snapshot()
 *   (free function) to obtain a torn-free snapshot.
 *
 * Single-field writers (e.g. a transport toggle that only stores `playing`)
 * do NOT need the seqlock — a single atomic store is already consistent.
 *
 * Cache-line layout:
 *   Fields are grouped by writer so stores from one thread don't invalidate
 *   another thread's line. The groups are:
 *     - sequence counter          (read every snapshot; own line)
 *     - timing                    (audio thread; host clock)
 *     - emotion                   (ML worker via updateParams)
 *     - musical intent            (bridge / UI writes)
 *     - track params              (bridge writes)
 *
 * Layout mirrors IntentFrame's EmotionState + TimingInfo + MusicalIntent
 * so the bridge can reconstruct the full picture without locking.
 */
// Verify atomics are lock-free on this platform (RT safety requirement).
static_assert(std::atomic<double>::is_always_lock_free,
              "double atomics must be lock-free for RT safety");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "uint64_t atomics must be lock-free for RT safety");
static_assert(std::atomic<float>::is_always_lock_free,
              "float atomics must be lock-free for RT safety");

struct RTState {
    // --- Sequence counter: owns its own cache line so rapid seqlock bumps
    // don't ping-pong timing/emotion data. Even when idle, reader snapshots
    // this atomic twice per call; keeping it hot on its own line is free.
    alignas(kCacheLine) std::atomic<uint64_t> sequence{0};

    // --- Group 1: Timing (audio thread) ---
    alignas(kCacheLine) std::atomic<double>   bpm{120.0};
    std::atomic<uint64_t> samplePosition{0};
    std::atomic<uint64_t> barStart{0};        // sample pos of current bar
    std::atomic<uint32_t> bar{0};             // current bar number
    std::atomic<uint32_t> beat{0};            // current beat within bar
    std::atomic<uint32_t> numerator{4};
    std::atomic<uint32_t> denominator{4};
    std::atomic<bool>     playing{false};

    // --- Group 2: Emotion (VAD model) — ML worker via AudioEmotionRunner ---
    alignas(kCacheLine) std::atomic<float> valence{0.0f};      // [-1.0, 1.0]
    std::atomic<float>    arousal{0.5f};                        // [0.0, 1.0]
    std::atomic<float>    dominance{0.5f};                      // [0.0, 1.0]
    std::atomic<int16_t>  discreteEmotionId{-1};                // -1 = unused
    std::atomic<float>    emotionIntensity{0.0f};
    std::atomic<float>    emotionConfidence{0.0f};

    // --- Group 3: Musical intent biases (bridge / UI writes) ---
    alignas(kCacheLine) std::atomic<float> tempoBias{0.0f};     // [-1.0, 1.0]
    std::atomic<float>    rhythmicDensity{0.5f};                // [0.0, 1.0]
    std::atomic<float>    grooveStrength{0.5f};                 // [0.0, 1.0]
    std::atomic<float>    harmonicTension{0.5f};                // [0.0, 1.0]
    std::atomic<float>    harmonicMotion{0.5f};                 // [0.0, 1.0]
    std::atomic<float>    melodicActivity{0.5f};                // [0.0, 1.0]
    std::atomic<float>    textureDensity{0.5f};                 // [0.0, 1.0]
    std::atomic<float>    dynamicRange{0.5f};                   // [0.0, 1.0]

    // --- Group 4: Track parameters (generic float array for extensibility) ---
    alignas(kCacheLine) std::array<std::atomic<float>, kMaxTrackParams> trackParams{};

    // Trailing pad so whatever follows this struct in an enclosing aggregate
    // doesn't share a line with trackParams.
    alignas(kCacheLine) std::byte _tail_pad[1]{};

    RTState() {
        for (auto& p : trackParams) p.store(0.0f, std::memory_order_relaxed);
    }

    // Non-copyable (atomics)
    RTState(const RTState&) = delete;
    RTState& operator=(const RTState&) = delete;

    // ----------------------------------------------------------------------
    // Seqlock publish protocol
    //
    // Writers that update multiple fields must wrap the batch:
    //     state.begin_publish();
    //     state.valence.store(v, std::memory_order_relaxed);
    //     state.arousal.store(a, std::memory_order_relaxed);
    //     state.end_publish();
    //
    // begin_publish() marks `sequence` as odd (writer in progress).
    // end_publish() returns it to even (publish complete). Readers that see
    // an odd sequence retry; readers that see the sequence change between
    // their two loads also retry (they observed a torn window).
    // ----------------------------------------------------------------------

    /**
     * Writer-side: call before publishing a batch of stores.
     *
     * Uses acq_rel so the odd marker is visible before subsequent relaxed
     * stores (pairs with the reader's acquire load of `sequence`).
     *
     * @return the previous (even) sequence value before the bump.
     */
    uint64_t begin_publish() noexcept {
        return sequence.fetch_add(1, std::memory_order_acq_rel);
    }

    /**
     * Writer-side: call after publishing. Release-order bump makes all
     * prior relaxed stores globally visible and returns sequence to even.
     */
    void end_publish() noexcept {
        sequence.fetch_add(1, std::memory_order_release);
    }
};

/**
 * rt_state_snapshot — reader-side seqlock helper.
 *
 * `copy_all` is a caller-supplied lambda that performs all the relaxed
 * loads into a caller-owned struct. Returns true on a stable snapshot,
 * false if 64 retries exhausted (writer thrashing). Each attempt is just
 * two atomic loads so the budget is negligible; 64 retries eliminates
 * spurious KELLY_ERROR_AGAIN on loaded systems while still bounding
 * spin time. If we exhaust it, something is genuinely wrong and the
 * caller should surface KELLY_ERROR_AGAIN rather than return torn state.
 *
 * Memory ordering:
 *   - seq1 is loaded acquire; pairs with the writer's release bump
 *     inside begin_publish() (acq_rel = release for this side).
 *   - the fence after copy_all is acquire, so the relaxed loads can be
 *     reordered no later than the fence and no earlier than the seq1 load.
 *   - seq2 is loaded relaxed because the acquire fence already dominates.
 */
template <typename F>
bool rt_state_snapshot(const RTState& s, F&& copy_all) noexcept(noexcept(copy_all())) {
    for (int attempt = 0; attempt < 64; ++attempt) {
        uint64_t seq1 = s.sequence.load(std::memory_order_acquire);
        if (seq1 & 1ull) continue;       // writer in progress
        copy_all();
        std::atomic_thread_fence(std::memory_order_acquire);
        uint64_t seq2 = s.sequence.load(std::memory_order_relaxed);
        if (seq1 == seq2) return true;   // stable snapshot
    }
    return false;
}

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
