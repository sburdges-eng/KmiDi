#pragma once

/**
 * AudioEmotionRunner — RT-safe JEPA emotion inference pipeline.
 *
 * Audio thread pushes samples, worker thread runs inference,
 * audio thread reads results and writes to RTState.
 *
 * Gated behind ENABLE_ONNX_RUNTIME. Compiles as no-op stub when disabled.
 */

#include "penta/common/RTState.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace penta::ml {

// ─── Result structs (POD, RT-safe) ──────────────────────────────────────────

struct EmotionResult {
    float valence    = 0.0f;   // [-1, 1]
    float arousal    = 0.5f;   // [0, 1]
    float dominance  = 0.5f;   // [0, 1]
    float confidence = 0.0f;   // [0, 1]
};

struct DSPSuggestion {
    float filter_cutoff = 0.5f;   // [0, 1] normalized
    float reverb_wet    = 0.2f;   // [0, 1]
    float drive_amount  = 0.0f;   // [0, 1]
};

struct EmotionRunnerResult {
    EmotionResult emotion;
    DSPSuggestion dsp;
    uint64_t sequence_id = 0;   // monotonic, for staleness detection
};

// ─── Configuration ──────────────────────────────────────────────────────────

struct AudioEmotionRunnerConfig {
    std::string model_path;                // Path to JEPA .onnx file
    std::string probe_model_path;          // Path to emotion probe .onnx file
    size_t sample_rate         = 48000;
    size_t ring_capacity       = 524288;   // ~10.9s at 48kHz
    float  slew_time_ms        = 20.0f;    // Per-parameter ramp
    float  watchdog_timeout_ms = 100.0f;   // Max staleness before fallback
    float  confidence_threshold = 0.3f;    // Below this, hold last-known-good
};

// ─── Forward declaration of implementation ──────────────────────────────────

struct AudioEmotionRunnerImpl;

// ─── Main class ─────────────────────────────────────────────────────────────

class AudioEmotionRunner {
public:
    AudioEmotionRunner();
    ~AudioEmotionRunner();

    // Non-copyable, non-movable
    AudioEmotionRunner(const AudioEmotionRunner&) = delete;
    AudioEmotionRunner& operator=(const AudioEmotionRunner&) = delete;

    /**
     * Initialize the runner: load ONNX model, allocate buffers, spawn worker.
     * Call from prepareToPlay() (non-RT thread).
     * @return true if successful
     */
    bool initialize(const AudioEmotionRunnerConfig& config);

    /**
     * Shut down: stop worker thread, release resources.
     * Call from releaseResources() (non-RT thread).
     */
    void shutdown();

    /**
     * Push audio samples into the ring buffer.
     * Call from audio thread. Non-blocking. Drops if ring is full.
     */
    void pushSamples(const float* samples, size_t count) noexcept;

    /**
     * Read latest inference result, apply slew limiting, write to RTState.
     * Call from audio thread each processBlock.
     * @param state  RTState to update (emotion + trackParams for DSP)
     * @param blockSize  Number of samples in this block (for slew coefficient)
     */
    void updateParams(penta::RTState& state, size_t blockSize) noexcept;

    // ─── Non-RT diagnostics ─────────────────────────────────────────────
    bool isRunning() const;
    float lastInferenceMs() const;
    uint64_t lastSequenceId() const;
    uint64_t droppedSamples() const;

    /**
     * T6.7: EMA drop-rate in samples/second.
     * Safe to call from any non-RT thread (reads two atomics with relaxed order).
     * Returns 0.0 until at least one call has established the baseline.
     */
    float dropRate() const;

private:
    std::unique_ptr<AudioEmotionRunnerImpl> impl_;
};

} // namespace penta::ml
