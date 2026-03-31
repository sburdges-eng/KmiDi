# AudioEmotionRunner Design Spec

**Date:** 2026-03-31
**Status:** Approved
**Phase:** 2-3 bridge (90-Day Demo Roadmap)

## Goal

A C++ component that accepts raw audio samples, runs emotion inference via the Audio JEPA ONNX model on a worker thread, and outputs both an `EmotionState` and suggested DSP parameters to `RTState` — with zero allocations on the audio thread.

## Architecture

### Threading Model

```
Audio Thread                     Worker Thread (AudioWorkerThread)
─────────────                    ─────────────────────────────────
processBlock()                   inference loop:
  ├─ pushSamples → ring buffer     ├─ drain ring buffer
  ├─ updateParams:                 ├─ compute mel spectrogram (128 bins, 512 frames)
  │   ├─ read latest result ◄────  ├─ run ONNX inference → (1, 512, 256) latent
  │   ├─ slew-limit DSP params     ├─ mean-pool latent → (256,)
  │   └─ write to RTState          ├─ map latent → EmotionResult (VAD + confidence)
  └─ continue DSP processing       ├─ map emotion → DSPSuggestion (filter/reverb/drive)
                                    └─ post EmotionRunnerResult → SPSC queue
```

### Key Invariants

- **Audio thread:** zero allocations, no locks, no blocking calls. Only reads from SPSC queue and applies slew limiting.
- **Worker thread:** runs on `AudioWorkerThread` (existing), joined to macOS audio workgroup for scheduling priority.
- **Fallback:** if no fresh result available or confidence below threshold, audio thread uses last-known-good values. Watchdog logs if worker falls behind.

### Location

- **Header:** `include/penta/ml/AudioEmotionRunner.h`
- **Implementation:** `src/ml/AudioEmotionRunner.cpp`
- **Namespace:** `penta::ml`

### Dependencies

- `ONNXInference` (existing, `src/ml/ONNXInference.h`) — ONNX model loading and inference
- `moodycamel::ReaderWriterQueue` (existing, fetched via CMake) — SPSC queues for samples and results
- `AudioWorkerThread` (existing, `include/penta/rt/AudioWorkerThread.h`) — worker thread with workgroup support
- `RTState` (existing, `include/penta/common/RTState.h`) — atomic parameter snapshot
- `IntentIR.h` (existing, `include/kmidi/IntentIR.h`) — `EmotionState` reference definition

## Data Flow

### Pipeline Stages

1. **Ring buffer accumulation** — Audio thread pushes float samples into SPSC ring (capacity ~524288 samples = ~10.9s at 48kHz). Worker drains when sufficient samples accumulated.

2. **Mel spectrogram** — Worker computes 128-bin log-mel spectrogram. Fixed output shape: `(1, 1, 128, 512)` matching the ONNX model. Sliding window: new samples shift the analysis window forward.

3. **ONNX inference** — Feeds mel into `ONNXInference::infer()` with pre-allocated buffers. Output: `(1, 512, 256)` latent sequence.

4. **Latent pooling** — Mean-pool over time dimension: `(1, 512, 256)` → `(256,)`.

5. **Emotion mapping** — Pooled latent → `EmotionResult` (valence, arousal, dominance, confidence). Small linear projection with pre-allocated weight matrix. Initially hand-tuned or trained offline for the demo.

6. **DSP mapping** — `EmotionResult` → `DSPSuggestion` (filter_cutoff, reverb_wet, drive_amount). Simple affine mapping, tunable via preset bank.

### Output Structs

```cpp
struct EmotionResult {
    float valence;      // [-1, 1]
    float arousal;      // [0, 1]
    float dominance;    // [0, 1]
    float confidence;   // [0, 1]
};

struct DSPSuggestion {
    float filter_cutoff;  // [0, 1] normalized
    float reverb_wet;     // [0, 1]
    float drive_amount;   // [0, 1]
};

struct EmotionRunnerResult {
    EmotionResult emotion;
    DSPSuggestion dsp;
    uint64_t sequence_id;   // monotonic, for staleness detection
};
```

### Slew Limiting

One-pole exponential smoothing per DSP parameter in `updateParams()`. Coefficient derived from `slew_time_ms` and block size. Runs on audio thread — pure arithmetic on pre-allocated state, no allocations.

## Public API

```cpp
namespace penta::ml {

class AudioEmotionRunner {
public:
    struct Config {
        std::string model_path;            // Path to .onnx file
        size_t sample_rate = 48000;        // Audio sample rate
        size_t ring_capacity = 524288;     // ~10.9s at 48kHz
        float slew_time_ms = 20.0f;       // Per-parameter ramp
        float watchdog_timeout_ms = 100;   // Max staleness before fallback
        float confidence_threshold = 0.3f; // Below this, hold last-known-good
    };

    // Non-RT: call before audio starts
    bool initialize(const Config& config);
    void shutdown();

    // Audio thread (RT-safe, noexcept)
    void pushSamples(const float* samples, size_t count) noexcept;
    void updateParams(penta::RTState& state) noexcept;

    // Non-RT: diagnostics
    bool isRunning() const;
    float lastInferenceMs() const;
    uint64_t lastSequenceId() const;
};

} // namespace penta::ml
```

### Lifecycle

1. **`initialize(config)`** — Allocates ring buffer, loads ONNX model via `ONNXInference`, pre-allocates mel/latent/output buffers, pre-computes mel filterbank, spawns worker thread. Called from `prepareToPlay()`.
2. **`pushSamples(samples, count)`** — Audio thread pushes each block into SPSC ring. Non-blocking. Drops samples if ring is full (increments atomic drop counter).
3. **`updateParams(state)`** — Audio thread calls each block. Drains SPSC result queue (takes latest, discards older). Applies slew limiting. Writes emotion + DSP fields to `RTState` atomics. If no fresh result or confidence below threshold, holds last-known-good.
4. **`shutdown()`** — Signals worker thread to stop, joins. Called from `releaseResources()`.

## Testing

### Unit Tests

- `EmotionResult` and `DSPSuggestion` struct layout and value clamping
- Slew limiter convergence: step input → verify exponential ramp toward target within expected time
- Mel spectrogram computation against known reference (precomputed fixture from Python)
- SPSC result queue: push from simulated worker, read from simulated audio thread

### Integration Tests

- Load real `audio_jepa_v01.onnx`, feed synthetic mel, verify output shape and non-NaN
- Full pipeline: push ~12s of sine wave → verify `EmotionRunnerResult` appears with values in valid ranges
- Watchdog: starve worker thread, verify `updateParams()` holds last-known-good and increments stale counter

### Acceptance Gates

- No XRuns at 64-sample buffer (`pushSamples` + `updateParams` combined < 100us)
- End-to-end latency: audio sample → RTState update < 30ms
- Zero heap allocations on audio thread (verified via allocation counter in test harness)

## Out of Scope

- Emotion prediction accuracy (model quality, not runner responsibility)
- Core ML inference path (deferred until coremltools supports Python 3.14)
- AU plugin packaging (Phase 4)
- Training the emotion mapping weights (demo uses hand-tuned or pre-trained projection)

## Build Integration

- Gated behind `ENABLE_ONNX_RUNTIME` CMake option (existing)
- When disabled, `AudioEmotionRunner` compiles as a no-op stub (same pattern as `ONNXInference`)
- No new external dependencies beyond what's already fetched
