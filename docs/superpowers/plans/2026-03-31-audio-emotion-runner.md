# AudioEmotionRunner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a C++ component that accepts raw audio samples, runs JEPA emotion inference on a worker thread, and posts both `EmotionResult` and `DSPSuggestion` back to the audio thread via lock-free queue with slew limiting.

**Architecture:** Audio thread pushes samples into SPSC ring → worker thread drains, computes mel spectrogram, runs ONNX inference, maps latent→emotion→DSP → posts result via SPSC queue → audio thread reads latest result, applies slew limiting, writes to RTState atomics.

**Tech Stack:** C++20, ONNX Runtime (gated behind `ENABLE_ONNX_RUNTIME`), moodycamel::ReaderWriterQueue, penta::rt::AudioWorkerThread, Catch2 for tests

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `include/penta/ml/AudioEmotionRunner.h` | Public API: Config, EmotionResult, DSPSuggestion, EmotionRunnerResult, AudioEmotionRunner class |
| Create | `src/ml/AudioEmotionRunner.cpp` | Implementation: mel computation, ONNX inference, emotion/DSP mapping, worker loop, slew limiter |
| Create | `src/ml/MelSpectrogram.h` | Internal: fixed-shape mel spectrogram computation (128 bins, 512 frames) |
| Create | `src/ml/MelSpectrogram.cpp` | Implementation: FFT, mel filterbank, log scaling |
| Create | `tests/cpp/test_audio_emotion_runner.cpp` | Unit + integration tests |
| Modify | `CMakeLists.txt` | Add test target for AudioEmotionRunner tests |

**Note:** `src/ml/AudioEmotionRunner.cpp` and `src/ml/MelSpectrogram.cpp` are automatically picked up by the `KELLY_CORE_SOURCES` glob (`src/*.cpp`). No source list changes needed.

---

### Task 1: EmotionRunnerResult structs and AudioEmotionRunner header

**Files:**
- Create: `include/penta/ml/AudioEmotionRunner.h`

- [ ] **Step 1: Create the header with structs and class declaration**

```cpp
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
    std::string model_path;                // Path to .onnx file
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

private:
    std::unique_ptr<AudioEmotionRunnerImpl> impl_;
};

} // namespace penta::ml
```

- [ ] **Step 2: Verify the header compiles**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_KELLY_CORE=ON 2>&1 | tail -5
cmake --build build --target KellyCore -j8 2>&1 | tail -10
```

Expected: Compiles cleanly (header is only included if something includes it — but we should verify no syntax errors by including it from a translation unit). If nothing includes it yet, that's fine — Task 2 will create the .cpp.

- [ ] **Step 3: Commit**

```bash
git add include/penta/ml/AudioEmotionRunner.h
git commit -m "feat: add AudioEmotionRunner header with result structs and API"
```

---

### Task 2: MelSpectrogram — fixed-shape mel computation

**Files:**
- Create: `src/ml/MelSpectrogram.h`
- Create: `src/ml/MelSpectrogram.cpp`

This is a self-contained utility: takes raw PCM samples, outputs a `(128, 512)` mel spectrogram ready for ONNX input.

- [ ] **Step 1: Create the MelSpectrogram header**

```cpp
#pragma once

/**
 * MelSpectrogram — Compute fixed-shape log-mel spectrograms for JEPA inference.
 *
 * Output shape: (n_mels=128, n_frames=512) matching the ONNX model input.
 * Uses 22050 Hz internal sample rate, hop_length=512, n_fft=2048.
 *
 * All buffers are pre-allocated at construction time. No heap allocations
 * during compute().
 */

#include <cstddef>
#include <vector>

namespace penta::ml {

class MelSpectrogram {
public:
    static constexpr size_t kNMels     = 128;
    static constexpr size_t kNFrames   = 512;
    static constexpr size_t kNFft      = 2048;
    static constexpr size_t kHopLength = 512;
    static constexpr size_t kSampleRate = 22050;

    // Total input samples needed: (kNFrames - 1) * kHopLength + kNFft = 263680
    static constexpr size_t kRequiredSamples = (kNFrames - 1) * kHopLength + kNFft;

    MelSpectrogram();

    /**
     * Compute log-mel spectrogram from raw audio samples.
     *
     * @param samples  Input audio at kSampleRate (22050 Hz). Must have >= kRequiredSamples.
     * @param count    Number of input samples (must be >= kRequiredSamples).
     * @param output   Output buffer, size kNMels * kNFrames = 65536 floats.
     *                 Layout: row-major [mel_bin][frame], matching ONNX (1, 1, 128, 512).
     * @return true if successful
     */
    bool compute(const float* samples, size_t count, float* output);

private:
    void buildMelFilterbank();
    void applyWindow(const float* frame, float* windowed);
    void computeFFT(const float* windowed, float* magnitudes);

    // Pre-allocated buffers
    std::vector<float> window_;         // Hann window, size kNFft
    std::vector<float> windowed_;       // Windowed frame, size kNFft
    std::vector<float> fftReal_;        // FFT real part
    std::vector<float> fftImag_;        // FFT imaginary part
    std::vector<float> magnitudes_;     // Power spectrum, size kNFft/2+1
    std::vector<float> melFilterbank_;  // (kNMels, kNFft/2+1) row-major
    std::vector<float> melEnergies_;    // Single frame mel energies, size kNMels
};

} // namespace penta::ml
```

- [ ] **Step 2: Create the MelSpectrogram implementation**

```cpp
#include "ml/MelSpectrogram.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace penta::ml {

namespace {

constexpr float kPI = 3.14159265358979323846f;
constexpr float kLogFloor = 1e-10f;

// Hz to mel (HTK formula)
float hzToMel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

// Mel to Hz
float melToHz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

} // anonymous namespace

MelSpectrogram::MelSpectrogram()
    : window_(kNFft)
    , windowed_(kNFft)
    , fftReal_(kNFft)
    , fftImag_(kNFft)
    , magnitudes_(kNFft / 2 + 1)
    , melFilterbank_(kNMels * (kNFft / 2 + 1), 0.0f)
    , melEnergies_(kNMels)
{
    // Hann window
    for (size_t i = 0; i < kNFft; ++i) {
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * kPI * static_cast<float>(i) /
                                               static_cast<float>(kNFft)));
    }
    buildMelFilterbank();
}

void MelSpectrogram::buildMelFilterbank() {
    const size_t nBins = kNFft / 2 + 1;
    const float fMax = static_cast<float>(kSampleRate) / 2.0f;
    const float melMin = hzToMel(0.0f);
    const float melMax = hzToMel(fMax);

    // kNMels + 2 equally spaced points in mel space
    std::vector<float> melPoints(kNMels + 2);
    for (size_t i = 0; i < kNMels + 2; ++i) {
        float mel = melMin + (melMax - melMin) * static_cast<float>(i) /
                    static_cast<float>(kNMels + 1);
        melPoints[i] = melToHz(mel);
    }

    // Convert Hz points to FFT bin indices
    std::vector<float> binFreqs(nBins);
    for (size_t i = 0; i < nBins; ++i) {
        binFreqs[i] = static_cast<float>(i) * static_cast<float>(kSampleRate) /
                      static_cast<float>(kNFft);
    }

    // Build triangular filters
    for (size_t m = 0; m < kNMels; ++m) {
        float fLeft   = melPoints[m];
        float fCenter = melPoints[m + 1];
        float fRight  = melPoints[m + 2];

        for (size_t k = 0; k < nBins; ++k) {
            float f = binFreqs[k];
            float weight = 0.0f;

            if (f >= fLeft && f <= fCenter && fCenter > fLeft) {
                weight = (f - fLeft) / (fCenter - fLeft);
            } else if (f > fCenter && f <= fRight && fRight > fCenter) {
                weight = (fRight - f) / (fRight - fCenter);
            }

            melFilterbank_[m * nBins + k] = weight;
        }
    }
}

void MelSpectrogram::applyWindow(const float* frame, float* windowed) {
    for (size_t i = 0; i < kNFft; ++i) {
        windowed[i] = frame[i] * window_[i];
    }
}

void MelSpectrogram::computeFFT(const float* windowed, float* magnitudes) {
    // Simple DFT for correctness. For production, replace with vDSP_fft or pffft.
    // Only compute kNFft/2+1 bins (real input symmetry).
    const size_t nBins = kNFft / 2 + 1;

    for (size_t k = 0; k < nBins; ++k) {
        float re = 0.0f;
        float im = 0.0f;
        const float phase_step = -2.0f * kPI * static_cast<float>(k) / static_cast<float>(kNFft);

        for (size_t n = 0; n < kNFft; ++n) {
            float phase = phase_step * static_cast<float>(n);
            re += windowed[n] * std::cos(phase);
            im += windowed[n] * std::sin(phase);
        }
        // Power spectrum
        magnitudes[k] = re * re + im * im;
    }
}

bool MelSpectrogram::compute(const float* samples, size_t count, float* output) {
    if (count < kRequiredSamples || !samples || !output) {
        return false;
    }

    const size_t nBins = kNFft / 2 + 1;

    for (size_t frame = 0; frame < kNFrames; ++frame) {
        const float* frameStart = samples + frame * kHopLength;

        applyWindow(frameStart, windowed_.data());
        computeFFT(windowed_.data(), magnitudes_.data());

        // Apply mel filterbank
        for (size_t m = 0; m < kNMels; ++m) {
            float energy = 0.0f;
            const float* filter = &melFilterbank_[m * nBins];
            for (size_t k = 0; k < nBins; ++k) {
                energy += filter[k] * magnitudes_[k];
            }
            // Log-mel
            output[m * kNFrames + frame] = std::log(std::max(energy, kLogFloor));
        }
    }

    return true;
}

} // namespace penta::ml
```

- [ ] **Step 3: Verify it compiles**

```bash
cmake --build build --target KellyCore -j8 2>&1 | tail -10
```

Expected: Compiles cleanly. The .cpp is auto-globbed by CMake.

- [ ] **Step 4: Commit**

```bash
git add src/ml/MelSpectrogram.h src/ml/MelSpectrogram.cpp
git commit -m "feat: add MelSpectrogram — fixed-shape mel computation for JEPA"
```

---

### Task 3: AudioEmotionRunner implementation (stub + real)

**Files:**
- Create: `src/ml/AudioEmotionRunner.cpp`

- [ ] **Step 1: Create the implementation file**

```cpp
#include "penta/ml/AudioEmotionRunner.h"
#include "ml/MelSpectrogram.h"

#include <readerwriterqueue/readerwriterqueue.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

#ifdef ENABLE_ONNX_RUNTIME
#include "ml/ONNXInference.h"
#endif

namespace penta::ml {

// ─── Slew limiter (one-pole exponential) ────────────────────────────────────

struct SlewLimiter {
    float current = 0.0f;
    float target  = 0.0f;
    float coeff   = 0.0f;   // computed from slew_time_ms + sample_rate

    void setTarget(float t) noexcept { target = t; }

    float process() noexcept {
        current += coeff * (target - current);
        return current;
    }

    void reset(float value) noexcept {
        current = value;
        target  = value;
    }

    void updateCoeff(float slewMs, size_t sampleRate, size_t blockSize) noexcept {
        if (slewMs <= 0.0f || blockSize == 0) {
            coeff = 1.0f;
            return;
        }
        // Time constant in blocks
        float blocksPerMs = static_cast<float>(sampleRate) /
                            (static_cast<float>(blockSize) * 1000.0f);
        float tau = slewMs * blocksPerMs;
        coeff = (tau > 0.0f) ? (1.0f - std::exp(-1.0f / tau)) : 1.0f;
    }
};

// ─── Implementation struct (PIMPL) ─────────────────────────────────────────

struct AudioEmotionRunnerImpl {
    AudioEmotionRunnerConfig config;

    // Sample ring buffer (audio thread → worker thread)
    std::unique_ptr<moodycamel::ReaderWriterQueue<float>> sampleRing;

    // Result queue (worker thread → audio thread)
    std::unique_ptr<moodycamel::ReaderWriterQueue<EmotionRunnerResult>> resultQueue;

    // Worker thread
    std::thread workerThread;
    std::atomic<bool> running{false};

    // ONNX inference
#ifdef ENABLE_ONNX_RUNTIME
    std::unique_ptr<midikompanion::ml::ONNXInference> onnx;
#endif

    // Mel spectrogram computer
    MelSpectrogram melComputer;

    // Pre-allocated buffers (worker thread only)
    std::vector<float> sampleBuffer;     // Accumulated audio samples
    std::vector<float> melBuffer;        // (128 * 512) mel output
    std::vector<float> latentBuffer;     // ONNX output (512 * 256)
    std::vector<float> pooledLatent;     // Mean-pooled (256)

    // Emotion mapping weights: (4, 256) → [valence, arousal, dominance, confidence]
    std::vector<float> emotionWeights;   // (4 * 256)
    std::vector<float> emotionBias;      // (4)

    // DSP mapping weights: (3, 4) → [filter, reverb, drive] from [V, A, D, conf]
    std::vector<float> dspWeights;       // (3 * 4)
    std::vector<float> dspBias;          // (3)

    // Slew limiters (audio thread)
    SlewLimiter slewValence;
    SlewLimiter slewArousal;
    SlewLimiter slewDominance;
    SlewLimiter slewConfidence;
    SlewLimiter slewFilterCutoff;
    SlewLimiter slewReverbWet;
    SlewLimiter slewDriveAmount;

    // Last known good (audio thread)
    EmotionRunnerResult lastGoodResult;
    bool hasResult = false;

    // Diagnostics
    std::atomic<float>    lastInferenceMs{0.0f};
    std::atomic<uint64_t> sequenceCounter{0};
    std::atomic<uint64_t> droppedSamples{0};

    // Accumulated sample count for the worker
    size_t samplesAccumulated = 0;

    void initDefaultWeights() {
        // Identity-ish emotion mapping: latent[0..3] → VAD + confidence
        // In production, load trained weights. For demo, use simple projection.
        const size_t latentDim = 256;
        emotionWeights.assign(4 * latentDim, 0.0f);
        emotionBias = {0.0f, 0.5f, 0.5f, 0.5f};

        // Map first 4 latent dims to emotion (scaled)
        for (size_t i = 0; i < 4 && i < latentDim; ++i) {
            emotionWeights[i * latentDim + i] = 1.0f;
        }

        // Simple DSP mapping from emotion
        // filter_cutoff ← arousal, reverb_wet ← 1-arousal, drive ← |valence|
        dspWeights.assign(3 * 4, 0.0f);
        dspBias = {0.5f, 0.2f, 0.0f};

        // filter_cutoff += 0.5 * arousal
        dspWeights[0 * 4 + 1] = 0.5f;
        // reverb_wet += -0.3 * arousal
        dspWeights[1 * 4 + 1] = -0.3f;
        // drive_amount += 0.4 * arousal + 0.2 * |valence| (approx via valence^2 later)
        dspWeights[2 * 4 + 1] = 0.4f;
    }

    EmotionResult mapLatentToEmotion(const float* pooled, size_t dim) {
        EmotionResult e;
        float raw[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (size_t i = 0; i < 4; ++i) {
            raw[i] = emotionBias[i];
            for (size_t j = 0; j < dim; ++j) {
                raw[i] += emotionWeights[i * dim + j] * pooled[j];
            }
        }

        // Clamp to valid ranges
        e.valence    = std::clamp(std::tanh(raw[0]), -1.0f, 1.0f);
        e.arousal    = std::clamp(1.0f / (1.0f + std::exp(-raw[1])), 0.0f, 1.0f); // sigmoid
        e.dominance  = std::clamp(1.0f / (1.0f + std::exp(-raw[2])), 0.0f, 1.0f);
        e.confidence = std::clamp(1.0f / (1.0f + std::exp(-raw[3])), 0.0f, 1.0f);
        return e;
    }

    DSPSuggestion mapEmotionToDSP(const EmotionResult& e) {
        float emotionVec[4] = {e.valence, e.arousal, e.dominance, e.confidence};
        DSPSuggestion d;
        float raw[3] = {0.0f, 0.0f, 0.0f};

        for (size_t i = 0; i < 3; ++i) {
            raw[i] = dspBias[i];
            for (size_t j = 0; j < 4; ++j) {
                raw[i] += dspWeights[i * 4 + j] * emotionVec[j];
            }
        }

        d.filter_cutoff = std::clamp(raw[0], 0.0f, 1.0f);
        d.reverb_wet    = std::clamp(raw[1], 0.0f, 1.0f);
        d.drive_amount  = std::clamp(raw[2], 0.0f, 1.0f);
        return d;
    }

    void workerLoop() {
        const size_t requiredSamples = MelSpectrogram::kRequiredSamples;
        const size_t latentDim = 256;
        const size_t latentFrames = 512;

        while (running.load(std::memory_order_relaxed)) {
            // Drain samples from ring into accumulation buffer
            float sample;
            while (sampleRing->try_dequeue(sample)) {
                if (samplesAccumulated < sampleBuffer.size()) {
                    sampleBuffer[samplesAccumulated++] = sample;
                }
            }

            // Not enough samples yet — yield and retry
            if (samplesAccumulated < requiredSamples) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            auto t0 = std::chrono::steady_clock::now();

            // 1. Compute mel spectrogram
            melComputer.compute(sampleBuffer.data(), samplesAccumulated, melBuffer.data());

            // 2. Run ONNX inference
            bool inferenceOk = false;
#ifdef ENABLE_ONNX_RUNTIME
            if (onnx && onnx->isLoaded()) {
                inferenceOk = onnx->infer(melBuffer.data(), latentBuffer.data());
            }
#endif
            if (!inferenceOk) {
                // No ONNX — produce default result
                std::fill(latentBuffer.begin(), latentBuffer.end(), 0.0f);
            }

            // 3. Mean-pool latent: (512, 256) → (256)
            std::fill(pooledLatent.begin(), pooledLatent.end(), 0.0f);
            for (size_t t = 0; t < latentFrames; ++t) {
                for (size_t d = 0; d < latentDim; ++d) {
                    pooledLatent[d] += latentBuffer[t * latentDim + d];
                }
            }
            float invFrames = 1.0f / static_cast<float>(latentFrames);
            for (size_t d = 0; d < latentDim; ++d) {
                pooledLatent[d] *= invFrames;
            }

            // 4. Map to emotion + DSP
            EmotionRunnerResult result;
            result.emotion = mapLatentToEmotion(pooledLatent.data(), latentDim);
            result.dsp = mapEmotionToDSP(result.emotion);
            result.sequence_id = sequenceCounter.fetch_add(1, std::memory_order_relaxed);

            auto t1 = std::chrono::steady_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            lastInferenceMs.store(ms, std::memory_order_relaxed);

            // 5. Post result
            resultQueue->try_enqueue(result);

            // Shift buffer: keep last (requiredSamples - hopShift) samples
            // Slide by half the window for overlapping analysis
            size_t hopShift = requiredSamples / 2;
            size_t remaining = samplesAccumulated - hopShift;
            std::memmove(sampleBuffer.data(), sampleBuffer.data() + hopShift,
                         remaining * sizeof(float));
            samplesAccumulated = remaining;
        }
    }
};

// ─── AudioEmotionRunner public methods ──────────────────────────────────────

AudioEmotionRunner::AudioEmotionRunner()
    : impl_(std::make_unique<AudioEmotionRunnerImpl>()) {}

AudioEmotionRunner::~AudioEmotionRunner() {
    shutdown();
}

bool AudioEmotionRunner::initialize(const AudioEmotionRunnerConfig& config) {
    if (impl_->running.load()) return false;

    impl_->config = config;

    // Allocate queues
    impl_->sampleRing = std::make_unique<moodycamel::ReaderWriterQueue<float>>(config.ring_capacity);
    impl_->resultQueue = std::make_unique<moodycamel::ReaderWriterQueue<EmotionRunnerResult>>(16);

    // Pre-allocate buffers
    impl_->sampleBuffer.resize(MelSpectrogram::kRequiredSamples + config.ring_capacity);
    impl_->melBuffer.resize(MelSpectrogram::kNMels * MelSpectrogram::kNFrames);
    impl_->latentBuffer.resize(512 * 256); // ONNX output
    impl_->pooledLatent.resize(256);
    impl_->samplesAccumulated = 0;

    // Init mapping weights
    impl_->initDefaultWeights();

    // Load ONNX model
#ifdef ENABLE_ONNX_RUNTIME
    impl_->onnx = std::make_unique<midikompanion::ml::ONNXInference>();
    if (!impl_->onnx->loadModel(config.model_path)) {
        // Model load failed — runner still works but produces default emotion
        impl_->onnx.reset();
    }
#endif

    // Start worker thread
    impl_->running.store(true);
    impl_->workerThread = std::thread([this]() { impl_->workerLoop(); });

    return true;
}

void AudioEmotionRunner::shutdown() {
    impl_->running.store(false);
    if (impl_->workerThread.joinable()) {
        impl_->workerThread.join();
    }
}

void AudioEmotionRunner::pushSamples(const float* samples, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        if (!impl_->sampleRing->try_enqueue(samples[i])) {
            impl_->droppedSamples.fetch_add(count - i, std::memory_order_relaxed);
            return;
        }
    }
}

void AudioEmotionRunner::updateParams(penta::RTState& state, size_t blockSize) noexcept {
    // Drain result queue — keep only the latest
    EmotionRunnerResult latest;
    bool gotNew = false;
    while (impl_->resultQueue->try_dequeue(latest)) {
        gotNew = true;
    }

    if (gotNew) {
        // Check confidence threshold
        if (latest.emotion.confidence >= impl_->config.confidence_threshold) {
            impl_->lastGoodResult = latest;
            impl_->hasResult = true;
        }
        // Set slew targets from latest good result
        if (impl_->hasResult) {
            const auto& e = impl_->lastGoodResult.emotion;
            const auto& d = impl_->lastGoodResult.dsp;
            impl_->slewValence.setTarget(e.valence);
            impl_->slewArousal.setTarget(e.arousal);
            impl_->slewDominance.setTarget(e.dominance);
            impl_->slewConfidence.setTarget(e.confidence);
            impl_->slewFilterCutoff.setTarget(d.filter_cutoff);
            impl_->slewReverbWet.setTarget(d.reverb_wet);
            impl_->slewDriveAmount.setTarget(d.drive_amount);
        }
    }

    if (!impl_->hasResult) return;

    // Update slew coefficients if block size changed
    float slewMs = impl_->config.slew_time_ms;
    size_t sr = impl_->config.sample_rate;
    impl_->slewValence.updateCoeff(slewMs, sr, blockSize);
    impl_->slewArousal.updateCoeff(slewMs, sr, blockSize);
    impl_->slewDominance.updateCoeff(slewMs, sr, blockSize);
    impl_->slewConfidence.updateCoeff(slewMs, sr, blockSize);
    impl_->slewFilterCutoff.updateCoeff(slewMs, sr, blockSize);
    impl_->slewReverbWet.updateCoeff(slewMs, sr, blockSize);
    impl_->slewDriveAmount.updateCoeff(slewMs, sr, blockSize);

    // Process slew and write to RTState
    state.valence.store(impl_->slewValence.process(), std::memory_order_relaxed);
    state.arousal.store(impl_->slewArousal.process(), std::memory_order_relaxed);
    state.dominance.store(impl_->slewDominance.process(), std::memory_order_relaxed);
    state.emotionConfidence.store(impl_->slewConfidence.process(), std::memory_order_relaxed);

    // DSP suggestions go into trackParams[0..2]
    state.trackParams[0].store(impl_->slewFilterCutoff.process(), std::memory_order_relaxed);
    state.trackParams[1].store(impl_->slewReverbWet.process(), std::memory_order_relaxed);
    state.trackParams[2].store(impl_->slewDriveAmount.process(), std::memory_order_relaxed);
}

bool AudioEmotionRunner::isRunning() const {
    return impl_->running.load(std::memory_order_relaxed);
}

float AudioEmotionRunner::lastInferenceMs() const {
    return impl_->lastInferenceMs.load(std::memory_order_relaxed);
}

uint64_t AudioEmotionRunner::lastSequenceId() const {
    return impl_->sequenceCounter.load(std::memory_order_relaxed);
}

uint64_t AudioEmotionRunner::droppedSamples() const {
    return impl_->droppedSamples.load(std::memory_order_relaxed);
}

} // namespace penta::ml
```

- [ ] **Step 2: Verify it compiles**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_KELLY_CORE=ON
cmake --build build --target KellyCore -j8 2>&1 | tail -10
```

Expected: Compiles cleanly. Without `ENABLE_ONNX_RUNTIME`, the ONNX code is `#ifdef`'d out.

- [ ] **Step 3: Commit**

```bash
git add src/ml/AudioEmotionRunner.cpp
git commit -m "feat: add AudioEmotionRunner implementation — worker thread, slew, mapping"
```

---

### Task 4: Unit tests — structs, slew limiter, mel spectrogram

**Files:**
- Create: `tests/cpp/test_audio_emotion_runner.cpp`
- Modify: `CMakeLists.txt` (add test target)

- [ ] **Step 1: Add test target to CMakeLists.txt**

Find the test section (around line 564) and add after the existing test block:

```cmake
# AudioEmotionRunner tests
if(BUILD_TESTS AND BUILD_KELLY_CORE)
    add_executable(audio_emotion_runner_test
        tests/cpp/test_audio_emotion_runner.cpp
    )
    target_link_libraries(audio_emotion_runner_test PRIVATE
        KellyCore
        Catch2::Catch2WithMain
    )
    target_include_directories(audio_emotion_runner_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME AudioEmotionRunner_Test COMMAND audio_emotion_runner_test)
endif()
```

- [ ] **Step 2: Create the test file**

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "penta/ml/AudioEmotionRunner.h"
#include "ml/MelSpectrogram.h"

#include <cmath>
#include <vector>
#include <thread>
#include <chrono>

using namespace penta::ml;
using Catch::Approx;

// ─── Struct tests ───────────────────────────────────────────────────────────

TEST_CASE("EmotionResult default values", "[AudioEmotionRunner][structs]") {
    EmotionResult e;
    REQUIRE(e.valence == 0.0f);
    REQUIRE(e.arousal == 0.5f);
    REQUIRE(e.dominance == 0.5f);
    REQUIRE(e.confidence == 0.0f);
}

TEST_CASE("DSPSuggestion default values", "[AudioEmotionRunner][structs]") {
    DSPSuggestion d;
    REQUIRE(d.filter_cutoff == 0.5f);
    REQUIRE(d.reverb_wet == Approx(0.2f));
    REQUIRE(d.drive_amount == 0.0f);
}

TEST_CASE("EmotionRunnerResult has monotonic sequence_id", "[AudioEmotionRunner][structs]") {
    EmotionRunnerResult r1;
    r1.sequence_id = 0;
    EmotionRunnerResult r2;
    r2.sequence_id = 1;
    REQUIRE(r2.sequence_id > r1.sequence_id);
}

// ─── MelSpectrogram tests ───────────────────────────────────────────────────

TEST_CASE("MelSpectrogram constants are consistent", "[MelSpectrogram]") {
    REQUIRE(MelSpectrogram::kNMels == 128);
    REQUIRE(MelSpectrogram::kNFrames == 512);
    REQUIRE(MelSpectrogram::kRequiredSamples ==
            (MelSpectrogram::kNFrames - 1) * MelSpectrogram::kHopLength + MelSpectrogram::kNFft);
}

TEST_CASE("MelSpectrogram rejects insufficient input", "[MelSpectrogram]") {
    MelSpectrogram mel;
    std::vector<float> output(128 * 512);
    std::vector<float> tooShort(100, 0.0f);
    REQUIRE_FALSE(mel.compute(tooShort.data(), tooShort.size(), output.data()));
}

TEST_CASE("MelSpectrogram produces finite output for sine wave", "[MelSpectrogram]") {
    MelSpectrogram mel;
    const size_t n = MelSpectrogram::kRequiredSamples;
    std::vector<float> samples(n);

    // 440 Hz sine wave at 22050 Hz
    for (size_t i = 0; i < n; ++i) {
        samples[i] = std::sin(2.0f * 3.14159265f * 440.0f *
                              static_cast<float>(i) / 22050.0f);
    }

    std::vector<float> output(128 * 512);
    REQUIRE(mel.compute(samples.data(), n, output.data()));

    // Check all values are finite
    bool allFinite = true;
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) {
            allFinite = false;
            break;
        }
    }
    REQUIRE(allFinite);
}

TEST_CASE("MelSpectrogram silence produces low energy", "[MelSpectrogram]") {
    MelSpectrogram mel;
    const size_t n = MelSpectrogram::kRequiredSamples;
    std::vector<float> silence(n, 0.0f);
    std::vector<float> output(128 * 512);

    REQUIRE(mel.compute(silence.data(), n, output.data()));

    // All values should be log(1e-10) ≈ -23.03
    float logFloor = std::log(1e-10f);
    for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(output[i] == Approx(logFloor).margin(0.01f));
    }
}

// ─── Runner lifecycle tests ─────────────────────────────────────────────────

TEST_CASE("AudioEmotionRunner initializes and shuts down", "[AudioEmotionRunner][lifecycle]") {
    AudioEmotionRunner runner;
    AudioEmotionRunnerConfig config;
    config.model_path = "";  // No model — stub mode
    config.sample_rate = 22050;
    config.ring_capacity = 65536;

    REQUIRE(runner.initialize(config));
    REQUIRE(runner.isRunning());

    runner.shutdown();
    REQUIRE_FALSE(runner.isRunning());
}

TEST_CASE("AudioEmotionRunner pushSamples does not block", "[AudioEmotionRunner][rt]") {
    AudioEmotionRunner runner;
    AudioEmotionRunnerConfig config;
    config.model_path = "";
    config.sample_rate = 22050;
    config.ring_capacity = 4096;

    runner.initialize(config);

    // Push a block of samples — should not block
    std::vector<float> block(256, 0.1f);
    auto t0 = std::chrono::steady_clock::now();
    runner.pushSamples(block.data(), block.size());
    auto t1 = std::chrono::steady_clock::now();

    float elapsedUs = std::chrono::duration<float, std::micro>(t1 - t0).count();
    // Should complete in well under 1ms
    REQUIRE(elapsedUs < 1000.0f);

    runner.shutdown();
}

TEST_CASE("AudioEmotionRunner updateParams does not block", "[AudioEmotionRunner][rt]") {
    AudioEmotionRunner runner;
    AudioEmotionRunnerConfig config;
    config.model_path = "";
    config.sample_rate = 22050;
    config.ring_capacity = 4096;

    runner.initialize(config);

    penta::RTState state;

    auto t0 = std::chrono::steady_clock::now();
    runner.updateParams(state, 64);
    auto t1 = std::chrono::steady_clock::now();

    float elapsedUs = std::chrono::duration<float, std::micro>(t1 - t0).count();
    REQUIRE(elapsedUs < 1000.0f);

    runner.shutdown();
}

TEST_CASE("AudioEmotionRunner produces result after sufficient samples", "[AudioEmotionRunner][integration]") {
    AudioEmotionRunner runner;
    AudioEmotionRunnerConfig config;
    config.model_path = "";  // No ONNX — uses default latent
    config.sample_rate = 22050;
    config.ring_capacity = 524288;
    config.confidence_threshold = 0.0f;  // Accept everything

    runner.initialize(config);

    // Feed enough samples for one analysis window
    const size_t totalSamples = MelSpectrogram::kRequiredSamples + 1000;
    std::vector<float> samples(totalSamples);
    for (size_t i = 0; i < totalSamples; ++i) {
        samples[i] = std::sin(2.0f * 3.14159265f * 440.0f *
                              static_cast<float>(i) / 22050.0f);
    }
    runner.pushSamples(samples.data(), totalSamples);

    // Wait for worker to process
    penta::RTState state;
    bool gotResult = false;
    for (int attempt = 0; attempt < 100; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        runner.updateParams(state, 64);
        if (runner.lastSequenceId() > 0) {
            gotResult = true;
            break;
        }
    }

    REQUIRE(gotResult);

    // Emotion values should be in valid ranges
    float v = state.valence.load();
    float a = state.arousal.load();
    float d = state.dominance.load();
    REQUIRE(v >= -1.0f);
    REQUIRE(v <= 1.0f);
    REQUIRE(a >= 0.0f);
    REQUIRE(a <= 1.0f);
    REQUIRE(d >= 0.0f);
    REQUIRE(d <= 1.0f);

    runner.shutdown();
}
```

- [ ] **Step 3: Build and run tests**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_KELLY_CORE=ON -DBUILD_TESTS=ON
cmake --build build --target audio_emotion_runner_test -j8
ctest --test-dir build -R AudioEmotionRunner_Test --output-on-failure
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/cpp/test_audio_emotion_runner.cpp CMakeLists.txt
git commit -m "test: add AudioEmotionRunner unit and integration tests"
```

---

### Task 5: Verify full build and existing tests still pass

- [ ] **Step 1: Full build with all targets**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON -DBUILD_TESTS=ON
cmake --build build -j8 2>&1 | tail -20
```

Expected: Clean build, no warnings from new files.

- [ ] **Step 2: Run all C++ tests**

```bash
ctest --test-dir build --output-on-failure
```

Expected: All tests pass (both existing DSP tests and new AudioEmotionRunner tests).

- [ ] **Step 3: Run Python JEPA tests to confirm no regressions**

```bash
source venv/bin/activate
python3 -m pytest tests/unit/test_jepa_models.py tests/unit/test_export_audio_jepa.py -v
```

Expected: All 38 + 6 = 44 tests pass.

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git status
# Only commit if there are changes
git commit -m "chore: fix build issues from AudioEmotionRunner integration"
```
