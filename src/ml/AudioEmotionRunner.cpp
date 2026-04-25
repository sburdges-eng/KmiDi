#include "penta/ml/AudioEmotionRunner.h"
#include "penta/common/Platform.h"     // T6.4: set_denormals_off helper
#include "ml/MelSpectrogram.h"
// T6.2: sampleRing switched from moodycamel::ReaderWriterQueue<float> to
// kelly::LockFreeRingBuffer<float,N> so pushSamples can use the bulk push()
// path (single memcpy per contiguous region, at most two).
#include "ml/LockFreeRingBuffer.h"

#include <readerwriterqueue.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>

// Note: SSE/NEON headers included transitively via penta/common/Platform.h (T6.4)

#ifdef ENABLE_ONNX_RUNTIME
#include "ml/ONNXInference.h"
#endif

namespace penta::ml {

// ─── Slew limiter (one-pole exponential) ────────────────────────────────────

struct SlewLimiter {
    float current = 0.0f;
    float target  = 0.0f;
    float coeff   = 0.0f;

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
        float blocksPerMs = static_cast<float>(sampleRate) /
                            (static_cast<float>(blockSize) * 1000.0f);
        float tau = slewMs * blocksPerMs;
        coeff = (tau > 0.0f) ? (1.0f - std::exp(-1.0f / tau)) : 1.0f;
    }
};

// ─── Implementation struct (PIMPL) ─────────────────────────────────────────

// T6.2: Fixed-capacity sample ring.  524288 == 2^19 satisfies the
// power-of-two requirement of LockFreeRingBuffer and matches the default
// ring_capacity in AudioEmotionRunnerConfig.
static constexpr size_t kSampleRingCapacity = 524288u;

struct AudioEmotionRunnerImpl {
    AudioEmotionRunnerConfig config;

    // T6.2: Sample ring buffer — kelly::LockFreeRingBuffer provides a bulk
    // push(data, count) that does at most two memcpys (one per contiguous
    // region), replacing the per-sample loop that existed previously.
    kelly::LockFreeRingBuffer<float, kSampleRingCapacity> sampleRing;

    // Result queue (worker thread → audio thread)
    std::unique_ptr<moodycamel::ReaderWriterQueue<EmotionRunnerResult>> resultQueue;

    // Worker thread
    std::thread workerThread;
    std::atomic<bool> running{false};

    // ONNX inference
#ifdef ENABLE_ONNX_RUNTIME
    std::unique_ptr<midikompanion::ml::ONNXInference> onnx;       // JEPA encoder
    std::unique_ptr<midikompanion::ml::ONNXInference> probeOnnx;  // Emotion probe
#endif

    // Mel spectrogram computer
    MelSpectrogram melComputer;

    // Pre-allocated buffers (worker thread only)
    std::vector<float> sampleBuffer;
    std::vector<float> melBuffer;
    std::vector<float> latentBuffer;
    std::vector<float> pooledLatent;

    // Emotion mapping weights: (4, 256) → [valence, arousal, dominance, confidence]
    std::vector<float> emotionWeights;
    std::vector<float> emotionBias;

    // DSP mapping weights: (3, 4) → [filter, reverb, drive] from [V, A, D, conf]
    std::vector<float> dspWeights;
    std::vector<float> dspBias;

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

    // T6.7: Drop-rate EMA state (non-atomic; read only from non-RT thread
    // calling dropRate()).  Protected by the caller's non-RT scheduling.
    mutable uint64_t dropRateLastSamples{0};
    mutable std::chrono::steady_clock::time_point dropRateLastTime{};
    mutable float dropRateEma{0.0f};
    mutable bool dropRateInitialized{false};

    // Accumulated sample count for the worker
    size_t samplesAccumulated = 0;

    void initDefaultWeights() {
        const size_t latentDim = 256;
        emotionWeights.assign(4 * latentDim, 0.0f);
        emotionBias = {0.0f, 0.5f, 0.5f, 0.5f};

        // Map first 4 latent dims to emotion (scaled)
        for (size_t i = 0; i < 4 && i < latentDim; ++i) {
            emotionWeights[i * latentDim + i] = 1.0f;
        }

        // Simple DSP mapping from emotion
        dspWeights.assign(3 * 4, 0.0f);
        dspBias = {0.5f, 0.2f, 0.0f};

        // filter_cutoff += 0.5 * arousal
        dspWeights[0 * 4 + 1] = 0.5f;
        // reverb_wet += -0.3 * arousal
        dspWeights[1 * 4 + 1] = -0.3f;
        // drive_amount += 0.4 * arousal
        dspWeights[2 * 4 + 1] = 0.4f;
    }

    EmotionResult mapLatentToEmotion(const float* pooled, size_t dim) {
        assert(dim <= 256 && "mapLatentToEmotion: dim exceeds allocated weight dimensions");
        assert(dim * 4 <= emotionWeights.size() && "mapLatentToEmotion: weight buffer too small for dim");

        EmotionResult e;
        float raw[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (size_t i = 0; i < 4; ++i) {
            raw[i] = emotionBias[i];
            for (size_t j = 0; j < dim; ++j) {
                raw[i] += emotionWeights[i * dim + j] * pooled[j];
            }
        }

        e.valence    = std::clamp(std::tanh(raw[0]), -1.0f, 1.0f);
        e.arousal    = std::clamp(1.0f / (1.0f + std::exp(-raw[1])), 0.0f, 1.0f);
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
        // T6.4: Suppress denormals via shared platform helper (replaces inlined
        // FTZ/DAZ code).  Per-thread state, must be set in each worker thread.
        penta::set_denormals_off();

        const size_t requiredSamples = MelSpectrogram::kRequiredSamples;
        const size_t latentDim = 256;
        const size_t latentFrames = 512;

        while (running.load(std::memory_order_relaxed)) {
            // T6.2: Drain samples from ring into accumulation buffer in bulk.
            {
                const size_t avail   = sampleRing.availableToRead();
                const size_t space   = sampleBuffer.size() - samplesAccumulated;
                const size_t toPop   = std::min(avail, space);
                if (toPop > 0) {
                    sampleRing.pop(sampleBuffer.data() + samplesAccumulated, toPop);
                    samplesAccumulated += toPop;
                    // Discard any overflow that didn't fit (ring draining).
                    // Buffer is local — `static` here would race across
                    // workerLoop instances when multiple plugins coexist
                    // in the same process (DAW with multiple inserts).
                    if (avail > toPop) {
                        const size_t overflow = avail - toPop;
                        constexpr size_t kDiscardChunk = 4096;
                        float discard[kDiscardChunk];
                        size_t remaining = overflow;
                        while (remaining > 0) {
                            const size_t chunk = std::min(remaining, kDiscardChunk);
                            sampleRing.pop(discard, chunk);
                            remaining -= chunk;
                        }
                    }
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
                // No ONNX — produce default result (zeros)
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

            // 4. Map to emotion: use probe ONNX if available, else hardcoded
            EmotionRunnerResult result;
#ifdef ENABLE_ONNX_RUNTIME
            if (probeOnnx && probeOnnx->isLoaded()) {
                float probeOut[2] = {0.0f, 0.0f};
                probeOnnx->infer(pooledLatent.data(), probeOut);
                result.emotion.valence   = std::clamp(probeOut[0], -1.0f, 1.0f);
                result.emotion.arousal   = std::clamp((probeOut[1] + 1.0f) * 0.5f, 0.0f, 1.0f);
                result.emotion.dominance = std::clamp(
                    0.5f + 0.3f * result.emotion.arousal + 0.2f * std::abs(result.emotion.valence),
                    0.0f, 1.0f);
                result.emotion.confidence = 0.8f;
            } else {
                result.emotion = mapLatentToEmotion(pooledLatent.data(), latentDim);
            }
#else
            result.emotion = mapLatentToEmotion(pooledLatent.data(), latentDim);
#endif
            result.dsp = mapEmotionToDSP(result.emotion);
            result.sequence_id = sequenceCounter.fetch_add(1, std::memory_order_relaxed);

            auto t1 = std::chrono::steady_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            lastInferenceMs.store(ms, std::memory_order_relaxed);

            // 5. Post result
            resultQueue->try_enqueue(result);

            // Shift buffer: slide by half the window for overlapping analysis
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

    // T6.2: sampleRing is now a statically-sized LockFreeRingBuffer — no heap
    // allocation here.  Clear it in case initialize() is called more than once.
    impl_->sampleRing.clear();
    impl_->resultQueue = std::make_unique<moodycamel::ReaderWriterQueue<EmotionRunnerResult>>(16);

    // Pre-allocate buffers
    impl_->sampleBuffer.resize(MelSpectrogram::kRequiredSamples + config.ring_capacity);
    impl_->melBuffer.resize(MelSpectrogram::kNMels * MelSpectrogram::kNFrames);
    impl_->latentBuffer.resize(512 * 256);
    impl_->pooledLatent.resize(256);
    impl_->samplesAccumulated = 0;

    // Init mapping weights
    impl_->initDefaultWeights();

    // Load ONNX model
#ifdef ENABLE_ONNX_RUNTIME
    impl_->onnx = std::make_unique<midikompanion::ml::ONNXInference>();
    if (!impl_->onnx->loadModel(config.model_path)) {
        impl_->onnx.reset();
    }

    if (!config.probe_model_path.empty()) {
        impl_->probeOnnx = std::make_unique<midikompanion::ml::ONNXInference>();
        if (!impl_->probeOnnx->loadModel(config.probe_model_path)) {
            impl_->probeOnnx.reset();
        }
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
    // T6.2: Bulk push — at most two memcpys (one per contiguous region).
    // On overflow, increment droppedSamples by the unsent delta in one add.
    const size_t avail = impl_->sampleRing.availableToWrite();
    if (avail == 0) {
        impl_->droppedSamples.fetch_add(count, std::memory_order_relaxed);
        return;
    }
    const size_t toWrite = std::min(count, avail);
    impl_->sampleRing.push(samples, toWrite);
    if (toWrite < count) {
        impl_->droppedSamples.fetch_add(count - toWrite, std::memory_order_relaxed);
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
        if (latest.emotion.confidence >= impl_->config.confidence_threshold) {
            impl_->lastGoodResult = latest;
            impl_->hasResult = true;
        }
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

    // Update slew coefficients
    float slewMs = impl_->config.slew_time_ms;
    size_t sr = impl_->config.sample_rate;
    impl_->slewValence.updateCoeff(slewMs, sr, blockSize);
    impl_->slewArousal.updateCoeff(slewMs, sr, blockSize);
    impl_->slewDominance.updateCoeff(slewMs, sr, blockSize);
    impl_->slewConfidence.updateCoeff(slewMs, sr, blockSize);
    impl_->slewFilterCutoff.updateCoeff(slewMs, sr, blockSize);
    impl_->slewReverbWet.updateCoeff(slewMs, sr, blockSize);
    impl_->slewDriveAmount.updateCoeff(slewMs, sr, blockSize);

    // Process slew and write to RTState.
    //
    // Seqlock: bracket the batch so readers (kelly_brain_get_rt_state,
    // kmidi_engine_get_state) never observe a half-published snapshot
    // (e.g. new valence paired with stale arousal). Any early-return
    // above this point skips the bracket entirely — begin/end must be
    // matched on this path only.
    state.begin_publish();
    state.valence.store(impl_->slewValence.process(), std::memory_order_relaxed);
    state.arousal.store(impl_->slewArousal.process(), std::memory_order_relaxed);
    state.dominance.store(impl_->slewDominance.process(), std::memory_order_relaxed);
    state.emotionConfidence.store(impl_->slewConfidence.process(), std::memory_order_relaxed);

    // DSP suggestions go into trackParams[0..2]
    state.trackParams[0].store(impl_->slewFilterCutoff.process(), std::memory_order_relaxed);
    state.trackParams[1].store(impl_->slewReverbWet.process(), std::memory_order_relaxed);
    state.trackParams[2].store(impl_->slewDriveAmount.process(), std::memory_order_relaxed);
    state.end_publish();
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

float AudioEmotionRunner::dropRate() const {
    // T6.7: EMA of dropped-samples/second.  Called from non-RT thread (UI timer).
    // alpha = 0.2 → ~5-sample window at 1 Hz polling.
    const uint64_t current = impl_->droppedSamples.load(std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();

    if (!impl_->dropRateInitialized) {
        impl_->dropRateLastSamples  = current;
        impl_->dropRateLastTime     = now;
        impl_->dropRateInitialized  = true;
        return 0.0f;
    }

    const double dtSec = std::chrono::duration<double>(now - impl_->dropRateLastTime).count();
    if (dtSec < 0.001) return impl_->dropRateEma;  // avoid divide-by-zero on rapid calls

    const uint64_t delta = current - impl_->dropRateLastSamples;
    const float instantRate = static_cast<float>(delta) / static_cast<float>(dtSec);

    constexpr float kAlpha = 0.2f;
    impl_->dropRateEma       = kAlpha * instantRate + (1.0f - kAlpha) * impl_->dropRateEma;
    impl_->dropRateLastSamples = current;
    impl_->dropRateLastTime    = now;

    return impl_->dropRateEma;
}

} // namespace penta::ml
