#pragma once
/*
 * MultiModelProcessor.h - Five-Head RTNeural Inference Stack
 * ============================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - ML Layer: ONNXInference, RTNeuralProcessor, MLFeatureExtractor (sibling
 * paths)
 * - Engine Layer: Emotion / groove / harmony consumers in Kelly pipeline
 * - Build: ENABLE_RTNEURAL gates RTNeural weights
 *
 * Purpose: Bundled real-time models for audio→emotion and generative
 * assistants.
 *
 * Features:
 * - ModelType enum (EmotionRecognizer, MelodyTransformer, HarmonyPredictor, …)
 * - Specs for ~1M params across five models; <10 ms inference target
 */

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <juce_core/juce_core.h>
#include <memory>
#include <mutex>
#include <vector>

#ifdef ENABLE_RTNEURAL
#include <RTNeural/RTNeural.h>
#endif

namespace Kelly {
namespace ML {

// ============================================================================
// Model Type Enumeration
// ============================================================================
enum class ModelType : size_t {
  EmotionRecognizer = 0, // Audio → Emotion
  MelodyTransformer = 1, // Emotion → MIDI
  HarmonyPredictor = 2,  // Context → Chords
  DynamicsEngine = 3,    // Context → Expression
  GroovePredictor = 4,   // Emotion → Groove
  COUNT = 5
};

// ============================================================================
// Model Configuration
// ============================================================================
struct ModelSpec {
  const char *name;
  size_t inputSize;
  size_t outputSize;
  size_t estimatedParams;
};

constexpr std::array<ModelSpec, 5> MODEL_SPECS = {
    {{"EmotionRecognizer", 128, 64, 497664},
     {"MelodyTransformer", 64, 128, 412672},
     {"HarmonyPredictor", 128, 64, 74048},
     {"DynamicsEngine", 32, 16, 13456},
     {"GroovePredictor", 64, 32, 19040}}};

// ============================================================================
// Inference Result Structure
// ============================================================================
struct InferenceResult {
  std::array<float, 64> emotionEmbedding{};     // Model 0 output
  std::array<float, 128> melodyProbabilities{}; // Model 1 output
  std::array<float, 64> harmonyPrediction{};    // Model 2 output
  std::array<float, 16> dynamicsOutput{};       // Model 3 output
  std::array<float, 32> grooveParameters{};     // Model 4 output
  bool valid = false;
};

// ============================================================================
// Single Model Wrapper (Heap Allocated)
// ============================================================================
class ModelWrapper {
public:
  explicit ModelWrapper(ModelType type);
  ~ModelWrapper() = default;

  bool loadWeights(const juce::File &file);
  std::vector<float> forward(const float *input, size_t inputSize);

  bool isLoaded() const { return loaded_.load(std::memory_order_acquire); }
  void setEnabled(bool e) { enabled_.store(e, std::memory_order_release); }
  bool isEnabled() const { return enabled_.load(std::memory_order_acquire); }
  const ModelSpec &spec() const { return spec_; }

private:
  void computeFallback(const float *input);

  ModelType type_;
  ModelSpec spec_;
  std::vector<float> output_;
  // Narrow scalar flags read off-lock by control/UI threads while the
  // worker runs inference; atomics keep those reads race-free.
  std::atomic<bool> loaded_{false};
  std::atomic<bool> enabled_{true};

#ifdef ENABLE_RTNEURAL
  std::unique_ptr<RTNeural::Model<float>> rtModel_;
#endif
};

// ============================================================================
// Multi-Model Manager
// ============================================================================
class MultiModelProcessor {
public:
  MultiModelProcessor();
  ~MultiModelProcessor() = default;

  bool initialize(const juce::File &modelsDir);

  void setModelEnabled(ModelType type, bool enabled);
  bool isModelEnabled(ModelType type) const;

  // Run single model inference
  std::vector<float> infer(ModelType type, const std::vector<float> &input);

  // Run full pipeline: audio features → all outputs
  InferenceResult runFullPipeline(const std::array<float, 128> &audioFeatures);

  size_t getTotalParams() const;
  size_t getTotalMemoryKB() const;

  bool isInitialized() const {
    return initialized_.load(std::memory_order_acquire);
  }

private:
  void logStats() const;

  std::array<std::unique_ptr<ModelWrapper>, 5> models_;
  std::mutex mutex_;
  // Read off-lock (e.g. by the plugin before deciding to initialize).
  std::atomic<bool> initialized_{false};
};

// ============================================================================
// Async Inference (Lock-Free for Audio Thread)
// ============================================================================
class InferenceThread; // Forward declaration

class AsyncMLPipeline {
public:
  explicit AsyncMLPipeline(MultiModelProcessor &processor);
  ~AsyncMLPipeline();

  // Control-plane lifecycle (message/background threads only — NOT RT-safe).
  // Thread-safe and idempotent: concurrent or repeated start()/stop() calls
  // are serialized, and stop() joins the worker cooperatively (no force-kill,
  // which could orphan the processor mutex mid-inference).
  void start();
  void stop();

  // Non-blocking submit (audio thread safe)
  // Returns a request id for tracking; 0 if a request is already pending.
  uint64_t submitFeatures(const std::array<float, 128> &features);

  // Non-blocking result check (audio thread safe)
  bool hasResult() const;
  bool hasResult(uint64_t requestId) const;

  InferenceResult getResult();
  InferenceResult getResult(uint64_t requestId);

private:
  friend class InferenceThread; // Allow InferenceThread to call run()
  // Worker loop; `self` is the executing thread, polled for cooperative exit.
  void run(juce::Thread &self);

  MultiModelProcessor &processor_;
  // Concrete ownership of the worker (single owner, no downcasts at use
  // sites). InferenceThread is defined in the .cpp; all special members of
  // AsyncMLPipeline are out-of-line so the incomplete type is fine here.
  std::unique_ptr<InferenceThread> thread_;
  // Serializes start()/stop() so thread_ has exactly one ordered
  // teardown path. Never taken on the RT/submit path.
  std::mutex lifecycleMutex_;
  std::atomic<bool> running_{false};
  std::atomic<bool> hasRequest_{false};
  std::atomic<bool> hasResult_{false};
  std::atomic<uint64_t> nextRequestId_{1};
  std::atomic<uint64_t> pendingRequestId_{0};
  std::atomic<uint64_t> latestResultId_{0};
  std::array<float, 128> pendingFeatures_{};

  // Double-buffer for lock-free result handoff between worker and reader.
  // Writer fills resultBuffers_[1 - activeResultIdx_], then swaps the index.
  // Reader always reads from resultBuffers_[activeResultIdx_].
  // Depth-2 caveat: a reader copying [active] can be overtaken if two full
  // inferences complete during the copy; acceptable because the submit gate
  // (hasRequest_ || hasResult_ → reject) keeps at most one result in flight.
  std::array<InferenceResult, 2> resultBuffers_;
  std::atomic<int> activeResultIdx_{0};
};

} // namespace ML
} // namespace Kelly
