#pragma once
/*
 * Kelly MIDI Companion - Multi-Model ML Processor
 * ================================================
 * 5-Model Architecture (~1M params, ~4MB, <10ms inference):
 *   1. EmotionRecognizer:  128→512→256→128→64  (~500K)
 *   2. MelodyTransformer:  64→256→256→256→128  (~400K)
 *   3. HarmonyPredictor:   128→256→128→64      (~100K)
 *   4. DynamicsEngine:     32→128→64→16        (~20K)
 *   5. GroovePredictor:    64→128→64→32        (~25K)
 */

#include <juce_core/juce_core.h>
#include "ml/ONNXInference.h"
#include <array>
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <cmath>
#include <cstdint>

#ifdef ENABLE_RTNEURAL
#include <RTNeural/RTNeural.h>
#endif

namespace Kelly {
namespace ML {

// ============================================================================
// Model Type Enumeration
// ============================================================================
enum class ModelType : size_t {
    EmotionRecognizer = 0,  // Audio → Emotion
    MelodyTransformer = 1,  // Emotion → MIDI
    HarmonyPredictor  = 2,  // Context → Chords
    DynamicsEngine    = 3,  // Context → Expression
    GroovePredictor   = 4,  // Emotion → Groove
    AudioJEPA         = 5,  // Mel → Latent (Self-Supervised)
    ChordJEPA         = 6,  // MIDI → Latent (Self-Supervised)
    LanguageModel     = 7,  // Text/Tokens → Decision (LLM)
    COUNT             = 8
};

// ============================================================================
// Model Configuration
// ============================================================================
struct ModelSpec {
    const char* name;
    size_t inputSize;
    size_t outputSize;
    size_t estimatedParams;
};

constexpr std::array<ModelSpec, 8> MODEL_SPECS = {{
    {"EmotionRecognizer", 128, 64,  497664},
    {"MelodyTransformer", 64,  128, 412672},
    {"HarmonyPredictor",  128, 64,  74048},
    {"DynamicsEngine",    32,  16,  13456},
    {"GroovePredictor",   64,  32,  19040},
    {"AudioJEPA",         256, 128, 800000}, // Placeholder sizes
    {"ChordJEPA",         128, 64,  300000},
    {"LanguageModel",     512, 128, 1500000} // LLaMA-style small
}};

// ============================================================================
// Inference Result Structure
// ============================================================================
struct InferenceResult {
    std::array<float, 64>  emotionEmbedding{};    // Model 0 output
    std::array<float, 128> melodyProbabilities{}; // Model 1 output
    std::array<float, 64>  harmonyPrediction{};   // Model 2 output
    std::array<float, 16>  dynamicsOutput{};      // Model 3 output
    std::array<float, 32>  grooveParameters{};    // Model 4 output
    std::array<float, 128> audioJepaLatent{};     // Model 5 output
    std::array<float, 64>  chordJepaLatent{};     // Model 6 output
    std::array<float, 128> llmDecision{};         // Model 7 output
    std::array<bool, static_cast<size_t>(ModelType::COUNT)> usedFallback{};
    bool valid = false;
};

// ============================================================================
// Single Model Wrapper (Heap Allocated)
// ============================================================================
class ModelWrapper {
public:
    explicit ModelWrapper(ModelType type);
    ~ModelWrapper() = default;

    bool loadWeights(const juce::File& file);
    std::vector<float> forward(const float* input, size_t inputSize);

    bool isLoaded() const { return loaded_; }
    bool isUsingFallback() const { return usingFallback_; }
    void setEnabled(bool e) { enabled_ = e; }
    bool isEnabled() const { return enabled_; }
    const ModelSpec& spec() const { return spec_; }
    juce::String getStatusSummary() const;

private:
    void computeFallback(const float* input);
    bool loadOnnxModel(const juce::File& file);
    bool loadConfiguredOnnxModel(const juce::File& file);
    void setFallbackState(const juce::String& reason, const juce::File& file);
    void sanitizeOutput();

    ModelType type_;
    ModelSpec spec_;
    std::vector<float> output_;
    bool loaded_ = false;
    bool enabled_ = true;
    bool usingFallback_ = true;
    juce::String modelSource_;
    juce::String statusMessage_;
    std::unique_ptr<midikompanion::ml::ONNXInference> onnxModel_;

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

    bool initialize(const juce::File& modelsDir);

    void setModelEnabled(ModelType type, bool enabled);
    bool isModelEnabled(ModelType type) const;

    // Run single model inference
    std::vector<float> infer(ModelType type, const std::vector<float>& input);

    // Run full pipeline: audio features → all outputs
    InferenceResult runFullPipeline(const std::array<float, 128>& audioFeatures);

    size_t getTotalParams() const;
    size_t getTotalMemoryKB() const;

    bool isInitialized() const { return initialized_; }

private:
    juce::File resolveModelFile(const juce::File& modelsDir, ModelType type) const;
    bool loadRegistry(juce::File modelsDir);
    void logStats() const;

    std::array<std::unique_ptr<ModelWrapper>, static_cast<size_t>(ModelType::COUNT)> models_;
    std::mutex mutex_;
    bool initialized_ = false;
};

// ============================================================================
// Async Inference (Lock-Free for Audio Thread)
// ============================================================================
class InferenceThread;  // Forward declaration

class AsyncMLPipeline {
public:
    explicit AsyncMLPipeline(MultiModelProcessor& processor);
    ~AsyncMLPipeline();

    void start();
    void stop();

    // Non-blocking submit (audio thread safe)
    // Returns a request id for tracking; 0 if a request is already pending.
    uint64_t submitFeatures(const std::array<float, 128>& features);

    // Non-blocking result check (audio thread safe)
    bool hasResult() const;
    bool hasResult(uint64_t requestId) const;

    InferenceResult getResult();
    InferenceResult getResult(uint64_t requestId);

private:
    friend class InferenceThread;  // Allow InferenceThread to call run()
    void run();

    MultiModelProcessor& processor_;
    std::unique_ptr<juce::Thread> thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> hasRequest_{false};
    std::atomic<bool> hasResult_{false};
    std::atomic<uint64_t> nextRequestId_{1};
    std::atomic<uint64_t> pendingRequestId_{0};
    std::atomic<uint64_t> latestResultId_{0};
    std::array<float, 128> pendingFeatures_{};
    InferenceResult latestResult_;
};

} // namespace ML
} // namespace Kelly
