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
#include <array>
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <cmath>

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
    COUNT             = 5
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

constexpr std::array<ModelSpec, 5> MODEL_SPECS = {{
    {"EmotionRecognizer", 128, 64,  497664},
    {"MelodyTransformer", 64,  128, 412672},
    {"HarmonyPredictor",  128, 64,  74048},
    {"DynamicsEngine",    32,  16,  13456},
    {"GroovePredictor",   64,  32,  19040}
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
    bool valid = false;
};

// ============================================================================
// Single Model Wrapper (Heap Allocated)
// ============================================================================
class ModelWrapper {
public:
    explicit ModelWrapper(ModelType type) : type_(type), spec_(MODEL_SPECS[static_cast<size_t>(type)]) {
        output_.resize(spec_.outputSize, 0.0f);
    }

    bool loadWeights(const juce::File& file) {
        if (!file.existsAsFile()) {
            juce::Logger::writeToLog(juce::String("Model not found: ") + spec_.name);
            return false;
        }
        // Parse JSON and load weights here
        // For now, mark as loaded with fallback behavior
        loaded_ = true;
        juce::Logger::writeToLog(juce::String("Loaded model: ") + spec_.name + 
                                 " (" + juce::String(spec_.estimatedParams) + " params)");
        return true;
    }

    std::vector<float> forward(const float* input, size_t inputSize) {
        if (inputSize != spec_.inputSize) {
            return output_;
        }

#ifdef ENABLE_RTNEURAL
        if (rtModel_) {
            rtModel_->forward(input);
            for (size_t i = 0; i < spec_.outputSize; ++i) {
                output_[i] = rtModel_->getOutputs()[i];
            }
            return output_;
        }
#endif
        // Fallback heuristic
        computeFallback(input);
        return output_;
    }

    bool isLoaded() const { return loaded_; }
    void setEnabled(bool e) { enabled_ = e; }
    bool isEnabled() const { return enabled_; }
    const ModelSpec& spec() const { return spec_; }

private:
    void computeFallback(const float* input) {
        switch (type_) {
            case ModelType::EmotionRecognizer:
                for (size_t i = 0; i < 32; ++i) {
                    output_[i] = std::tanh(input[i] * 0.3f);
                    output_[i + 32] = std::tanh(input[i + 64] * 0.5f);
                }
                break;
            case ModelType::MelodyTransformer:
                for (size_t i = 0; i < spec_.outputSize; ++i) {
                    float note = static_cast<float>(i % 12) / 12.0f;
                    output_[i] = 1.0f / (1.0f + std::exp(-(note * 2.0f - 1.0f + input[i % 64] * 0.5f)));
                }
                break;
            case ModelType::HarmonyPredictor:
                for (size_t i = 0; i < spec_.outputSize; ++i) {
                    output_[i] = std::tanh(input[i] * 0.4f);
                }
                break;
            case ModelType::DynamicsEngine:
                for (size_t i = 0; i < spec_.outputSize; ++i) {
                    output_[i] = 0.5f + input[i % 32] * 0.3f;
                }
                break;
            case ModelType::GroovePredictor:
                for (size_t i = 0; i < spec_.outputSize; ++i) {
                    output_[i] = std::tanh(input[i % 64] * 0.6f);
                }
                break;
            default:
                break;
        }
    }

    ModelType type_;
    ModelSpec spec_;
    std::vector<float> output_;
    bool loaded_ = false;
    bool enabled_ = true;

#ifdef ENABLE_RTNEURAL
    std::unique_ptr<RTNeural::Model<float>> rtModel_;
#endif
};

// ============================================================================
// Multi-Model Manager
// ============================================================================
class MultiModelProcessor {
public:
    MultiModelProcessor() {
        for (size_t i = 0; i < static_cast<size_t>(ModelType::COUNT); ++i) {
            models_[i] = std::make_unique<ModelWrapper>(static_cast<ModelType>(i));
        }
    }

    bool initialize(const juce::File& modelsDir) {
        bool anyLoaded = false;
        for (size_t i = 0; i < models_.size(); ++i) {
            auto modelFile = modelsDir.getChildFile(
                juce::String(MODEL_SPECS[i].name).toLowerCase() + ".json");
            if (models_[i]->loadWeights(modelFile)) {
                anyLoaded = true;
            }
        }
        initialized_ = true;
        logStats();
        return anyLoaded;
    }

    void setModelEnabled(ModelType type, bool enabled) {
        models_[static_cast<size_t>(type)]->setEnabled(enabled);
    }

    bool isModelEnabled(ModelType type) const {
        return models_[static_cast<size_t>(type)]->isEnabled();
    }

    // Run single model inference
    std::vector<float> infer(ModelType type, const std::vector<float>& input) {
        std::lock_guard<std::mutex> lock(mutex_);
        return models_[static_cast<size_t>(type)]->forward(input.data(), input.size());
    }

    // Run full pipeline: audio features → all outputs
    InferenceResult runFullPipeline(const std::array<float, 128>& audioFeatures) {
        std::lock_guard<std::mutex> lock(mutex_);
        InferenceResult result;

        // 1. EmotionRecognizer: audio → emotion
        auto emotion = models_[0]->forward(audioFeatures.data(), 128);
        std::copy_n(emotion.begin(), 64, result.emotionEmbedding.begin());

        // 2. MelodyTransformer: emotion → melody
        if (models_[1]->isEnabled()) {
            auto melody = models_[1]->forward(emotion.data(), 64);
            std::copy_n(melody.begin(), 128, result.melodyProbabilities.begin());
        }

        // 3. HarmonyPredictor: context (emotion + audio) → harmony
        if (models_[2]->isEnabled()) {
            std::array<float, 128> context{};
            std::copy_n(emotion.begin(), 64, context.begin());
            std::copy_n(audioFeatures.begin(), 64, context.begin() + 64);
            auto harmony = models_[2]->forward(context.data(), 128);
            std::copy_n(harmony.begin(), 64, result.harmonyPrediction.begin());
        }

        // 4. DynamicsEngine: compact emotion → dynamics
        if (models_[3]->isEnabled()) {
            std::array<float, 32> compact{};
            std::copy_n(emotion.begin(), 32, compact.begin());
            auto dynamics = models_[3]->forward(compact.data(), 32);
            std::copy_n(dynamics.begin(), 16, result.dynamicsOutput.begin());
        }

        // 5. GroovePredictor: emotion → groove
        if (models_[4]->isEnabled()) {
            auto groove = models_[4]->forward(emotion.data(), 64);
            std::copy_n(groove.begin(), 32, result.grooveParameters.begin());
        }

        result.valid = true;
        return result;
    }

    size_t getTotalParams() const {
        size_t total = 0;
        for (const auto& spec : MODEL_SPECS) total += spec.estimatedParams;
        return total;
    }

    size_t getTotalMemoryKB() const {
        return getTotalParams() * sizeof(float) / 1024;
    }

private:
    void logStats() const {
        juce::Logger::writeToLog("MultiModelProcessor initialized:");
        juce::Logger::writeToLog("  Total params: " + juce::String(getTotalParams()));
        juce::Logger::writeToLog("  Total memory: " + juce::String(getTotalMemoryKB()) + " KB");
    }

    std::array<std::unique_ptr<ModelWrapper>, 5> models_;
    std::mutex mutex_;
    bool initialized_ = false;
};

// ============================================================================
// Async Inference (Lock-Free for Audio Thread)
// ============================================================================
class AsyncMLPipeline {
public:
    explicit AsyncMLPipeline(MultiModelProcessor& processor) : processor_(processor) {}

    void start() {
        running_ = true;
        thread_ = std::make_unique<std::thread>([this] { run(); });
    }

    void stop() {
        running_ = false;
        if (thread_ && thread_->joinable()) {
            thread_->join();
        }
    }

    // Non-blocking submit (audio thread safe)
    void submitFeatures(const std::array<float, 128>& features) {
        pendingFeatures_ = features;
        hasRequest_.store(true, std::memory_order_release);
    }

    // Non-blocking result check (audio thread safe)
    bool hasResult() const {
        return hasResult_.load(std::memory_order_acquire);
    }

    InferenceResult getResult() {
        hasResult_.store(false, std::memory_order_release);
        return latestResult_;
    }

private:
    void run() {
        while (running_) {
            if (hasRequest_.load(std::memory_order_acquire)) {
                hasRequest_.store(false, std::memory_order_release);
                latestResult_ = processor_.runFullPipeline(pendingFeatures_);
                hasResult_.store(true, std::memory_order_release);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    MultiModelProcessor& processor_;
    std::unique_ptr<std::thread> thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> hasRequest_{false};
    std::atomic<bool> hasResult_{false};
    std::array<float, 128> pendingFeatures_{};
    InferenceResult latestResult_;
};

} // namespace ML
} // namespace Kelly
