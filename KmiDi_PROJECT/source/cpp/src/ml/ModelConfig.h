#pragma once
/*
 * ModelConfig.h - ML Model Configuration
 * ======================================
 *
 * Central configuration for Kelly's ML models.
 * Defines paths, sizes, and runtime settings.
 */

#include <juce_core/juce_core.h>
#include <array>
#include <string>

namespace kelly {
namespace ml {

/**
 * Default model directory relative to application data.
 */
inline juce::File getDefaultModelsDirectory() {
    // Try several locations in order of preference:
    // 1. Environment variable
    // 2. Alongside executable
    // 3. User application data
    // 4. System application data

    // Check environment variable
    const char* envPath = std::getenv("KELLY_MODELS_PATH");
    if (envPath && juce::File(envPath).isDirectory()) {
        return juce::File(envPath);
    }

    // Check alongside executable
    auto execDir = juce::File::getSpecialLocation(
        juce::File::currentExecutableFile).getParentDirectory();
    auto execModels = execDir.getChildFile("models");
    if (execModels.isDirectory()) {
        return execModels;
    }

    // Check Resources folder (macOS bundle)
    auto resourcesModels = execDir.getParentDirectory()
        .getChildFile("Resources").getChildFile("models");
    if (resourcesModels.isDirectory()) {
        return resourcesModels;
    }

    // User application data
    auto userModels = juce::File::getSpecialLocation(
        juce::File::userApplicationDataDirectory)
        .getChildFile("Kelly").getChildFile("models");
    if (userModels.isDirectory()) {
        return userModels;
    }

    // Default: create in user data
    userModels.createDirectory();
    return userModels;
}

/**
 * Model inference configuration.
 */
struct InferenceConfig {
    // Threading
    int inferenceThreadPriority = 7;  // 0-10, higher = more priority
    int maxPendingRequests = 16;
    int resultBufferSize = 16;

    // Timing
    float maxInferenceTimeMs = 10.0f;  // Target latency
    int lookaheadBufferMs = 20;         // Lookahead for latency compensation

    // Features
    int featureHopSizeMs = 10;          // Feature extraction hop size
    int featureWindowSizeMs = 50;       // Feature extraction window

    // Model selection
    bool enableEmotionRecognizer = true;
    bool enableMelodyTransformer = true;
    bool enableHarmonyPredictor = true;
    bool enableDynamicsEngine = true;
    bool enableGroovePredictor = true;

    // Fallback behavior
    bool useFallbackOnError = true;     // Use heuristics if model fails
    bool logInferenceTime = false;      // Log inference timing
};

/**
 * Model file names (lowercase, no extension).
 */
struct ModelFiles {
    static constexpr const char* EmotionRecognizer = "emotionrecognizer";
    static constexpr const char* MelodyTransformer = "melodytransformer";
    static constexpr const char* HarmonyPredictor = "harmonypredictor";
    static constexpr const char* DynamicsEngine = "dynamicsengine";
    static constexpr const char* GroovePredictor = "groovepredictor";
};

/**
 * Get model file path for given model name.
 * Tries RTNeural (.json) first, then ONNX (.onnx).
 */
inline juce::File getModelPath(const juce::File& modelsDir, const char* modelName) {
    // Try RTNeural JSON first
    auto jsonPath = modelsDir.getChildFile(juce::String(modelName) + ".json");
    if (jsonPath.existsAsFile()) {
        return jsonPath;
    }

    // Try ONNX
    auto onnxPath = modelsDir.getChildFile(juce::String(modelName) + ".onnx");
    if (onnxPath.existsAsFile()) {
        return onnxPath;
    }

    // Return JSON path (will trigger fallback mode)
    return jsonPath;
}

/**
 * Check if all required models exist.
 */
inline bool validateModelsDirectory(const juce::File& modelsDir) {
    if (!modelsDir.isDirectory()) {
        return false;
    }

    const char* requiredModels[] = {
        ModelFiles::EmotionRecognizer,
        ModelFiles::MelodyTransformer,
        ModelFiles::HarmonyPredictor,
        ModelFiles::DynamicsEngine,
        ModelFiles::GroovePredictor
    };

    for (const auto* model : requiredModels) {
        auto path = getModelPath(modelsDir, model);
        if (!path.existsAsFile()) {
            juce::Logger::writeToLog(
                juce::String("Missing model: ") + model +
                " (will use fallback heuristics)");
        }
    }

    return true;  // Allow fallback mode
}

/**
 * Runtime configuration singleton.
 */
class ModelConfigManager {
public:
    static ModelConfigManager& getInstance() {
        static ModelConfigManager instance;
        return instance;
    }

    void setModelsDirectory(const juce::File& dir) {
        modelsDir_ = dir;
    }

    juce::File getModelsDirectory() const {
        return modelsDir_.isDirectory() ? modelsDir_ : getDefaultModelsDirectory();
    }

    void setConfig(const InferenceConfig& config) {
        config_ = config;
    }

    const InferenceConfig& getConfig() const {
        return config_;
    }

    InferenceConfig& getConfig() {
        return config_;
    }

private:
    ModelConfigManager() = default;

    juce::File modelsDir_;
    InferenceConfig config_;
};

} // namespace ml
} // namespace kelly

