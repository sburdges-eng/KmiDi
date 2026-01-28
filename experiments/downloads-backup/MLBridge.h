#pragma once
/*
 * MLBridge.h - Connect ML Models to IntentPipeline
 * =================================================
 * Bridges the gap between:
 *   - MultiModelProcessor (neural network inference)
 *   - IntentPipeline (wound → musical parameters)
 * 
 * Flow:
 *   Audio → FeatureExtractor → EmotionRecognizer → EmotionNode
 *   EmotionNode → IntentPipeline → IntentResult
 *   IntentResult + MelodyTransformer → GeneratedMidi
 */

#include "KellyTypes.h"
#include "KellyBrain.h"
#include "MultiModelProcessor.h"
#include <memory>
#include <array>

namespace kelly {

// =============================================================================
// ML → Emotion Mapping
// =============================================================================

class EmotionEmbeddingMapper {
public:
    EmotionEmbeddingMapper() = default;
    explicit EmotionEmbeddingMapper(std::shared_ptr<EmotionThesaurus> thesaurus);
    
    // Convert 64-dim embedding to EmotionNode
    EmotionNode embeddingToEmotion(const std::array<float, 64>& embedding);
    
    // Convert EmotionNode to 64-dim embedding (for training)
    std::array<float, 64> emotionToEmbedding(const EmotionNode& node);
    
    // Batch conversion
    std::vector<EmotionNode> embeddingsToEmotions(
        const std::vector<std::array<float, 64>>& embeddings);
    
    // Interpretation helpers
    struct EmbeddingInterpretation {
        float valence;           // Derived from dims 0-15
        float arousal;           // Derived from dims 16-31
        float dominance;         // Derived from dims 32-47
        float complexity;        // Derived from dims 48-63
        EmotionCategory primaryCategory;
        float confidence;
    };
    
    EmbeddingInterpretation interpret(const std::array<float, 64>& embedding);

private:
    std::shared_ptr<EmotionThesaurus> thesaurus_;
    
    // Learned mapping weights (can be loaded from file)
    std::array<std::array<float, 64>, 8> categoryWeights_;  // 8 categories
    void initializeDefaultWeights();
};

// =============================================================================
// ML-Enhanced Intent Pipeline
// =============================================================================

class MLIntentPipeline {
public:
    MLIntentPipeline();
    ~MLIntentPipeline() = default;
    
    // Initialize with model directory
    bool initialize(const std::string& modelsPath, const std::string& dataPath);
    
    // === Audio-Driven Generation ===
    
    // Process audio buffer → full pipeline
    IntentResult processAudio(const float* audioData, size_t numSamples);
    
    // Async version for real-time use
    void submitAudio(const float* audioData, size_t numSamples);
    bool hasResult() const;
    IntentResult getResult();
    
    // === Hybrid: Audio + Text ===
    
    // Combine audio emotion with text description
    IntentResult processHybrid(
        const float* audioData, size_t numSamples,
        const std::string& textDescription);
    
    // === Generate from ML Outputs ===
    
    // Use MelodyTransformer output directly
    std::vector<MidiNote> generateMelodyFromProbabilities(
        const std::array<float, 128>& noteProbabilities,
        const IntentResult& intent,
        int bars = 4);
    
    // Use HarmonyPredictor output
    std::vector<Chord> generateHarmonyFromPrediction(
        const std::array<float, 64>& harmonyPrediction,
        const IntentResult& intent);
    
    // Use DynamicsEngine output
    void applyDynamics(
        std::vector<MidiNote>& notes,
        const std::array<float, 16>& dynamicsOutput);
    
    // Use GroovePredictor output
    void applyGroove(
        std::vector<MidiNote>& notes,
        const std::array<float, 32>& grooveParams);
    
    // === Full Generation ===
    
    GeneratedMidi generateFromAudio(
        const float* audioData, size_t numSamples,
        int bars = 8);
    
    // Access components
    IntentPipeline& intentPipeline() { return *intentPipeline_; }
    ML::MultiModelProcessor& mlProcessor() { return *mlProcessor_; }
    EmotionEmbeddingMapper& mapper() { return *mapper_; }
    
    // Enable/disable ML models
    void setMLEnabled(bool enabled) { mlEnabled_ = enabled; }
    bool isMLEnabled() const { return mlEnabled_; }
    
    // Per-model control
    void setModelEnabled(ML::ModelType type, bool enabled);

private:
    std::unique_ptr<IntentPipeline> intentPipeline_;
    std::unique_ptr<ML::MultiModelProcessor> mlProcessor_;
    std::unique_ptr<ML::AsyncMLPipeline> asyncPipeline_;
    std::unique_ptr<EmotionEmbeddingMapper> mapper_;
    std::unique_ptr<ML::FeatureExtractor> featureExtractor_;
    
    bool mlEnabled_ = true;
    bool initialized_ = false;
    
    // Feature extraction settings
    double sampleRate_ = 44100.0;
    size_t fftSize_ = 2048;
};

// =============================================================================
// Feature Extractor (Audio → 128-dim mel features)
// =============================================================================

namespace ML {

class FeatureExtractor {
public:
    FeatureExtractor(double sampleRate = 44100.0, size_t fftSize = 2048);
    ~FeatureExtractor() = default;
    
    void prepare(double sampleRate, int samplesPerBlock);
    void reset();
    
    // Extract features from audio buffer
    std::array<float, 128> extractFeatures(const float* audioData, size_t numSamples);
    
    // Incremental extraction (for streaming)
    void pushSamples(const float* audioData, size_t numSamples);
    bool hasFeaturesReady() const;
    std::array<float, 128> popFeatures();

private:
    double sampleRate_;
    size_t fftSize_;
    size_t hopSize_;
    
    std::vector<std::vector<float>> melFilterbank_;
    std::vector<float> circularBuffer_;
    size_t bufferWritePos_ = 0;
    size_t samplesAccumulated_ = 0;
    
    std::vector<float> windowFunction_;
    
    void computeMelFilterbank();
    void applyWindow(float* data, size_t size);
};

} // namespace ML

// =============================================================================
// Convenience Functions
// =============================================================================

// One-shot audio → MIDI
inline GeneratedMidi audioToMidi(
    const float* audioData, size_t numSamples,
    const std::string& modelsPath = "./models",
    int bars = 8) 
{
    MLIntentPipeline pipeline;
    pipeline.initialize(modelsPath, "./data");
    return pipeline.generateFromAudio(audioData, numSamples, bars);
}

// Create ML-enhanced pipeline
inline std::unique_ptr<MLIntentPipeline> createMLPipeline() {
    return std::make_unique<MLIntentPipeline>();
}

} // namespace kelly
