#pragma once

#include "EmotionState.h"
#include "KellyMLModel.h"
#include "KellyMLOutput.h"

#include <array>
#include <cstddef>
#include <string>

namespace kelly::ml {

class KellyMLPipeline {
public:
    KellyMLPipeline() = default;

    // Load all five models. Missing paths are tolerated (fallbacks will be used).
    void loadModels(const std::string& emotionPath,
                    const std::string& melodyPath,
                    const std::string& harmonyPath,
                    const std::string& dynamicsPath,
                    const std::string& groovePath);

    void setBypassed(bool bypassed) noexcept { bypassed_ = bypassed; }
    bool isBypassed() const noexcept { return bypassed_; }

    // Step 1: process raw audio features (128 floats) -> Emotion embedding + EmotionState.
    EmotionState processAudioFeatures(const float* audioFeatures, std::size_t length);

    // Step 2: process an EmotionState or embedding through remaining models.
    KellyMLOutput generateMusicalSuggestions(const EmotionState& emotion,
                                             const float* emotionEmbedding,
                                             std::size_t embeddingLength,
                                             const float* audioContext128,
                                             std::size_t contextLength);

private:
    // Fallback helpers (deterministic, no allocations).
    void fallbackEmotion(const float* features, std::size_t length, float* out64) const noexcept;
    void fallbackMelody(const float* embedding, std::size_t length, std::array<float, 128>& out) const noexcept;
    void fallbackHarmony(const float* context128, std::size_t length, std::array<float, 64>& out) const noexcept;
    void fallbackDynamics(const EmotionState& emotion, std::array<float, 16>& out) const noexcept;
    void fallbackGroove(const float* embedding, std::size_t length, std::array<float, 32>& out) const noexcept;

    KellyMLModel emotionRecognizer_;
    KellyMLModel melodyTransformer_;
    KellyMLModel harmonyPredictor_;
    KellyMLModel dynamicsEngine_;
    KellyMLModel groovePredictor_;

    bool bypassed_ {false};
};

} // namespace kelly::ml
