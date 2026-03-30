#include "KellyMLPipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace kelly::ml {

namespace {
template <std::size_t N>
inline void zero(std::array<float, N>& arr) noexcept {
    for (auto& v : arr) v = 0.0f;
}
} // namespace

void KellyMLPipeline::loadModels(const std::string& emotionPath,
                                 const std::string& melodyPath,
                                 const std::string& harmonyPath,
                                 const std::string& dynamicsPath,
                                 const std::string& groovePath) {
    emotionRecognizer_.loadFromJson(emotionPath, 128, 64);
    melodyTransformer_.loadFromJson(melodyPath, 64, 128);
    harmonyPredictor_.loadFromJson(harmonyPath, 128, 64);
    dynamicsEngine_.loadFromJson(dynamicsPath, 32, 16);
    groovePredictor_.loadFromJson(groovePath, 64, 32);
}

EmotionState KellyMLPipeline::processAudioFeatures(const float* audioFeatures, std::size_t length) {
    float embedding[64] = {};
    if (bypassed_) {
        return EmotionState{};
    }

    const bool ok = emotionRecognizer_.process(audioFeatures, embedding);
    if (!ok) {
        fallbackEmotion(audioFeatures, length, embedding);
    }
    return EmotionState::FromEmbedding(embedding, 64);
}

KellyMLOutput KellyMLPipeline::generateMusicalSuggestions(const EmotionState& emotion,
                                                          const float* emotionEmbedding,
                                                          std::size_t embeddingLength,
                                                          const float* audioContext128,
                                                          std::size_t contextLength) {
    KellyMLOutput output{};
    if (bypassed_) {
        return output;
    }

    float embedding64[64] = {};
    const float* useEmbedding = emotionEmbedding;
    if (emotionEmbedding == nullptr || embeddingLength < 64) {
        // Derive a minimal embedding from EmotionState if none provided.
        embedding64[0] = emotion.valence;
        embedding64[1] = emotion.arousal;
        embedding64[2] = emotion.dominance;
        embedding64[3] = emotion.complexity;
        useEmbedding = embedding64;
        embeddingLength = 64;
    }

    // Melody
    if (!melodyTransformer_.process(useEmbedding, output.noteProbabilities.data())) {
        fallbackMelody(useEmbedding, embeddingLength, output.noteProbabilities);
    }

    // Harmony (expect 128 float context: emotion + audio context)
    float context[128] = {};
    if (audioContext128 && contextLength >= 128) {
        std::memcpy(context, audioContext128, 128 * sizeof(float));
    } else {
        // Build a simple context: repeat embedding twice if no context given.
        for (std::size_t i = 0; i < 64; ++i) {
            context[i] = useEmbedding[i % embeddingLength];
            context[i + 64] = useEmbedding[i % embeddingLength];
        }
    }
    if (!harmonyPredictor_.process(context, output.chordProbabilities.data())) {
        fallbackHarmony(context, 128, output.chordProbabilities);
    }

    // Dynamics (compress emotion to 32 inputs)
    float dynInput[32] = {};
    for (std::size_t i = 0; i < 32; ++i) {
        dynInput[i] = useEmbedding[i % embeddingLength];
    }
    if (!dynamicsEngine_.process(dynInput, output.dynamics.data())) {
        fallbackDynamics(emotion, output.dynamics);
    }

    // Groove
    if (!groovePredictor_.process(useEmbedding, output.groove.data())) {
        fallbackGroove(useEmbedding, embeddingLength, output.groove);
    }

    return output;
}

void KellyMLPipeline::fallbackEmotion(const float* features, std::size_t length, float* out64) const noexcept {
    // Simple RMS + banded averages to 64 bins.
    for (std::size_t i = 0; i < 64; ++i) out64[i] = 0.0f;
    if (!features || length == 0) return;
    const std::size_t bucket = length / 64;
    for (std::size_t i = 0; i < 64; ++i) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < bucket && (i * bucket + j) < length; ++j) {
            const float v = features[i * bucket + j];
            sum += v * v;
        }
        out64[i] = std::sqrt(sum / static_cast<float>(bucket > 0 ? bucket : 1));
    }
}

void KellyMLPipeline::fallbackMelody(const float* embedding, std::size_t length, std::array<float, 128>& out) const noexcept {
    zero(out);
    if (!embedding || length == 0) return;
    // Simple pentatonic weighting cycle.
    const int scaleDegrees[5] = {0, 2, 4, 7, 9};
    for (int i = 0; i < 128; ++i) {
        const float w = embedding[i % length];
        out[i] = (std::find(std::begin(scaleDegrees), std::end(scaleDegrees), i % 12) != std::end(scaleDegrees))
            ? 0.7f + 0.3f * w
            : 0.1f * w;
    }
}

void KellyMLPipeline::fallbackHarmony(const float* context128, std::size_t length, std::array<float, 64>& out) const noexcept {
    zero(out);
    if (!context128 || length < 12) {
        // Circle of fifths center around C (fallback values).
        for (int i = 0; i < 12 && i < 64; ++i) {
            out[i] = 1.0f - (0.05f * i);
        }
        return;
    }
    // Use energy in thirds to bias major/minor preference.
    const float energy = context128[0] + context128[5] + context128[7];
    const bool majorBias = energy >= 0.0f;
    for (int i = 0; i < 12 && i < 64; ++i) {
        const bool isMajorTriad = (i % 12 == 0 || i % 12 == 4 || i % 12 == 7);
        out[i] = majorBias == isMajorTriad ? 0.9f : 0.3f;
    }
}

void KellyMLPipeline::fallbackDynamics(const EmotionState& emotion, std::array<float, 16>& out) const noexcept {
    // Envelope-like shaping: map valence/arousal to attack/decay/sustain/release.
    out[0] = 0.02f + 0.01f * (1.0f + emotion.arousal); // attack
    out[1] = 0.15f + 0.05f * (1.0f - emotion.valence); // decay
    out[2] = 0.6f + 0.2f * (emotion.valence);          // sustain
    out[3] = 0.25f + 0.05f * (1.0f - emotion.dominance); // release
    for (std::size_t i = 4; i < out.size(); ++i) {
        out[i] = out[i % 4];
    }
}

void KellyMLPipeline::fallbackGroove(const float* embedding, std::size_t length, std::array<float, 32>& out) const noexcept {
    zero(out);
    if (!embedding || length == 0) return;
    // Tempo-based swing proxy: use first embedding bin as swing amount.
    const float swing = std::clamp(0.5f + 0.25f * embedding[0], 0.35f, 0.65f);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const bool isOffbeat = (i % 2) == 1;
        out[i] = isOffbeat ? swing : (1.0f - swing);
    }
}

} // namespace kelly::ml
