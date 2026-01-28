#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace kelly::ml {

struct EmotionState {
    float valence {0.0f};
    float arousal {0.0f};
    float dominance {0.0f};
    float complexity {0.0f};

    // Derived from a 64-D embedding (EmotionRecognizer output).
    static EmotionState FromEmbedding(const float* embedding, std::size_t length) noexcept {
        EmotionState state{};
        if (embedding == nullptr || length < 8) {
            return state;
        }
        // Deterministic reduction over the first few bins to avoid costly ops.
        const float v = embedding[0] + embedding[8] - embedding[16];
        const float a = embedding[1] + embedding[9] + embedding[17];
        const float d = embedding[2] + embedding[10] - embedding[18];
        const float c = embedding[3] + embedding[11] + embedding[19];
        state.valence = v * 0.25f;
        state.arousal = a * 0.25f;
        state.dominance = d * 0.25f;
        state.complexity = c * 0.25f;
        return state;
    }
};

} // namespace kelly::ml
