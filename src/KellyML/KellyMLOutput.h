#pragma once

#include <array>
#include <cstddef>

namespace kelly::ml {

struct KellyMLOutput {
    std::array<float, 128> noteProbabilities{};   // MelodyTransformer output (or fallback)
    std::array<float, 64> chordProbabilities{};   // HarmonyPredictor output (or fallback)
    std::array<float, 32> groove{};               // GroovePredictor output (or fallback)
    std::array<float, 16> dynamics{};             // DynamicsEngine output (or fallback)
};

} // namespace kelly::ml
