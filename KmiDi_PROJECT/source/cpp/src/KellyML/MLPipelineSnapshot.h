#pragma once

#include "EmotionStateSnapshot.h"
#include "KellyMLOutput.h"
#include "KellyMLModel.h"

#include <string>
#include <array>

namespace kelly::ml {

// Aggregated snapshot of ML pipeline state including debug information.
struct MLPipelineSnapshot {
    static constexpr int CURRENT_VERSION = 1;

    int version {CURRENT_VERSION};
    double timestamp {0.0};

    EmotionStateSnapshot emotionSnapshot;
    KellyMLOutput output;

    // Debug information
    struct MLDebugInfo {
        bool emotionRecognizerEnabled {false};
        bool melodyTransformerEnabled {false};
        bool harmonyPredictorEnabled {false};
        bool dynamicsEngineEnabled {false};
        bool groovePredictorEnabled {false};

        double inferenceTimeMs {0.0};
        bool fallbackActive {false};

        // Summary of output vectors (first few values, not full dumps)
        std::array<float, 4> noteProbSummary{};      // First 4 of 128
        std::array<float, 4> chordProbSummary{};     // First 4 of 64
        std::array<float, 4> grooveSummary{};        // First 4 of 32
        std::array<float, 4> dynamicsSummary{};      // First 4 of 16
    } debugInfo;

    // Serialize to JSON string. Returns empty string on error.
    std::string toJson() const;

    // Write snapshot to file. Returns true on success.
    bool writeToFile(const std::string& filePath) const;

    // Create snapshot from pipeline state.
    static MLPipelineSnapshot fromPipeline(
        const EmotionStateSnapshot& emotion,
        const KellyMLOutput& output,
        bool emotionRecognizerEnabled,
        bool melodyTransformerEnabled,
        bool harmonyPredictorEnabled,
        bool dynamicsEngineEnabled,
        bool groovePredictorEnabled,
        double inferenceTimeMs,
        bool fallbackActive);
};

} // namespace kelly::ml
