#include "MLPipelineSnapshot.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace kelly::ml {

std::string MLPipelineSnapshot::toJson() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);

    oss << "{\n";
    oss << "  \"version\": " << version << ",\n";
    oss << "  \"timestamp\": " << timestamp << ",\n";
    oss << "  \"emotion\": {\n";
    oss << "    \"valence\": " << emotionSnapshot.emotion.valence << ",\n";
    oss << "    \"arousal\": " << emotionSnapshot.emotion.arousal << ",\n";
    oss << "    \"dominance\": " << emotionSnapshot.emotion.dominance << ",\n";
    oss << "    \"complexity\": " << emotionSnapshot.emotion.complexity << "\n";
    oss << "  }";

    if (!emotionSnapshot.labelPrimary.empty() || !emotionSnapshot.labelSecondary.empty()) {
        oss << ",\n";
        oss << "  \"labels\": {\n";
        if (!emotionSnapshot.labelPrimary.empty()) {
            oss << "    \"primary\": \"" << emotionSnapshot.labelPrimary << "\"";
            if (!emotionSnapshot.labelSecondary.empty()) {
                oss << ",\n";
            }
        }
        if (!emotionSnapshot.labelSecondary.empty()) {
            if (emotionSnapshot.labelPrimary.empty()) {
                oss << "    ";
            }
            oss << "    \"secondary\": \"" << emotionSnapshot.labelSecondary << "\"\n";
        } else {
            oss << "\n";
        }
        oss << "  }";
    }

    oss << ",\n";
    oss << "  \"ml_debug\": {\n";
    oss << "    \"emotion_recognizer_enabled\": " << (debugInfo.emotionRecognizerEnabled ? "true" : "false") << ",\n";
    oss << "    \"melody_transformer_enabled\": " << (debugInfo.melodyTransformerEnabled ? "true" : "false") << ",\n";
    oss << "    \"harmony_predictor_enabled\": " << (debugInfo.harmonyPredictorEnabled ? "true" : "false") << ",\n";
    oss << "    \"dynamics_engine_enabled\": " << (debugInfo.dynamicsEngineEnabled ? "true" : "false") << ",\n";
    oss << "    \"groove_predictor_enabled\": " << (debugInfo.groovePredictorEnabled ? "true" : "false") << ",\n";
    oss << "    \"inference_time_ms\": " << debugInfo.inferenceTimeMs << ",\n";
    oss << "    \"fallback_active\": " << (debugInfo.fallbackActive ? "true" : "false") << ",\n";
    oss << "    \"note_prob_summary\": [";
    for (std::size_t i = 0; i < debugInfo.noteProbSummary.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << debugInfo.noteProbSummary[i];
    }
    oss << "],\n";
    oss << "    \"chord_prob_summary\": [";
    for (std::size_t i = 0; i < debugInfo.chordProbSummary.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << debugInfo.chordProbSummary[i];
    }
    oss << "],\n";
    oss << "    \"groove_summary\": [";
    for (std::size_t i = 0; i < debugInfo.grooveSummary.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << debugInfo.grooveSummary[i];
    }
    oss << "],\n";
    oss << "    \"dynamics_summary\": [";
    for (std::size_t i = 0; i < debugInfo.dynamicsSummary.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << debugInfo.dynamicsSummary[i];
    }
    oss << "]\n";
    oss << "  }\n";
    oss << "}";

    return oss.str();
}

bool MLPipelineSnapshot::writeToFile(const std::string& filePath) const {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        return false;
    }

    file << toJson();
    file.close();
    return file.good();
}

MLPipelineSnapshot MLPipelineSnapshot::fromPipeline(
    const EmotionStateSnapshot& emotion,
    const KellyMLOutput& output,
    bool emotionRecognizerEnabled,
    bool melodyTransformerEnabled,
    bool harmonyPredictorEnabled,
    bool dynamicsEngineEnabled,
    bool groovePredictorEnabled,
    double inferenceTimeMs,
    bool fallbackActive) {

    MLPipelineSnapshot snapshot;
    snapshot.emotionSnapshot = emotion;
    snapshot.output = output;
    snapshot.timestamp = emotion.timestamp;

    snapshot.debugInfo.emotionRecognizerEnabled = emotionRecognizerEnabled;
    snapshot.debugInfo.melodyTransformerEnabled = melodyTransformerEnabled;
    snapshot.debugInfo.harmonyPredictorEnabled = harmonyPredictorEnabled;
    snapshot.debugInfo.dynamicsEngineEnabled = dynamicsEngineEnabled;
    snapshot.debugInfo.groovePredictorEnabled = groovePredictorEnabled;
    snapshot.debugInfo.inferenceTimeMs = inferenceTimeMs;
    snapshot.debugInfo.fallbackActive = fallbackActive;

    // Copy first 4 elements of each output vector for summary
    constexpr std::size_t summarySize = 4;
    std::copy_n(output.noteProbabilities.begin(),
                std::min(summarySize, output.noteProbabilities.size()),
                snapshot.debugInfo.noteProbSummary.begin());
    std::copy_n(output.chordProbabilities.begin(),
                std::min(summarySize, output.chordProbabilities.size()),
                snapshot.debugInfo.chordProbSummary.begin());
    std::copy_n(output.groove.begin(),
                std::min(summarySize, output.groove.size()),
                snapshot.debugInfo.grooveSummary.begin());
    std::copy_n(output.dynamics.begin(),
                std::min(summarySize, output.dynamics.size()),
                snapshot.debugInfo.dynamicsSummary.begin());

    return snapshot;
}

} // namespace kelly::ml
