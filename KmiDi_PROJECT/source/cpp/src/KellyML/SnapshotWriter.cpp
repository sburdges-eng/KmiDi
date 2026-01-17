#include "SnapshotWriter.h"
#include <filesystem>

namespace kelly::ml {

void SnapshotWriter::setOutputDirectory(const std::string& dir) {
    outputDir_ = dir;
    // Ensure directory exists
    if (!outputDir_.empty()) {
        std::filesystem::create_directories(outputDir_);
    }
}

void SnapshotWriter::writeEmotionSnapshot(
    const EmotionState& emotion,
    const std::string& primaryLabel,
    const std::string& secondaryLabel) {

    if (outputDir_.empty())
        return;

    auto snapshot = EmotionStateSnapshot::fromEmotionState(emotion);
    snapshot.labelPrimary = primaryLabel;
    snapshot.labelSecondary = secondaryLabel;

    snapshot.writeToFile(getEmotionSnapshotPath());
}

void SnapshotWriter::writePipelineSnapshot(
    const EmotionState& emotion,
    const KellyMLOutput& output,
    const KellyMLPipeline& pipeline,
    double inferenceTimeMs,
    bool fallbackActive,
    const std::string& primaryLabel,
    const std::string& secondaryLabel) {

    if (outputDir_.empty())
        return;

    auto emotionSnapshot = EmotionStateSnapshot::fromEmotionState(emotion);
    emotionSnapshot.labelPrimary = primaryLabel;
    emotionSnapshot.labelSecondary = secondaryLabel;

    // Note: This requires exposing model enabled state from KellyMLPipeline
    // For now, we'll use a simplified version - in a full implementation,
    // you'd add getter methods to KellyMLPipeline to expose model states

    // Simplified: assume all models are enabled (can be enhanced later)
    auto pipelineSnapshot = MLPipelineSnapshot::fromPipeline(
        emotionSnapshot,
        output,
        true,  // emotionRecognizerEnabled - should come from pipeline
        true,  // melodyTransformerEnabled
        true,  // harmonyPredictorEnabled
        true,  // dynamicsEngineEnabled
        true,  // groovePredictorEnabled
        inferenceTimeMs,
        fallbackActive
    );

    pipelineSnapshot.writeToFile(getPipelineSnapshotPath());
}

std::string SnapshotWriter::getEmotionSnapshotPath() const {
    if (outputDir_.empty())
        return "";
    return outputDir_ + "/emotion_snapshot.json";
}

std::string SnapshotWriter::getPipelineSnapshotPath() const {
    if (outputDir_.empty())
        return "";
    return outputDir_ + "/ml_debug_snapshot.json";
}

} // namespace kelly::ml
