#pragma once

#include "EmotionStateSnapshot.h"
#include "MLPipelineSnapshot.h"
#include "KellyMLPipeline.h"
#include "KellyMLOutput.h"

#include <string>
#include <chrono>

namespace kelly::ml {

// Helper class for writing snapshots from KellyMLPipeline.
// Integrates snapshot writing into the pipeline workflow.
class SnapshotWriter {
public:
    SnapshotWriter() = default;

    // Set output directory for snapshots
    void setOutputDirectory(const std::string& dir);

    // Write emotion snapshot
    void writeEmotionSnapshot(const EmotionState& emotion,
                              const std::string& primaryLabel = "",
                              const std::string& secondaryLabel = "");

    // Write full ML pipeline snapshot
    void writePipelineSnapshot(
        const EmotionState& emotion,
        const KellyMLOutput& output,
        const KellyMLPipeline& pipeline,
        double inferenceTimeMs,
        bool fallbackActive,
        const std::string& primaryLabel = "",
        const std::string& secondaryLabel = "");

    // Get current snapshot directory
    const std::string& outputDirectory() const { return outputDir_; }

private:
    std::string outputDir_;
    std::string getEmotionSnapshotPath() const;
    std::string getPipelineSnapshotPath() const;
};

} // namespace kelly::ml
