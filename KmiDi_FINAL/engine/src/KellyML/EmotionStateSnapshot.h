#pragma once

#include "EmotionState.h"

#include <string>
#include <chrono>

namespace kelly::ml {

// Snapshot of EmotionState with metadata for serialization.
struct EmotionStateSnapshot {
    static constexpr int CURRENT_VERSION = 1;

    int version {CURRENT_VERSION};
    double timestamp {0.0};  // Unix timestamp with fractional seconds
    EmotionState emotion;

    // Optional labels (from Python mapping tables or heuristics)
    std::string labelPrimary;
    std::string labelSecondary;

    // Serialize to JSON string. Returns empty string on error.
    std::string toJson() const;

    // Write snapshot to file. Returns true on success.
    bool writeToFile(const std::string& filePath) const;

    // Create snapshot from EmotionState with current timestamp.
    static EmotionStateSnapshot fromEmotionState(const EmotionState& state);
};

} // namespace kelly::ml
