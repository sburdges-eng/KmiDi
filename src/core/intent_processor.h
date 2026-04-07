#pragma once
/*
 * intent_processor.h - Legacy Intent Phase Stub
 * ==============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Engine Layer: engine/IntentProcessor.h (full three-phase port — use instead)
 * - Core Layer: Minimal Wound + IntentPhase placeholder
 *
 * Purpose: Early scaffold; kept to avoid breaking includes in legacy targets.
 *
 * Features:
 * - processWound placeholder returning int status
 */

#include <string>

namespace kelly {

enum class IntentPhase {
    Wound,
    Emotion,
    RuleBreak
};

struct Wound {
    std::string description;
    float intensity;
    std::string source;
};

class IntentProcessor {
public:
    IntentProcessor() = default;
    ~IntentProcessor() = default;

    int processWound(const Wound& wound);
    // Additional methods would be implemented

private:
    // Implementation details
};

} // namespace kelly
