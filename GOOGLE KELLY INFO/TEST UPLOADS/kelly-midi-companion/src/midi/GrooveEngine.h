#pragma once

#include "common/Types.h"
#include <vector>
#include <random>

namespace kelly {

/**
 * Groove Engine - Applies humanization and groove patterns to MIDI.
 * 
 * Ported from Python groove engine with full humanization support:
 * - Timing drift and micro-timing adjustments
 * - Velocity humanization
 * - Note dropouts based on complexity
 * - Ghost notes
 * - Emotion-based timing adjustments
 */
class GrooveEngine {
public:
    GrooveEngine();
    ~GrooveEngine() = default;
    
    /**
     * Apply groove/humanization to MIDI notes.
     * @param notes Input notes
     * @param grooveType Type of groove to apply
     * @param humanizationLevel 0.0 (tight) to 1.0 (loose)
     * @return Humanized notes
     */
    std::vector<MidiNote> applyGroove(
        const std::vector<MidiNote>& notes,
        GrooveType grooveType,
        float humanizationLevel
    );
    
    /**
     * Apply emotion-based timing adjustments.
     * Sad emotions drag behind, angry emotions rush ahead.
     */
    std::vector<MidiNote> applyEmotionTiming(
        const std::vector<MidiNote>& notes,
        const EmotionNode& emotion
    );

private:
    std::mt19937 rng_;  // Random number generator for humanization
};

} // namespace kelly

