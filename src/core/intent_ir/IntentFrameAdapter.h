#pragma once

/**
 * IntentFrameAdapter.h - Adapter functions for converting IntentFrame to engine types
 * 
 * This header provides helper functions to convert IntentFrame data to existing
 * engine types (VADState, MusicalParameters, etc.) for backward compatibility
 * during migration.
 */

#include "IntentFrame.h"
#include "common/KellyTypes.h"
#include <optional>

namespace kelly {
namespace intent_ir {

/**
 * Convert IntentFrame emotion to VADState
 */
inline VADState emotionToVAD(const EmotionState& emotion) {
    VADState vad;
    vad.valence = emotion.valence;
    vad.arousal = emotion.arousal;
    vad.dominance = emotion.dominance;
    return vad;
}

/**
 * Convert IntentFrame emotion to EmotionNode (if discrete_id is set)
 */
inline std::optional<EmotionNode> emotionToNode(const EmotionState& emotion) {
    if (emotion.discrete_id < 0) {
        return std::nullopt;
    }
    
    // Create minimal EmotionNode from discrete_id
    // Full conversion would require access to EmotionThesaurus
    EmotionNode node;
    node.id = emotion.discrete_id;
    node.valence = emotion.valence;
    node.arousal = emotion.arousal;
    node.dominance = emotion.dominance;
    node.intensity = emotion.intensity;
    return node;
}

/**
 * Convert IntentFrame music to MusicalParameters
 */
inline MusicalParameters musicToParams(const MusicalIntent& music) {
    MusicalParameters params;
    
    // Map tempo bias to tempo range
    // tempo_bias: -1.0 (slower) to +1.0 (faster)
    // Assuming base tempo of 120 BPM, range of 60-180 BPM
    float tempo_multiplier = 1.0f + (music.tempo_bias * 0.5f); // 0.5x to 1.5x
    params.tempoSuggested = static_cast<int>(120.0f * tempo_multiplier);
    
    // Map mode preference
    if (music.mode_preference > 0) {
        params.modeSuggested = "major";
    } else if (music.mode_preference < 0) {
        params.modeSuggested = "minor";
    } else {
        params.modeSuggested = "neutral";
    }
    
    // Map other parameters (these are approximations)
    params.dynamicsRange = music.dynamic_range;
    params.density = music.texture_density;
    params.dissonance = music.harmonic_tension;
    
    return params;
}

/**
 * Convert IntentFrame to IntentResult (for backward compatibility)
 */
inline IntentResult frameToIntentResult(const IntentFrame& frame) {
    IntentResult result;
    
    // Extract emotion
    result.emotion.valence = frame.emotion.valence;
    result.emotion.arousal = frame.emotion.arousal;
    result.emotion.dominance = frame.emotion.dominance;
    
    // Extract music parameters
    auto params = musicToParams(frame.music);
    result.tempoBpm = params.tempoSuggested;
    result.key = "C"; // Default, would need more context
    result.mode = params.modeSuggested;
    
    // Map texture density to melodic range (0-1)
    result.melodicRange = frame.music.texture_density;
    
    return result;
}

} // namespace intent_ir
} // namespace kelly
