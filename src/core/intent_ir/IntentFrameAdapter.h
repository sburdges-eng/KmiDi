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
    params.tempoBpm = static_cast<int>(120.0f * tempo_multiplier);
    
    // Map mode preference
    if (music.mode_preference > 0) {
        params.mode = "major";
    } else if (music.mode_preference < 0) {
        params.mode = "minor";
    } else {
        params.mode = "neutral";
    }
    
    // Map other parameters (these are approximations)
    params.dynamicRange = music.dynamic_range;
    params.textureDensity = music.texture_density;
    params.harmonicTension = music.harmonic_tension;
    
    return params;
}

/**
 * Convert IntentFrame to IntentResult (for backward compatibility)
 */
inline IntentResult frameToIntentResult(const IntentFrame& frame) {
    IntentResult result;
    
    // Extract emotion
    result.valence = frame.emotion.valence;
    result.arousal = frame.emotion.arousal;
    result.dominance = frame.emotion.dominance;
    
    // Extract music parameters
    auto params = musicToParams(frame.music);
    result.bpm = params.tempoBpm;
    result.key = "C"; // Default, would need more context
    result.mode = params.mode;
    
    // Set complexity from texture density
    result.complexity = frame.music.texture_density;
    
    return result;
}

} // namespace intent_ir
} // namespace kelly
