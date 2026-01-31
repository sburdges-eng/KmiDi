#pragma once

/**
 * IntentIRExtractor - Helper utilities for engines to extract values from IntentFrame
 *
 * Provides convenient accessors for engines to read IntentFrame fields
 * without needing to know the full structure.
 */

#include "kmidi/IntentIR.h"

namespace kelly {

/**
 * Extract melody-specific parameters from IntentFrame.
 */
struct MelodyParams {
    float melodic_activity;
    float contour_variance;
    float harmonic_tension;
    int8_t mode_preference;

    static MelodyParams fromIntentFrame(const IntentFrame& frame) {
        return {
            .melodic_activity = frame.music.melodic_activity,
            .contour_variance = frame.music.contour_variance,
            .harmonic_tension = frame.music.harmonic_tension,
            .mode_preference = frame.music.mode_preference,
        };
    }
};

/**
 * Extract rhythm/groove-specific parameters from IntentFrame.
 */
struct RhythmParams {
    float rhythmic_density;
    float groove_strength;
    float tempo_bias;

    static RhythmParams fromIntentFrame(const IntentFrame& frame) {
        return {
            .rhythmic_density = frame.music.rhythmic_density,
            .groove_strength = frame.music.groove_strength,
            .tempo_bias = frame.music.tempo_bias,
        };
    }
};

/**
 * Extract harmony-specific parameters from IntentFrame.
 */
struct HarmonyParams {
    float harmonic_tension;
    float harmonic_motion;
    int8_t mode_preference;

    static HarmonyParams fromIntentFrame(const IntentFrame& frame) {
        return {
            .harmonic_tension = frame.music.harmonic_tension,
            .harmonic_motion = frame.music.harmonic_motion,
            .mode_preference = frame.music.mode_preference,
        };
    }
};

/**
 * Extract dynamics/texture parameters from IntentFrame.
 */
struct DynamicsParams {
    float dynamic_range;
    float texture_density;

    static DynamicsParams fromIntentFrame(const IntentFrame& frame) {
        return {
            .dynamic_range = frame.music.dynamic_range,
            .texture_density = frame.music.texture_density,
        };
    }
};

/**
 * Extract emotion parameters from IntentFrame.
 */
struct EmotionParams {
    float valence;
    float arousal;
    float dominance;
    float confidence;

    static EmotionParams fromIntentFrame(const IntentFrame& frame) {
        return {
            .valence = frame.emotion.valence,
            .arousal = frame.emotion.arousal,
            .dominance = frame.emotion.dominance,
            .confidence = frame.emotion.confidence,
        };
    }
};

} // namespace kelly
