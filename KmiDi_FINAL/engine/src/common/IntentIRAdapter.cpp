#include "IntentIRAdapter.h"
#include <algorithm>
#include <cmath>

namespace kelly {

// Helper: Map tempo_bias (-1.0 to +1.0) to BPM
// Assumes center (0.0) = 120 BPM, range is ±60 BPM
static int tempoBiasToBPM(float tempo_bias) {
    float normalized = std::clamp(tempo_bias, -1.0f, 1.0f);
    return static_cast<int>(120.0f + (normalized * 60.0f));
}

// Helper: Map mode_preference to mode string
static std::string modePreferenceToMode(int8_t mode_preference) {
    if (mode_preference > 0) {
        return "major";
    } else if (mode_preference < 0) {
        return "minor";
    } else {
        return "major";  // Default to major for neutral
    }
}

// Helper: Map rhythmic_density to swing amount
static float densityToSwing(float rhythmic_density) {
    // Higher density = less swing (more straight)
    return std::clamp(1.0f - rhythmic_density, 0.0f, 1.0f) * 0.3f;  // Max 30% swing
}

// Helper: Map groove_strength to syncopation
static float grooveStrengthToSyncopation(float groove_strength) {
    return std::clamp(groove_strength, 0.0f, 1.0f) * 0.5f;  // Max 50% syncopation
}

// Helper: Map harmonic_tension to allowChromaticism
static bool tensionToChromaticism(float harmonic_tension) {
    return harmonic_tension > 0.6f;  // High tension allows chromaticism
}

// Helper: Map melodic_activity to melodic range
static float activityToRange(float melodic_activity) {
    return std::clamp(melodic_activity, 0.0f, 1.0f) * 0.8f + 0.2f;  // 0.2 to 1.0
}

// Helper: Map contour_variance to leap probability
static float varianceToLeapProbability(float contour_variance) {
    return std::clamp(contour_variance, 0.0f, 1.0f) * 0.4f + 0.1f;  // 0.1 to 0.5
}

// Helper: Map dynamic_range to base velocity and dynamic range
static float rangeToBaseVelocity(float dynamic_range) {
    // Higher dynamic range = lower base velocity (more room to grow)
    return std::clamp(1.0f - dynamic_range * 0.3f, 0.4f, 0.8f);
}

static float rangeToDynamicRange(float dynamic_range) {
    return std::clamp(dynamic_range, 0.0f, 1.0f) * 0.6f;  // Max 60% dynamic range
}

IntentResult convertIntentIRToIntentResult(const IntentFrame& frame) {
    IntentResult result;

    // Meta - not directly mappable to IntentResult
    // IntentResult doesn't have version tracking

    // Emotion - map to EmotionNode
    result.emotion.valence = frame.emotion.valence;
    result.emotion.arousal = frame.emotion.arousal;
    result.emotion.dominance = frame.emotion.dominance;
    result.emotion.intensity = frame.emotion.intensity;
    result.emotion.mlConfidence = frame.emotion.confidence;
    // discrete_id not directly mappable

    // Musical parameters from IR biases
    result.tempoBpm = tempoBiasToBPM(frame.music.tempo_bias);
    result.mode = modePreferenceToMode(frame.music.mode_preference);
    result.key = "C";  // Default - IR doesn't specify key

    // Time signature - default 4/4 (IR doesn't specify)
    result.timeSignature = {4, 4};

    // Harmonic choices - IR doesn't specify chords, use empty
    result.chordProgression.clear();

    // Rule breaks - IR doesn't specify rule breaks directly
    result.ruleBreaks.clear();

    // Melodic guidance
    result.melodicRange = activityToRange(frame.music.melodic_activity);
    result.leapProbability = varianceToLeapProbability(frame.music.contour_variance);
    result.allowChromaticism = tensionToChromaticism(frame.music.harmonic_tension);
    result.allowDissonance = result.allowChromaticism;

    // Rhythmic guidance
    result.swingAmount = densityToSwing(frame.music.rhythmic_density);
    result.syncopationLevel = grooveStrengthToSyncopation(frame.music.groove_strength);
    result.humanization = frame.music.groove_strength * 0.2f;  // Max 20% humanization

    // Dynamics
    result.baseVelocity = rangeToBaseVelocity(frame.music.dynamic_range);
    result.dynamicRange = rangeToDynamicRange(frame.music.dynamic_range);

    // Production notes - IR doesn't specify
    result.productionNotes.clear();
    result.narrativeArc = "";  // IR doesn't specify narrative arc as string

    // Source tracking
    result.sourceWound = Wound();  // IR doesn't preserve wound
    result.confidence = frame.emotion.confidence;

    // Compatibility fields
    result.tempo = static_cast<float>(result.tempoBpm) / 120.0f;  // Normalize to 1.0 = 120 BPM

    return result;
}

IntentFrame convertIntentResultToIntentIR(const IntentResult& result) {
    IntentFrame frame;

    // Meta
    frame.meta.ir_version = INTENT_IR_VERSION;
    frame.meta.intent_id = 0;  // Not available in IntentResult
    frame.meta.session_id = 0;  // Not available in IntentResult

    // Emotion
    frame.emotion.valence = result.emotion.valence;
    frame.emotion.arousal = result.emotion.arousal;
    frame.emotion.dominance = result.emotion.dominance;
    frame.emotion.discrete_id = -1;  // Not available
    frame.emotion.intensity = result.emotion.intensity;
    frame.emotion.confidence = result.confidence;

    // Musical Intent - reverse map from concrete params
    // Tempo bias: map BPM to -1.0 to +1.0 (center at 120)
    float tempo_normalized = (static_cast<float>(result.tempoBpm) - 120.0f) / 60.0f;
    frame.music.tempo_bias = std::clamp(tempo_normalized, -1.0f, 1.0f);

    // Rhythmic density: inverse of swing (more swing = less density)
    frame.music.rhythmic_density = std::clamp(1.0f - (result.swingAmount / 0.3f), 0.0f, 1.0f);

    // Groove strength: map from syncopation
    frame.music.groove_strength = std::clamp(result.syncopationLevel / 0.5f, 0.0f, 1.0f);

    // Harmonic tension: map from chromaticism
    frame.music.harmonic_tension = result.allowChromaticism ? 0.7f : 0.3f;

    // Harmonic motion: not directly available, use default
    frame.music.harmonic_motion = 0.5f;

    // Mode preference: map from mode string
    std::string mode_lower = result.mode;
    std::transform(mode_lower.begin(), mode_lower.end(), mode_lower.begin(), ::tolower);
    if (mode_lower.find("minor") != std::string::npos) {
        frame.music.mode_preference = -1;
    } else if (mode_lower.find("major") != std::string::npos) {
        frame.music.mode_preference = 1;
    } else {
        frame.music.mode_preference = 0;
    }

    // Melodic activity: map from melodic range
    frame.music.melodic_activity = std::clamp((result.melodicRange - 0.2f) / 0.8f, 0.0f, 1.0f);

    // Contour variance: map from leap probability
    frame.music.contour_variance = std::clamp((result.leapProbability - 0.1f) / 0.4f, 0.0f, 1.0f);

    // Dynamic range: map from base velocity and dynamic range
    float base_vel_normalized = (result.baseVelocity - 0.4f) / 0.4f;  // 0.4-0.8 -> 0.0-1.0
    frame.music.dynamic_range = std::clamp((1.0f - base_vel_normalized) + (result.dynamicRange / 0.6f), 0.0f, 1.0f);

    // Texture density: not directly available, use default
    frame.music.texture_density = 0.5f;

    // Time scope - not available in IntentResult, use defaults
    frame.time.start_bar = -1;
    frame.time.end_bar = -1;
    frame.time.fade_in_beats = 0.0f;
    frame.time.fade_out_beats = 0.0f;

    // Constraints - not available in IntentResult, use defaults
    frame.constraints.allowed_engines_mask = 0xFFFFFFFF;
    frame.constraints.forbidden_engines_mask = 0;
    frame.constraints.max_cpu_cost = 1.0f;
    frame.constraints.max_event_rate = 1000.0f;

    // Provenance - not available in IntentResult, use default
    frame.provenance.source = SOURCE_UI_DIRECT;
    frame.provenance.user_override_weight = 0.5f;

    return frame;
}

bool isIntentIRVersionSupported(uint16_t version) {
    return version == INTENT_IR_VERSION;
}

void prepareIntentFrame(IntentFrame& frame) {
    // Clamp all values to valid ranges
    // This should call the Rust validator, but for now we do basic clamping

    // EmotionState
    frame.emotion.valence = std::clamp(frame.emotion.valence, -1.0f, 1.0f);
    frame.emotion.arousal = std::clamp(frame.emotion.arousal, 0.0f, 1.0f);
    frame.emotion.dominance = std::clamp(frame.emotion.dominance, 0.0f, 1.0f);
    frame.emotion.intensity = std::clamp(frame.emotion.intensity, 0.0f, 1.0f);
    frame.emotion.confidence = std::clamp(frame.emotion.confidence, 0.0f, 1.0f);

    // MusicalIntent
    frame.music.tempo_bias = std::clamp(frame.music.tempo_bias, -1.0f, 1.0f);
    frame.music.rhythmic_density = std::clamp(frame.music.rhythmic_density, 0.0f, 1.0f);
    frame.music.groove_strength = std::clamp(frame.music.groove_strength, 0.0f, 1.0f);
    frame.music.harmonic_tension = std::clamp(frame.music.harmonic_tension, 0.0f, 1.0f);
    frame.music.harmonic_motion = std::clamp(frame.music.harmonic_motion, 0.0f, 1.0f);
    frame.music.mode_preference = std::clamp(frame.music.mode_preference, static_cast<int8_t>(-1), static_cast<int8_t>(1));
    frame.music.melodic_activity = std::clamp(frame.music.melodic_activity, 0.0f, 1.0f);
    frame.music.contour_variance = std::clamp(frame.music.contour_variance, 0.0f, 1.0f);
    frame.music.dynamic_range = std::clamp(frame.music.dynamic_range, 0.0f, 1.0f);
    frame.music.texture_density = std::clamp(frame.music.texture_density, 0.0f, 1.0f);

    // TimeScope
    frame.time.fade_in_beats = std::max(0.0f, frame.time.fade_in_beats);
    frame.time.fade_out_beats = std::max(0.0f, frame.time.fade_out_beats);

    // IntentConstraints
    frame.constraints.max_cpu_cost = std::max(0.0f, frame.constraints.max_cpu_cost);
    frame.constraints.max_event_rate = std::max(0.0f, frame.constraints.max_event_rate);

    // IntentProvenance
    frame.provenance.user_override_weight = std::clamp(frame.provenance.user_override_weight, 0.0f, 1.0f);
}

} // namespace kelly
