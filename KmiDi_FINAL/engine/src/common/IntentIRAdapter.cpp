#include "IntentIRAdapter.h"
#include "kmidi/IntentIR.h"  // IntentIR types (needed before FFI header)
#include "intent_ir_ffi.h"  // Rust FFI functions (includes IntentIR.h)
#include "penta/common/RTLogger.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unwind.h>
#include <vector>

namespace kelly {

extern "C" _Unwind_Reason_Code __gcc_personality_v0(
    int version,
    _Unwind_Action actions,
    std::uint64_t exceptionClass,
    _Unwind_Exception* exceptionObject,
    _Unwind_Context* context
);

// Keep this in a frequently-linked TU so static archive link order does not drop it.
extern "C" _Unwind_Reason_Code rust_eh_personality(
    int version,
    _Unwind_Action actions,
    std::uint64_t exceptionClass,
    _Unwind_Exception* exceptionObject,
    _Unwind_Context* context
) {
    return __gcc_personality_v0(version, actions, exceptionClass, exceptionObject, context);
}

// Helper: Map tempo_bias (-1.0 to +1.0) to BPM
// Assumes center (0.0) = 120 BPM, range is ±60 BPM
static int tempoBiasToBPM(float tempo_bias) {
    // Validate input
    if (std::isnan(tempo_bias) || std::isinf(tempo_bias)) {
        penta::getLogger().logRT(penta::LogLevel::Error,
            ("IntentIRAdapter::tempoBiasToBPM: Invalid tempo_bias (NaN/Inf): " +
             std::to_string(tempo_bias)).c_str());
        return 120; // Default to 120 BPM
    }

    float normalized = std::clamp(tempo_bias, -1.0f, 1.0f);
    float bpm_float = 120.0f + (normalized * 60.0f);

    // Check for overflow
    if (bpm_float < static_cast<float>(std::numeric_limits<int>::min()) ||
        bpm_float > static_cast<float>(std::numeric_limits<int>::max())) {
        penta::getLogger().logRT(penta::LogLevel::Error,
            "IntentIRAdapter::tempoBiasToBPM: BPM calculation would overflow");
        return 120; // Default
    }

    int bpm = static_cast<int>(std::round(bpm_float));

    // Validate result
    if (bpm < 1 || bpm > 300) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            ("IntentIRAdapter::tempoBiasToBPM: BPM out of typical range: " +
             std::to_string(bpm)).c_str());
    }

    return bpm;
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
    // Validate frame before conversion
    if (frame.meta.ir_version != INTENT_IR_VERSION) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            ("IntentIRAdapter::convertIntentIRToIntentResult: IR version mismatch: " +
             std::to_string(frame.meta.ir_version) + " (expected " +
             std::to_string(INTENT_IR_VERSION) + ")").c_str());
    }

    IntentResult result;

    // Meta - not directly mappable to IntentResult
    // IntentResult doesn't have version tracking

    // Emotion - map to EmotionNode with validation
    result.emotion.valence = std::clamp(frame.emotion.valence, -1.0f, 1.0f);
    result.emotion.arousal = std::clamp(frame.emotion.arousal, 0.0f, 1.0f);
    result.emotion.dominance = std::clamp(frame.emotion.dominance, 0.0f, 1.0f);
    result.emotion.intensity = std::clamp(frame.emotion.intensity, 0.0f, 1.0f);
    result.emotion.mlConfidence = std::clamp(frame.emotion.confidence, 0.0f, 1.0f);
    // discrete_id not directly mappable

    // Log if values were clamped
    if (frame.emotion.valence != result.emotion.valence ||
        frame.emotion.arousal != result.emotion.arousal ||
        frame.emotion.dominance != result.emotion.dominance ||
        frame.emotion.intensity != result.emotion.intensity) {
        penta::getLogger().logRT(penta::LogLevel::Debug,
            "IntentIRAdapter::convertIntentIRToIntentResult: Emotion values clamped");
    }

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
    // Pre-validation checks
    std::vector<std::string> warnings;
    std::vector<std::string> errors;

    // Check IR version
    if (frame.meta.ir_version != INTENT_IR_VERSION) {
        warnings.push_back("IR version mismatch: " + std::to_string(frame.meta.ir_version) +
                          " (expected " + std::to_string(INTENT_IR_VERSION) + ")");
    }

    // Validate emotion values before Rust call
    if (frame.emotion.valence < -1.0f || frame.emotion.valence > 1.0f) {
        warnings.push_back("Emotion valence out of range: " + std::to_string(frame.emotion.valence) +
                          ", will be clamped");
    }

    if (frame.emotion.arousal < 0.0f || frame.emotion.arousal > 1.0f) {
        warnings.push_back("Emotion arousal out of range: " + std::to_string(frame.emotion.arousal) +
                          ", will be clamped");
    }

    if (frame.emotion.dominance < 0.0f || frame.emotion.dominance > 1.0f) {
        warnings.push_back("Emotion dominance out of range: " + std::to_string(frame.emotion.dominance) +
                          ", will be clamped");
    }

    if (frame.emotion.intensity < 0.0f || frame.emotion.intensity > 1.0f) {
        warnings.push_back("Emotion intensity out of range: " + std::to_string(frame.emotion.intensity) +
                          ", will be clamped");
    }

    // Validate music parameters
    if (frame.music.tempo_bias < -1.0f || frame.music.tempo_bias > 1.0f) {
        warnings.push_back("Tempo bias out of range: " + std::to_string(frame.music.tempo_bias) +
                          ", will be clamped");
    }

    // Log warnings
    for (const auto& warning : warnings) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            ("IntentIRAdapter::prepareIntentFrame: " + warning).c_str());
    }

    // Use shared IntentIR API so adapter code stays decoupled from raw FFI symbols.
    intent_frame_clamp(&frame);

    // Post-validation checks
    if (frame.emotion.valence < -1.0f || frame.emotion.valence > 1.0f) {
        errors.push_back("Emotion valence still out of range after clamping: " +
                        std::to_string(frame.emotion.valence));
    }

    // Log errors
    for (const auto& error : errors) {
        penta::getLogger().logRT(penta::LogLevel::Error,
            ("IntentIRAdapter::prepareIntentFrame: " + error).c_str());
    }
}

} // namespace kelly
