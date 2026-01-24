//! Validation and clamping for IntentFrame

use crate::types::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValidationError {
    InvalidVersion,
    InvalidValence,
    InvalidArousal,
    InvalidDominance,
    InvalidIntensity,
    InvalidConfidence,
    InvalidTempoBias,
    InvalidDensity,
    InvalidModePreference,
    InvalidTimeScope,
    InvalidUserOverrideWeight,
}

pub type ValidationResult = Result<(), ValidationError>;

/// Clamp all float values in an IntentFrame to valid ranges
pub fn clamp_intent_frame(frame: &mut IntentFrame) {
    // Meta - no clamping needed (version checked separately)

    // EmotionState
    frame.emotion.valence = frame.emotion.valence.clamp(-1.0, 1.0);
    frame.emotion.arousal = frame.emotion.arousal.clamp(0.0, 1.0);
    frame.emotion.dominance = frame.emotion.dominance.clamp(0.0, 1.0);
    frame.emotion.intensity = frame.emotion.intensity.clamp(0.0, 1.0);
    frame.emotion.confidence = frame.emotion.confidence.clamp(0.0, 1.0);

    // MusicalIntent
    frame.music.tempo_bias = frame.music.tempo_bias.clamp(-1.0, 1.0);
    frame.music.rhythmic_density = frame.music.rhythmic_density.clamp(0.0, 1.0);
    frame.music.groove_strength = frame.music.groove_strength.clamp(0.0, 1.0);
    frame.music.harmonic_tension = frame.music.harmonic_tension.clamp(0.0, 1.0);
    frame.music.harmonic_motion = frame.music.harmonic_motion.clamp(0.0, 1.0);
    frame.music.mode_preference = frame.music.mode_preference.clamp(-1, 1);
    frame.music.melodic_activity = frame.music.melodic_activity.clamp(0.0, 1.0);
    frame.music.contour_variance = frame.music.contour_variance.clamp(0.0, 1.0);
    frame.music.dynamic_range = frame.music.dynamic_range.clamp(0.0, 1.0);
    frame.music.texture_density = frame.music.texture_density.clamp(0.0, 1.0);

    // TimeScope - no clamping (int32_t and f32 are fine as-is)
    // But ensure fade values are non-negative
    frame.time.fade_in_beats = frame.time.fade_in_beats.max(0.0);
    frame.time.fade_out_beats = frame.time.fade_out_beats.max(0.0);

    // IntentConstraints
    frame.constraints.max_cpu_cost = frame.constraints.max_cpu_cost.max(0.0);
    frame.constraints.max_event_rate = frame.constraints.max_event_rate.max(0.0);

    // IntentProvenance
    frame.provenance.user_override_weight = frame.provenance.user_override_weight.clamp(0.0, 1.0);
}

/// Validate an IntentFrame
pub fn validate_intent_frame(frame: &IntentFrame) -> ValidationResult {
    // Check version
    if frame.meta.ir_version != INTENT_IR_VERSION {
        return Err(ValidationError::InvalidVersion);
    }

    // Validate EmotionState
    if frame.emotion.valence < -1.0 || frame.emotion.valence > 1.0 {
        return Err(ValidationError::InvalidValence);
    }
    if frame.emotion.arousal < 0.0 || frame.emotion.arousal > 1.0 {
        return Err(ValidationError::InvalidArousal);
    }
    if frame.emotion.dominance < 0.0 || frame.emotion.dominance > 1.0 {
        return Err(ValidationError::InvalidDominance);
    }
    if frame.emotion.intensity < 0.0 || frame.emotion.intensity > 1.0 {
        return Err(ValidationError::InvalidIntensity);
    }
    if frame.emotion.confidence < 0.0 || frame.emotion.confidence > 1.0 {
        return Err(ValidationError::InvalidConfidence);
    }

    // Validate MusicalIntent
    // Note: InvalidDensity is used as a generic error for all musical intent range violations
    // (except tempo_bias and mode_preference which have specific errors)
    // This keeps the error enum small while still catching validation failures
    if frame.music.tempo_bias < -1.0 || frame.music.tempo_bias > 1.0 {
        return Err(ValidationError::InvalidTempoBias);
    }
    if frame.music.rhythmic_density < 0.0 || frame.music.rhythmic_density > 1.0 {
        return Err(ValidationError::InvalidDensity);
    }
    if frame.music.groove_strength < 0.0 || frame.music.groove_strength > 1.0 {
        return Err(ValidationError::InvalidDensity);
    }
    if frame.music.harmonic_tension < 0.0 || frame.music.harmonic_tension > 1.0 {
        return Err(ValidationError::InvalidDensity);
    }
    if frame.music.harmonic_motion < 0.0 || frame.music.harmonic_motion > 1.0 {
        return Err(ValidationError::InvalidDensity);
    }
    if frame.music.mode_preference < -1 || frame.music.mode_preference > 1 {
        return Err(ValidationError::InvalidModePreference);
    }
    if frame.music.melodic_activity < 0.0 || frame.music.melodic_activity > 1.0 {
        return Err(ValidationError::InvalidDensity);
    }
    if frame.music.contour_variance < 0.0 || frame.music.contour_variance > 1.0 {
        return Err(ValidationError::InvalidDensity);
    }
    if frame.music.dynamic_range < 0.0 || frame.music.dynamic_range > 1.0 {
        return Err(ValidationError::InvalidDensity);
    }
    if frame.music.texture_density < 0.0 || frame.music.texture_density > 1.0 {
        return Err(ValidationError::InvalidDensity);
    }

    // Validate TimeScope
    // start_bar and end_bar can be -1 (special values)
    // If both are set, end_bar should be > start_bar (unless end_bar is -1)
    if frame.time.end_bar != -1 && frame.time.start_bar != -1 {
        if frame.time.end_bar <= frame.time.start_bar {
            return Err(ValidationError::InvalidTimeScope);
        }
    }

    // Validate IntentProvenance
    if frame.provenance.user_override_weight < 0.0 || frame.provenance.user_override_weight > 1.0 {
        return Err(ValidationError::InvalidUserOverrideWeight);
    }

    Ok(())
}

/// Check if version is supported
pub fn version_supported(version: u16) -> bool {
    version == INTENT_IR_VERSION
}
