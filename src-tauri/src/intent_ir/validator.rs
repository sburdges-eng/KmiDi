//! Intent IR Validator - Validation, version negotiation, clamping

use super::*;
use std::fmt;

/// Validation error
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// Invalid IR version
    InvalidVersion { expected: u16, got: u16 },
    /// Value out of range
    OutOfRange { field: String, min: f32, max: f32, value: f32 },
    /// Invalid discrete emotion ID
    InvalidDiscreteId { value: i16 },
    /// Invalid time scope
    InvalidTimeScope { start: i32, end: i32 },
    /// Invalid constraint values
    InvalidConstraints { reason: String },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::InvalidVersion { expected, got } => {
                write!(f, "Invalid IR version: expected {}, got {}", expected, got)
            }
            ValidationError::OutOfRange { field, min, max, value } => {
                write!(f, "Field '{}' out of range: {} (expected {}-{})", field, value, min, max)
            }
            ValidationError::InvalidDiscreteId { value } => {
                write!(f, "Invalid discrete emotion ID: {} (must be -1 or >= 0)", value)
            }
            ValidationError::InvalidTimeScope { start, end } => {
                write!(f, "Invalid time scope: start={}, end={}", start, end)
            }
            ValidationError::InvalidConstraints { reason } => {
                write!(f, "Invalid constraints: {}", reason)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validation result
pub type ValidationResult = Result<IntentFrame, ValidationError>;

/// Clamp a value to a range
#[inline]
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

/// Clamp a value to a range, returning error if out of bounds
#[inline]
pub fn clamp_or_error(value: f32, min: f32, max: f32, field: &str) -> Result<f32, ValidationError> {
    if value < min || value > max {
        Err(ValidationError::OutOfRange {
            field: field.to_string(),
            min,
            max,
            value,
        })
    } else {
        Ok(value)
    }
}

/// Validate and clamp an IntentFrame
pub fn validate_and_clamp(frame: IntentFrame) -> ValidationResult {
    // Version check
    if frame.meta.ir_version != INTENT_IR_VERSION {
        return Err(ValidationError::InvalidVersion {
            expected: INTENT_IR_VERSION,
            got: frame.meta.ir_version,
        });
    }
    
    // Validate emotion state
    clamp_or_error(frame.emotion.valence, -1.0, 1.0, "emotion.valence")?;
    clamp_or_error(frame.emotion.arousal, 0.0, 1.0, "emotion.arousal")?;
    clamp_or_error(frame.emotion.dominance, 0.0, 1.0, "emotion.dominance")?;
    clamp_or_error(frame.emotion.intensity, 0.0, 1.0, "emotion.intensity")?;
    clamp_or_error(frame.emotion.confidence, 0.0, 1.0, "emotion.confidence")?;
    
    if frame.emotion.discrete_id < -1 {
        return Err(ValidationError::InvalidDiscreteId {
            value: frame.emotion.discrete_id,
        });
    }
    
    // Validate musical intent
    clamp_or_error(frame.music.tempo_bias, -1.0, 1.0, "music.tempo_bias")?;
    clamp_or_error(frame.music.rhythmic_density, 0.0, 1.0, "music.rhythmic_density")?;
    clamp_or_error(frame.music.groove_strength, 0.0, 1.0, "music.groove_strength")?;
    clamp_or_error(frame.music.harmonic_tension, 0.0, 1.0, "music.harmonic_tension")?;
    clamp_or_error(frame.music.harmonic_motion, 0.0, 1.0, "music.harmonic_motion")?;
    clamp_or_error(frame.music.melodic_activity, 0.0, 1.0, "music.melodic_activity")?;
    clamp_or_error(frame.music.contour_variance, 0.0, 1.0, "music.contour_variance")?;
    clamp_or_error(frame.music.dynamic_range, 0.0, 1.0, "music.dynamic_range")?;
    clamp_or_error(frame.music.texture_density, 0.0, 1.0, "music.texture_density")?;
    
    if frame.music.mode_preference < -1 || frame.music.mode_preference > 1 {
        return Err(ValidationError::OutOfRange {
            field: "music.mode_preference".to_string(),
            min: -1.0,
            max: 1.0,
            value: frame.music.mode_preference as f32,
        });
    }
    
    // Validate time scope
    if frame.time.start_bar != -1 && frame.time.end_bar != -1 {
        if frame.time.start_bar >= frame.time.end_bar {
            return Err(ValidationError::InvalidTimeScope {
                start: frame.time.start_bar,
                end: frame.time.end_bar,
            });
        }
    }
    
    if frame.time.fade_in_beats < 0.0 || frame.time.fade_out_beats < 0.0 {
        return Err(ValidationError::InvalidConstraints {
            reason: "Fade times must be >= 0.0".to_string(),
        });
    }
    
    // Validate constraints
    if frame.constraints.max_cpu_cost < 0.0 {
        return Err(ValidationError::InvalidConstraints {
            reason: "max_cpu_cost must be >= 0.0".to_string(),
        });
    }
    
    if frame.constraints.max_event_rate < 0.0 {
        return Err(ValidationError::InvalidConstraints {
            reason: "max_event_rate must be >= 0.0".to_string(),
        });
    }
    
    // Validate provenance
    clamp_or_error(frame.provenance.user_override_weight, 0.0, 1.0, "provenance.user_override_weight")?;
    
    Ok(frame)
}

/// Clamp an IntentFrame (always succeeds, clamps to valid ranges)
pub fn clamp_frame(mut frame: IntentFrame) -> IntentFrame {
    // Version must match
    frame.meta.ir_version = INTENT_IR_VERSION;
    
    // Clamp emotion
    frame.emotion.valence = clamp(frame.emotion.valence, -1.0, 1.0);
    frame.emotion.arousal = clamp(frame.emotion.arousal, 0.0, 1.0);
    frame.emotion.dominance = clamp(frame.emotion.dominance, 0.0, 1.0);
    frame.emotion.intensity = clamp(frame.emotion.intensity, 0.0, 1.0);
    frame.emotion.confidence = clamp(frame.emotion.confidence, 0.0, 1.0);
    if frame.emotion.discrete_id < -1 {
        frame.emotion.discrete_id = -1;
    }
    
    // Clamp musical intent
    frame.music.tempo_bias = clamp(frame.music.tempo_bias, -1.0, 1.0);
    frame.music.rhythmic_density = clamp(frame.music.rhythmic_density, 0.0, 1.0);
    frame.music.groove_strength = clamp(frame.music.groove_strength, 0.0, 1.0);
    frame.music.harmonic_tension = clamp(frame.music.harmonic_tension, 0.0, 1.0);
    frame.music.harmonic_motion = clamp(frame.music.harmonic_motion, 0.0, 1.0);
    frame.music.mode_preference = clamp(frame.music.mode_preference as f32, -1.0, 1.0) as i8;
    frame.music.melodic_activity = clamp(frame.music.melodic_activity, 0.0, 1.0);
    frame.music.contour_variance = clamp(frame.music.contour_variance, 0.0, 1.0);
    frame.music.dynamic_range = clamp(frame.music.dynamic_range, 0.0, 1.0);
    frame.music.texture_density = clamp(frame.music.texture_density, 0.0, 1.0);
    
    // Clamp time scope (fix invalid ranges)
    if frame.time.start_bar != -1 && frame.time.end_bar != -1 {
        if frame.time.start_bar >= frame.time.end_bar {
            frame.time.end_bar = frame.time.start_bar + 1;
        }
    }
    frame.time.fade_in_beats = frame.time.fade_in_beats.max(0.0);
    frame.time.fade_out_beats = frame.time.fade_out_beats.max(0.0);
    
    // Clamp constraints
    frame.constraints.max_cpu_cost = frame.constraints.max_cpu_cost.max(0.0);
    frame.constraints.max_event_rate = frame.constraints.max_event_rate.max(0.0);
    
    // Clamp provenance
    frame.provenance.user_override_weight = clamp(frame.provenance.user_override_weight, 0.0, 1.0);
    
    frame
}

/// Check if an engine supports a given IR version
pub fn supports_version(engine_version: u16, ir_version: u16) -> bool {
    // For now, exact match required
    // Future: could support version ranges
    engine_version == ir_version
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_valid_frame() {
        let frame = IntentFrame::default();
        assert!(validate_and_clamp(frame).is_ok());
    }
    
    #[test]
    fn test_validate_invalid_version() {
        let mut frame = IntentFrame::default();
        frame.meta.ir_version = 999;
        assert!(validate_and_clamp(frame).is_err());
    }
    
    #[test]
    fn test_validate_out_of_range() {
        let mut frame = IntentFrame::default();
        frame.emotion.valence = 2.0;
        assert!(validate_and_clamp(frame).is_err());
    }
    
    #[test]
    fn test_clamp_frame() {
        let mut frame = IntentFrame::default();
        frame.emotion.valence = 2.0;
        frame.emotion.arousal = -1.0;
        let clamped = clamp_frame(frame);
        assert_eq!(clamped.emotion.valence, 1.0);
        assert_eq!(clamped.emotion.arousal, 0.0);
    }
}
