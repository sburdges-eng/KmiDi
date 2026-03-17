//! Safe construction API for IntentFrame

use crate::types::*;
use crate::validator::{clamp_intent_frame, validate_intent_frame, ValidationResult};

/// Builder for IntentFrame with method chaining
pub struct IntentFrameBuilder {
    frame: IntentFrame,
}

impl IntentFrameBuilder {
    pub fn new() -> Self {
        Self {
            frame: IntentFrame::default(),
        }
    }

    pub fn with_meta(mut self, ir_version: u16, intent_id: u64, session_id: u64) -> Self {
        self.frame.meta.ir_version = ir_version;
        self.frame.meta.intent_id = intent_id;
        self.frame.meta.session_id = session_id;
        self
    }

    pub fn with_emotion(
        mut self,
        valence: f32,
        arousal: f32,
        dominance: f32,
        discrete_id: i16,
        intensity: f32,
        confidence: f32,
    ) -> Self {
        self.frame.emotion.valence = valence;
        self.frame.emotion.arousal = arousal;
        self.frame.emotion.dominance = dominance;
        self.frame.emotion.discrete_id = discrete_id;
        self.frame.emotion.intensity = intensity;
        self.frame.emotion.confidence = confidence;
        self
    }

    pub fn with_musical_intent(
        mut self,
        tempo_bias: f32,
        rhythmic_density: f32,
        groove_strength: f32,
        harmonic_tension: f32,
        harmonic_motion: f32,
        mode_preference: i8,
        melodic_activity: f32,
        contour_variance: f32,
        dynamic_range: f32,
        texture_density: f32,
    ) -> Self {
        self.frame.music.tempo_bias = tempo_bias;
        self.frame.music.rhythmic_density = rhythmic_density;
        self.frame.music.groove_strength = groove_strength;
        self.frame.music.harmonic_tension = harmonic_tension;
        self.frame.music.harmonic_motion = harmonic_motion;
        self.frame.music.mode_preference = mode_preference;
        self.frame.music.melodic_activity = melodic_activity;
        self.frame.music.contour_variance = contour_variance;
        self.frame.music.dynamic_range = dynamic_range;
        self.frame.music.texture_density = texture_density;
        self
    }

    pub fn with_time_scope(
        mut self,
        start_bar: i32,
        end_bar: i32,
        fade_in_beats: f32,
        fade_out_beats: f32,
    ) -> Self {
        self.frame.time.start_bar = start_bar;
        self.frame.time.end_bar = end_bar;
        self.frame.time.fade_in_beats = fade_in_beats;
        self.frame.time.fade_out_beats = fade_out_beats;
        self
    }

    pub fn with_constraints(
        mut self,
        allowed_engines_mask: u32,
        forbidden_engines_mask: u32,
        max_cpu_cost: f32,
        max_event_rate: f32,
    ) -> Self {
        self.frame.constraints.allowed_engines_mask = allowed_engines_mask;
        self.frame.constraints.forbidden_engines_mask = forbidden_engines_mask;
        self.frame.constraints.max_cpu_cost = max_cpu_cost;
        self.frame.constraints.max_event_rate = max_event_rate;
        self
    }

    pub fn with_provenance(
        mut self,
        source: IntentSource,
        user_override_weight: f32,
    ) -> Self {
        self.frame.provenance.source = source;
        self.frame.provenance.user_override_weight = user_override_weight;
        self
    }

    /// Build and validate the IntentFrame
    /// Returns Ok(()) if valid, Err(error) otherwise
    pub fn build(mut self) -> ValidationResult {
        // Clamp values first
        clamp_intent_frame(&mut self.frame);

        // Then validate
        validate_intent_frame(&self.frame)?;

        Ok(())
    }

    /// Build without validation (use with caution)
    /// Returns the built frame
    pub fn build_unchecked(mut self) -> IntentFrame {
        clamp_intent_frame(&mut self.frame);
        self.frame
    }

    /// Get the frame (consumes builder)
    pub fn into_frame(mut self) -> IntentFrame {
        clamp_intent_frame(&mut self.frame);
        self.frame
    }

    /// Get mutable reference to frame (for advanced use)
    pub fn frame_mut(&mut self) -> &mut IntentFrame {
        &mut self.frame
    }
}

impl Default for IntentFrameBuilder {
    fn default() -> Self {
        Self::new()
    }
}
