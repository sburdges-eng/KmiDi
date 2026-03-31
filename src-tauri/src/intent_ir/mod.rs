//! Intent IR v1 - Canonical representation of musical intent
//!
//! This module defines the IntentFrame structure that serves as the single
//! source of truth for musical intent, independent of UI, ML models, or DSP implementation.

#![allow(dead_code, unused_imports)]

pub mod validator;
pub mod serde;
pub mod ffi;
pub mod manager;

#[cfg(feature = "ffi")]
pub mod ffi_exports;

use ::serde::{Deserialize, Serialize};

/// Intent IR version (increments on breaking changes)
pub const INTENT_IR_VERSION: u16 = 1;

/// Intent Meta - Routing, compatibility, debugging
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct IntentMeta {
    /// IR version (e.g., 1)
    pub ir_version: u16,
    /// Stable hash for this intent
    pub intent_id: u64,
    /// Session-scoped ID
    pub session_id: u64,
}

impl Default for IntentMeta {
    fn default() -> Self {
        Self {
            ir_version: INTENT_IR_VERSION,
            intent_id: 0,
            session_id: 0,
        }
    }
}

/// Emotion State - VAD coordinates with optional discrete mapping
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct EmotionState {
    /// Valence: -1.0 (negative) to +1.0 (positive)
    pub valence: f32,
    /// Arousal: 0.0 (calm) to 1.0 (excited)
    pub arousal: f32,
    /// Dominance: 0.0 (submissive) to 1.0 (dominant)
    pub dominance: f32,
    
    /// Optional discrete emotion ID (EmotionThesaurus ID)
    /// -1 if unused
    pub discrete_id: i16,
    /// Intensity: 0.0 to 1.0
    pub intensity: f32,
    
    /// Confidence in this emotional reading: 0.0 to 1.0
    pub confidence: f32,
}

impl Default for EmotionState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            dominance: 0.5,
            discrete_id: -1,
            intensity: 0.0,
            confidence: 0.0,
        }
    }
}

/// Musical Intent - Biases and tendencies (no notes, no MIDI)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct MusicalIntent {
    /// Tempo bias: -1.0 (slower) to +1.0 (faster)
    pub tempo_bias: f32,
    /// Rhythmic density: 0.0 (sparse) to 1.0 (dense)
    pub rhythmic_density: f32,
    /// Groove strength: 0.0 (loose) to 1.0 (tight)
    pub groove_strength: f32,
    
    /// Harmonic tension: 0.0 (consonant) to 1.0 (tense)
    pub harmonic_tension: f32,
    /// Harmonic motion: 0.0 (static) to 1.0 (active)
    pub harmonic_motion: f32,
    /// Mode preference: -1 (minor), 0 (neutral), +1 (major)
    pub mode_preference: i8,
    
    /// Melodic activity: 0.0 (minimal) to 1.0 (active)
    pub melodic_activity: f32,
    /// Contour variance: 0.0 (flat) to 1.0 (wide)
    pub contour_variance: f32,
    
    /// Dynamic range: 0.0 (flat) to 1.0 (wide)
    pub dynamic_range: f32,
    /// Texture density: 0.0 (thin) to 1.0 (thick)
    pub texture_density: f32,
}

impl Default for MusicalIntent {
    fn default() -> Self {
        Self {
            tempo_bias: 0.0,
            rhythmic_density: 0.5,
            groove_strength: 0.5,
            harmonic_tension: 0.5,
            harmonic_motion: 0.5,
            mode_preference: 0,
            melodic_activity: 0.5,
            contour_variance: 0.5,
            dynamic_range: 0.5,
            texture_density: 0.5,
        }
    }
}

/// Time Scope - Intent without time is noise
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct TimeScope {
    /// Start bar (inclusive), -1 = immediate
    pub start_bar: i32,
    /// End bar (exclusive), -1 = open-ended
    pub end_bar: i32,
    /// Fade in beats: 0.0 = hard
    pub fade_in_beats: f32,
    /// Fade out beats: 0.0 = hard
    pub fade_out_beats: f32,
}

impl Default for TimeScope {
    fn default() -> Self {
        Self {
            start_bar: -1,
            end_bar: -1,
            fade_in_beats: 0.0,
            fade_out_beats: 0.0,
        }
    }
}

/// Intent Constraints - Limit generation, not force it
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct IntentConstraints {
    /// Bitmask of allowed engines
    pub allowed_engines_mask: u32,
    /// Bitmask of forbidden engines
    pub forbidden_engines_mask: u32,
    /// Max CPU cost (hint, not guarantee)
    pub max_cpu_cost: f32,
    /// Max event rate
    pub max_event_rate: f32,
}

impl Default for IntentConstraints {
    fn default() -> Self {
        Self {
            allowed_engines_mask: u32::MAX,
            forbidden_engines_mask: 0,
            max_cpu_cost: 1.0,
            max_event_rate: f32::MAX,
        }
    }
}

/// Intent Source - Provenance tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum IntentSource {
    /// Direct UI input
    UiDirect = 0,
    /// UI edit of existing intent
    UiEdit = 1,
    /// ML text analysis
    MlText = 2,
    /// ML audio analysis
    MlAudio = 3,
    /// Preset
    Preset = 4,
    /// Automation
    Automation = 5,
}

/// Intent Provenance - Debugging and trust
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct IntentProvenance {
    /// Source of this intent
    pub source: IntentSource,
    /// User override weight: 0.0 (ML dominates) to 1.0 (user dominates)
    pub user_override_weight: f32,
}

impl Default for IntentProvenance {
    fn default() -> Self {
        Self {
            source: IntentSource::UiDirect,
            user_override_weight: 1.0,
        }
    }
}

/// IntentFrame - Top-level unit representing one musical intention
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct IntentFrame {
    pub meta: IntentMeta,
    pub emotion: EmotionState,
    pub music: MusicalIntent,
    pub time: TimeScope,
    pub constraints: IntentConstraints,
    pub provenance: IntentProvenance,
}

impl Default for IntentFrame {
    fn default() -> Self {
        Self {
            meta: IntentMeta::default(),
            emotion: EmotionState::default(),
            music: MusicalIntent::default(),
            time: TimeScope::default(),
            constraints: IntentConstraints::default(),
            provenance: IntentProvenance::default(),
        }
    }
}

impl IntentFrame {
    /// Create a new IntentFrame with default values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a new IntentFrame with specific intent and session IDs
    pub fn with_ids(intent_id: u64, session_id: u64) -> Self {
        Self {
            meta: IntentMeta {
                ir_version: INTENT_IR_VERSION,
                intent_id,
                session_id,
            },
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_intent_frame_default() {
        let frame = IntentFrame::default();
        assert_eq!(frame.meta.ir_version, INTENT_IR_VERSION);
        assert_eq!(frame.emotion.valence, 0.0);
        assert_eq!(frame.music.tempo_bias, 0.0);
    }
    
    #[test]
    fn test_intent_frame_with_ids() {
        let frame = IntentFrame::with_ids(123, 456);
        assert_eq!(frame.meta.intent_id, 123);
        assert_eq!(frame.meta.session_id, 456);
    }
}
