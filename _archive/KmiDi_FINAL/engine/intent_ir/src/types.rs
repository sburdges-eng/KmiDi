//! Type definitions matching C structs

pub const INTENT_IR_VERSION: u16 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntentSource {
    UiDirect = 0,
    UiEdit = 1,
    MlText = 2,
    MlAudio = 3,
    Preset = 4,
    Automation = 5,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IntentMeta {
    pub ir_version: u16,
    pub intent_id: u64,
    pub session_id: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmotionState {
    pub valence: f32,      // [-1.0, 1.0]
    pub arousal: f32,      // [0.0, 1.0]
    pub dominance: f32,    // [0.0, 1.0]
    pub discrete_id: i16,  // -1 if unused
    pub intensity: f32,    // [0.0, 1.0]
    pub confidence: f32,    // [0.0, 1.0]
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MusicalIntent {
    pub tempo_bias: f32,        // -1.0 → +1.0
    pub rhythmic_density: f32,  // 0.0 → 1.0
    pub groove_strength: f32,   // 0.0 → 1.0
    pub harmonic_tension: f32,   // 0.0 → 1.0
    pub harmonic_motion: f32,   // 0.0 → 1.0
    pub mode_preference: i8,     // -1, 0, +1
    pub melodic_activity: f32,   // 0.0 → 1.0
    pub contour_variance: f32, // 0.0 → 1.0
    pub dynamic_range: f32,     // 0.0 → 1.0
    pub texture_density: f32,   // 0.0 → 1.0
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TimeScope {
    pub start_bar: i32,      // -1 = immediate
    pub end_bar: i32,        // -1 = open-ended
    pub fade_in_beats: f32,
    pub fade_out_beats: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IntentConstraints {
    pub allowed_engines_mask: u32,
    pub forbidden_engines_mask: u32,
    pub max_cpu_cost: f32,
    pub max_event_rate: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IntentProvenance {
    pub source: IntentSource,
    pub user_override_weight: f32, // 0.0 → 1.0
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
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
            meta: IntentMeta {
                ir_version: INTENT_IR_VERSION,
                intent_id: 0,
                session_id: 0,
            },
            emotion: EmotionState {
                valence: 0.0,
                arousal: 0.5,
                dominance: 0.5,
                discrete_id: -1,
                intensity: 0.0,
                confidence: 1.0,
            },
            music: MusicalIntent {
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
            },
            time: TimeScope {
                start_bar: -1,
                end_bar: -1,
                fade_in_beats: 0.0,
                fade_out_beats: 0.0,
            },
            constraints: IntentConstraints {
                allowed_engines_mask: 0xFFFFFFFF,
                forbidden_engines_mask: 0,
                max_cpu_cost: 1.0,
                max_event_rate: 1000.0,
            },
            provenance: IntentProvenance {
                source: IntentSource::UiDirect,
                user_override_weight: 0.5,
            },
        }
    }
}
