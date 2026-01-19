//! Intent IR FFI - C-compatible structs and conversion functions

use super::*;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_float};
use std::ptr;

/// C-compatible IntentMeta
#[repr(C)]
pub struct CIntentMeta {
    pub ir_version: c_uint16,
    pub intent_id: c_uint64,
    pub session_id: c_uint64,
}

/// C-compatible EmotionState
#[repr(C)]
pub struct CEmotionState {
    pub valence: c_float,
    pub arousal: c_float,
    pub dominance: c_float,
    pub discrete_id: c_int16,
    pub intensity: c_float,
    pub confidence: c_float,
}

/// C-compatible MusicalIntent
#[repr(C)]
pub struct CMusicalIntent {
    pub tempo_bias: c_float,
    pub rhythmic_density: c_float,
    pub groove_strength: c_float,
    pub harmonic_tension: c_float,
    pub harmonic_motion: c_float,
    pub mode_preference: c_int8,
    pub melodic_activity: c_float,
    pub contour_variance: c_float,
    pub dynamic_range: c_float,
    pub texture_density: c_float,
}

/// C-compatible TimeScope
#[repr(C)]
pub struct CTimeScope {
    pub start_bar: c_int32,
    pub end_bar: c_int32,
    pub fade_in_beats: c_float,
    pub fade_out_beats: c_float,
}

/// C-compatible IntentConstraints
#[repr(C)]
pub struct CIntentConstraints {
    pub allowed_engines_mask: c_uint32,
    pub forbidden_engines_mask: c_uint32,
    pub max_cpu_cost: c_float,
    pub max_event_rate: c_float,
}

/// C-compatible IntentSource
#[repr(u8)]
pub enum CIntentSource {
    UiDirect = 0,
    UiEdit = 1,
    MlText = 2,
    MlAudio = 3,
    Preset = 4,
    Automation = 5,
}

/// C-compatible IntentProvenance
#[repr(C)]
pub struct CIntentProvenance {
    pub source: CIntentSource,
    pub user_override_weight: c_float,
}

/// C-compatible IntentFrame
#[repr(C)]
pub struct CIntentFrame {
    pub meta: CIntentMeta,
    pub emotion: CEmotionState,
    pub music: CMusicalIntent,
    pub time: CTimeScope,
    pub constraints: CIntentConstraints,
    pub provenance: CIntentProvenance,
}

/// Convert Rust IntentFrame to C struct
pub fn to_c_frame(frame: &IntentFrame) -> CIntentFrame {
    CIntentFrame {
        meta: CIntentMeta {
            ir_version: frame.meta.ir_version,
            intent_id: frame.meta.intent_id,
            session_id: frame.meta.session_id,
        },
        emotion: CEmotionState {
            valence: frame.emotion.valence,
            arousal: frame.emotion.arousal,
            dominance: frame.emotion.dominance,
            discrete_id: frame.emotion.discrete_id,
            intensity: frame.emotion.intensity,
            confidence: frame.emotion.confidence,
        },
        music: CMusicalIntent {
            tempo_bias: frame.music.tempo_bias,
            rhythmic_density: frame.music.rhythmic_density,
            groove_strength: frame.music.groove_strength,
            harmonic_tension: frame.music.harmonic_tension,
            harmonic_motion: frame.music.harmonic_motion,
            mode_preference: frame.music.mode_preference,
            melodic_activity: frame.music.melodic_activity,
            contour_variance: frame.music.contour_variance,
            dynamic_range: frame.music.dynamic_range,
            texture_density: frame.music.texture_density,
        },
        time: CTimeScope {
            start_bar: frame.time.start_bar,
            end_bar: frame.time.end_bar,
            fade_in_beats: frame.time.fade_in_beats,
            fade_out_beats: frame.time.fade_out_beats,
        },
        constraints: CIntentConstraints {
            allowed_engines_mask: frame.constraints.allowed_engines_mask,
            forbidden_engines_mask: frame.constraints.forbidden_engines_mask,
            max_cpu_cost: frame.constraints.max_cpu_cost,
            max_event_rate: frame.constraints.max_event_rate,
        },
        provenance: CIntentProvenance {
            source: match frame.provenance.source {
                IntentSource::UiDirect => CIntentSource::UiDirect,
                IntentSource::UiEdit => CIntentSource::UiEdit,
                IntentSource::MlText => CIntentSource::MlText,
                IntentSource::MlAudio => CIntentSource::MlAudio,
                IntentSource::Preset => CIntentSource::Preset,
                IntentSource::Automation => CIntentSource::Automation,
            },
            user_override_weight: frame.provenance.user_override_weight,
        },
    }
}

/// Convert C struct to Rust IntentFrame
pub fn from_c_frame(c_frame: &CIntentFrame) -> IntentFrame {
    IntentFrame {
        meta: IntentMeta {
            ir_version: c_frame.meta.ir_version,
            intent_id: c_frame.meta.intent_id,
            session_id: c_frame.meta.session_id,
        },
        emotion: EmotionState {
            valence: c_frame.emotion.valence,
            arousal: c_frame.emotion.arousal,
            dominance: c_frame.emotion.dominance,
            discrete_id: c_frame.emotion.discrete_id,
            intensity: c_frame.emotion.intensity,
            confidence: c_frame.emotion.confidence,
        },
        music: MusicalIntent {
            tempo_bias: c_frame.music.tempo_bias,
            rhythmic_density: c_frame.music.rhythmic_density,
            groove_strength: c_frame.music.groove_strength,
            harmonic_tension: c_frame.music.harmonic_tension,
            harmonic_motion: c_frame.music.harmonic_motion,
            mode_preference: c_frame.music.mode_preference,
            melodic_activity: c_frame.music.melodic_activity,
            contour_variance: c_frame.music.contour_variance,
            dynamic_range: c_frame.music.dynamic_range,
            texture_density: c_frame.music.texture_density,
        },
        time: TimeScope {
            start_bar: c_frame.time.start_bar,
            end_bar: c_frame.time.end_bar,
            fade_in_beats: c_frame.time.fade_in_beats,
            fade_out_beats: c_frame.time.fade_out_beats,
        },
        constraints: IntentConstraints {
            allowed_engines_mask: c_frame.constraints.allowed_engines_mask,
            forbidden_engines_mask: c_frame.constraints.forbidden_engines_mask,
            max_cpu_cost: c_frame.constraints.max_cpu_cost,
            max_event_rate: c_frame.constraints.max_event_rate,
        },
        provenance: IntentProvenance {
            source: match c_frame.provenance.source {
                CIntentSource::UiDirect => IntentSource::UiDirect,
                CIntentSource::UiEdit => IntentSource::UiEdit,
                CIntentSource::MlText => IntentSource::MlText,
                CIntentSource::MlAudio => IntentSource::MlAudio,
                CIntentSource::Preset => IntentSource::Preset,
                CIntentSource::Automation => IntentSource::Automation,
            },
            user_override_weight: c_frame.provenance.user_override_weight,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_c_frame_roundtrip() {
        let frame = IntentFrame::with_ids(123, 456);
        let c_frame = to_c_frame(&frame);
        let restored = from_c_frame(&c_frame);
        assert_eq!(frame.meta.intent_id, restored.meta.intent_id);
        assert_eq!(frame.meta.session_id, restored.meta.session_id);
    }
}
