//! FFI exports for C compatibility

use crate::types::*;
use crate::validator::{clamp_intent_frame, validate_intent_frame, ValidationError};
use crate::builder::IntentFrameBuilder;
use core::ffi::c_int;

/// Validation error codes (C-compatible)
#[repr(C)]
pub enum ValidationErrorCode {
    Success = 0,
    InvalidVersion = 1,
    InvalidValence = 2,
    InvalidArousal = 3,
    InvalidDominance = 4,
    InvalidIntensity = 5,
    InvalidConfidence = 6,
    InvalidTempoBias = 7,
    InvalidDensity = 8,
    InvalidModePreference = 9,
    InvalidTimeScope = 10,
    InvalidUserOverrideWeight = 11,
}

fn error_to_code(err: ValidationError) -> ValidationErrorCode {
    match err {
        ValidationError::InvalidVersion => ValidationErrorCode::InvalidVersion,
        ValidationError::InvalidValence => ValidationErrorCode::InvalidValence,
        ValidationError::InvalidArousal => ValidationErrorCode::InvalidArousal,
        ValidationError::InvalidDominance => ValidationErrorCode::InvalidDominance,
        ValidationError::InvalidIntensity => ValidationErrorCode::InvalidIntensity,
        ValidationError::InvalidConfidence => ValidationErrorCode::InvalidConfidence,
        ValidationError::InvalidTempoBias => ValidationErrorCode::InvalidTempoBias,
        ValidationError::InvalidDensity => ValidationErrorCode::InvalidDensity,
        ValidationError::InvalidModePreference => ValidationErrorCode::InvalidModePreference,
        ValidationError::InvalidTimeScope => ValidationErrorCode::InvalidTimeScope,
        ValidationError::InvalidUserOverrideWeight => ValidationErrorCode::InvalidUserOverrideWeight,
    }
}

/// Validate an IntentFrame
/// Returns 0 on success, error code on failure
#[no_mangle]
pub extern "C" fn validate_intent_frame_ffi(frame: *const IntentFrame) -> c_int {
    if frame.is_null() {
        return ValidationErrorCode::InvalidVersion as c_int;
    }

    let frame_ref = unsafe { &*frame };
    match validate_intent_frame(frame_ref) {
        Ok(()) => ValidationErrorCode::Success as c_int,
        Err(err) => error_to_code(err) as c_int,
    }
}

/// Clamp all float values in an IntentFrame to valid ranges
/// Modifies frame in-place
#[no_mangle]
pub extern "C" fn clamp_intent_frame_ffi(frame: *mut IntentFrame) {
    if frame.is_null() {
        return;
    }

    let frame_ref = unsafe { &mut *frame };
    clamp_intent_frame(frame_ref);
}

/// Check if version is supported
#[no_mangle]
pub extern "C" fn intent_frame_version_supported_ffi(version: u16) -> bool {
    version == INTENT_IR_VERSION
}

/// Opaque builder handle
#[repr(C)]
pub struct IntentFrameBuilderHandle {
    builder: Box<IntentFrameBuilder>,
}

/// Create a new IntentFrameBuilder
#[no_mangle]
pub extern "C" fn create_intent_frame_builder() -> *mut IntentFrameBuilderHandle {
    let builder = IntentFrameBuilder::new();
    let handle = Box::new(IntentFrameBuilderHandle {
        builder: Box::new(builder),
    });
    Box::into_raw(handle)
}

/// Free an IntentFrameBuilder
#[no_mangle]
pub extern "C" fn IntentFrameBuilder_free(handle: *mut IntentFrameBuilderHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

/// Set emotion on builder
#[no_mangle]
pub extern "C" fn IntentFrameBuilder_set_emotion(
    handle: *mut IntentFrameBuilderHandle,
    valence: f32,
    arousal: f32,
    dominance: f32,
    discrete_id: i16,
    intensity: f32,
    confidence: f32,
) {
    if handle.is_null() {
        return;
    }

    unsafe {
        let handle_ref = &mut *handle;
        let builder = core::ptr::replace(&mut handle_ref.builder, Box::new(IntentFrameBuilder::new()));
        let new_builder = (*builder).with_emotion(valence, arousal, dominance, discrete_id, intensity, confidence);
        handle_ref.builder = Box::new(new_builder);
    }
}

/// Set musical intent on builder
#[no_mangle]
pub extern "C" fn IntentFrameBuilder_set_musical_intent(
    handle: *mut IntentFrameBuilderHandle,
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
) {
    if handle.is_null() {
        return;
    }

    unsafe {
        let handle_ref = &mut *handle;
        let builder = core::ptr::replace(&mut handle_ref.builder, Box::new(IntentFrameBuilder::new()));
        let new_builder = (*builder).with_musical_intent(
            tempo_bias, rhythmic_density, groove_strength,
            harmonic_tension, harmonic_motion, mode_preference,
            melodic_activity, contour_variance, dynamic_range, texture_density,
        );
        handle_ref.builder = Box::new(new_builder);
    }
}

/// Set time scope on builder
#[no_mangle]
pub extern "C" fn IntentFrameBuilder_set_time_scope(
    handle: *mut IntentFrameBuilderHandle,
    start_bar: i32,
    end_bar: i32,
    fade_in_beats: f32,
    fade_out_beats: f32,
) {
    if handle.is_null() {
        return;
    }

    unsafe {
        let handle_ref = &mut *handle;
        let builder = core::ptr::replace(&mut handle_ref.builder, Box::new(IntentFrameBuilder::new()));
        let new_builder = (*builder).with_time_scope(start_bar, end_bar, fade_in_beats, fade_out_beats);
        handle_ref.builder = Box::new(new_builder);
    }
}

/// Set constraints on builder
#[no_mangle]
pub extern "C" fn IntentFrameBuilder_set_constraints(
    handle: *mut IntentFrameBuilderHandle,
    allowed_engines_mask: u32,
    forbidden_engines_mask: u32,
    max_cpu_cost: f32,
    max_event_rate: f32,
) {
    if handle.is_null() {
        return;
    }

    unsafe {
        let handle_ref = &mut *handle;
        let builder = core::ptr::replace(&mut handle_ref.builder, Box::new(IntentFrameBuilder::new()));
        let new_builder = (*builder).with_constraints(allowed_engines_mask, forbidden_engines_mask, max_cpu_cost, max_event_rate);
        handle_ref.builder = Box::new(new_builder);
    }
}

/// Set provenance on builder
#[no_mangle]
pub extern "C" fn IntentFrameBuilder_set_provenance(
    handle: *mut IntentFrameBuilderHandle,
    source: IntentSource,
    user_override_weight: f32,
) {
    if handle.is_null() {
        return;
    }

    unsafe {
        let handle_ref = &mut *handle;
        let builder = core::ptr::replace(&mut handle_ref.builder, Box::new(IntentFrameBuilder::new()));
        let new_builder = (*builder).with_provenance(source, user_override_weight);
        handle_ref.builder = Box::new(new_builder);
    }
}

/// Build the IntentFrame from builder
/// Returns 0 on success, error code on failure
/// On success, frame_out is populated
/// Note: Builder is consumed on call (success or failure)
#[no_mangle]
pub extern "C" fn IntentFrameBuilder_build(
    handle: *mut IntentFrameBuilderHandle,
    frame_out: *mut IntentFrame,
) -> c_int {
    if handle.is_null() || frame_out.is_null() {
        return ValidationErrorCode::InvalidVersion as c_int;
    }

    unsafe {
        let handle_ref = &mut *handle;
        // Take ownership of builder (replace with new empty one)
        let builder = core::ptr::replace(&mut handle_ref.builder, Box::new(IntentFrameBuilder::new()));

        // Build frame (clamps values) - consumes builder
        let frame = (*builder).build_unchecked();

        // Validate the frame
        match validate_intent_frame(&frame) {
            Ok(()) => {
                *frame_out = frame;
                ValidationErrorCode::Success as c_int
            }
            Err(err) => {
                // Builder was consumed, can't restore it
                // Caller will need to recreate if needed
                error_to_code(err) as c_int
            }
        }
    }
}
