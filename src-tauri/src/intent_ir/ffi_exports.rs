//! Intent IR FFI Exports - C-compatible functions for FFI

use super::*;
use super::manager::{get_intent_ir_manager, initialize};
use super::validator::ValidationError;
use super::ffi::{CIntentFrame, from_c_frame, to_c_frame};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::OnceLock;

/// Error code type matching C header
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntentIRErrorCode {
    Success = 0,
    ErrorInvalidVersion = -1,
    ErrorOutOfRange = -2,
    ErrorInvalidDiscreteId = -3,
    ErrorInvalidTimeScope = -4,
    ErrorInvalidConstraints = -5,
    ErrorNullPointer = -6,
    ErrorMemory = -7,
    ErrorUnknown = -999,
}

impl From<ValidationError> for IntentIRErrorCode {
    fn from(err: ValidationError) -> Self {
        match err {
            ValidationError::InvalidVersion { .. } => IntentIRErrorCode::ErrorInvalidVersion,
            ValidationError::OutOfRange { .. } => IntentIRErrorCode::ErrorOutOfRange,
            ValidationError::InvalidDiscreteId { .. } => IntentIRErrorCode::ErrorInvalidDiscreteId,
            ValidationError::InvalidTimeScope { .. } => IntentIRErrorCode::ErrorInvalidTimeScope,
            ValidationError::InvalidConstraints { .. } => IntentIRErrorCode::ErrorInvalidConstraints,
        }
    }
}

/// Thread-local error message storage
thread_local! {
    static LAST_ERROR: std::cell::RefCell<String> = std::cell::RefCell::new(String::new());
}

fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = msg;
    });
}

fn get_last_error() -> String {
    LAST_ERROR.with(|e| e.borrow().clone())
}

/// Initialize Intent IR Manager
#[no_mangle]
pub extern "C" fn intent_ir_initialize(session_id: u64) -> IntentIRErrorCode {
    let session_id_opt = if session_id == 0 { None } else { Some(session_id) };
    let _ = initialize(session_id_opt);
    IntentIRErrorCode::Success
}

/// Validate and store an IntentFrame
#[no_mangle]
pub extern "C" fn intent_ir_validate_and_store(frame: *const CIntentFrame) -> IntentIRErrorCode {
    if frame.is_null() {
        set_last_error("Null pointer for IntentFrame".to_string());
        return IntentIRErrorCode::ErrorNullPointer;
    }
    
    let c_frame = unsafe { &*frame };
    let rust_frame = from_c_frame(c_frame);
    
    let manager = get_intent_ir_manager();
    match manager.validate_and_store(rust_frame) {
        Ok(()) => IntentIRErrorCode::Success,
        Err(e) => {
            set_last_error(e.to_string());
            e.into()
        }
    }
}

/// Clamp and store an IntentFrame (always succeeds)
#[no_mangle]
pub extern "C" fn intent_ir_clamp_and_store(frame: *const CIntentFrame) {
    if frame.is_null() {
        return;
    }
    
    let c_frame = unsafe { &*frame };
    let rust_frame = from_c_frame(c_frame);
    
    let manager = get_intent_ir_manager();
    manager.clamp_and_store(rust_frame);
}

/// Get current IntentFrame snapshot
#[no_mangle]
pub extern "C" fn intent_ir_get_snapshot(frame_out: *mut CIntentFrame) -> IntentIRErrorCode {
    if frame_out.is_null() {
        set_last_error("Null pointer for output IntentFrame".to_string());
        return IntentIRErrorCode::ErrorNullPointer;
    }
    
    let manager = get_intent_ir_manager();
    let snapshot = manager.get_c_snapshot();
    
    unsafe {
        *frame_out = snapshot;
    }
    
    IntentIRErrorCode::Success
}

/// Get current IntentFrame snapshot (alias for C++)
#[no_mangle]
pub extern "C" fn intent_ir_get_c_snapshot(frame_out: *mut CIntentFrame) -> IntentIRErrorCode {
    intent_ir_get_snapshot(frame_out)
}

/// Check if current frame is valid
#[no_mangle]
pub extern "C" fn intent_ir_is_valid() -> bool {
    let manager = get_intent_ir_manager();
    manager.is_valid()
}

/// Get current intent ID
#[no_mangle]
pub extern "C" fn intent_ir_get_current_intent_id() -> u64 {
    let manager = get_intent_ir_manager();
    manager.get_current_intent_id()
}

/// Get current session ID
#[no_mangle]
pub extern "C" fn intent_ir_get_current_session_id() -> u64 {
    let manager = get_intent_ir_manager();
    manager.get_current_session_id()
}

/// Generate new session ID
#[no_mangle]
pub extern "C" fn intent_ir_new_session_id() -> u64 {
    let manager = get_intent_ir_manager();
    manager.new_session_id()
}

/// Generate new intent ID
#[no_mangle]
pub extern "C" fn intent_ir_new_intent_id() -> u64 {
    let manager = get_intent_ir_manager();
    manager.new_intent_id()
}

/// Update emotion state
#[no_mangle]
pub extern "C" fn intent_ir_update_emotion(
    valence: f32,
    arousal: f32,
    dominance: f32,
    discrete_id: i16,
    intensity: f32,
    confidence: f32,
) -> IntentIRErrorCode {
    let manager = get_intent_ir_manager();
    match manager.update_emotion(
        valence,
        arousal,
        dominance,
        if discrete_id < 0 { None } else { Some(discrete_id) },
        intensity,
        confidence,
    ) {
        Ok(()) => IntentIRErrorCode::Success,
        Err(e) => {
            set_last_error(e.to_string());
            e.into()
        }
    }
}

/// Update musical intent
#[no_mangle]
pub extern "C" fn intent_ir_update_music(
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
) -> IntentIRErrorCode {
    let manager = get_intent_ir_manager();
    match manager.update_music(
        tempo_bias,
        rhythmic_density,
        groove_strength,
        harmonic_tension,
        harmonic_motion,
        mode_preference,
        melodic_activity,
        contour_variance,
        dynamic_range,
        texture_density,
    ) {
        Ok(()) => IntentIRErrorCode::Success,
        Err(e) => {
            set_last_error(e.to_string());
            e.into()
        }
    }
}

/// NUL-terminated static C strings for FFI (safe for C consumers)
static ERR_SUCCESS: &[u8] = b"Success\0";
static ERR_INVALID_VERSION: &[u8] = b"Invalid IR version\0";
static ERR_OUT_OF_RANGE: &[u8] = b"Value out of range\0";
static ERR_INVALID_DISCRETE_ID: &[u8] = b"Invalid discrete emotion ID\0";
static ERR_INVALID_TIME_SCOPE: &[u8] = b"Invalid time scope\0";
static ERR_INVALID_CONSTRAINTS: &[u8] = b"Invalid constraints\0";
static ERR_NULL_POINTER: &[u8] = b"Null pointer\0";
static ERR_MEMORY: &[u8] = b"Memory error\0";
static ERR_UNKNOWN: &[u8] = b"Unknown error\0";
static ERR_SEE_CODE: &[u8] = b"See error code for details\0";

/// Get error message for error code
#[no_mangle]
pub extern "C" fn intent_ir_get_error_message(error_code: IntentIRErrorCode) -> *const c_char {
    match error_code {
        IntentIRErrorCode::Success => ERR_SUCCESS.as_ptr() as *const c_char,
        IntentIRErrorCode::ErrorInvalidVersion => ERR_INVALID_VERSION.as_ptr() as *const c_char,
        IntentIRErrorCode::ErrorOutOfRange => ERR_OUT_OF_RANGE.as_ptr() as *const c_char,
        IntentIRErrorCode::ErrorInvalidDiscreteId => ERR_INVALID_DISCRETE_ID.as_ptr() as *const c_char,
        IntentIRErrorCode::ErrorInvalidTimeScope => ERR_INVALID_TIME_SCOPE.as_ptr() as *const c_char,
        IntentIRErrorCode::ErrorInvalidConstraints => ERR_INVALID_CONSTRAINTS.as_ptr() as *const c_char,
        IntentIRErrorCode::ErrorNullPointer => ERR_NULL_POINTER.as_ptr() as *const c_char,
        IntentIRErrorCode::ErrorMemory => ERR_MEMORY.as_ptr() as *const c_char,
        IntentIRErrorCode::ErrorUnknown => ERR_UNKNOWN.as_ptr() as *const c_char,
    }
}

/// Get last error message
#[no_mangle]
pub extern "C" fn intent_ir_get_last_error() -> *const c_char {
    ERR_SEE_CODE.as_ptr() as *const c_char
}
