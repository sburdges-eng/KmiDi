//! Intent IR Validation Tests

use idaw::intent_ir::*;

#[test]
fn test_valid_ir_acceptance() {
    let frame = IntentFrame::with_ids(123, 456);
    let result = validator::validate_and_clamp(frame);
    assert!(result.is_ok());
}

#[test]
fn test_invalid_version_rejection() {
    let mut frame = IntentFrame::default();
    frame.meta.ir_version = 999;
    let result = validator::validate_and_clamp(frame);
    assert!(result.is_err());
    match result.unwrap_err() {
        validator::ValidationError::InvalidVersion { expected, got } => {
            assert_eq!(expected, 1);
            assert_eq!(got, 999);
        }
        _ => panic!("Wrong error type"),
    }
}

#[test]
fn test_out_of_range_rejection() {
    let mut frame = IntentFrame::default();
    frame.emotion.valence = 2.0;
    let result = validator::validate_and_clamp(frame);
    assert!(result.is_err());
}

#[test]
fn test_clamping_behavior() {
    let mut frame = IntentFrame::default();
    frame.emotion.valence = 2.0;
    frame.emotion.arousal = -1.0;
    let clamped = validator::clamp_frame(frame);
    assert_eq!(clamped.emotion.valence, 1.0);
    assert_eq!(clamped.emotion.arousal, 0.0);
}

#[test]
fn test_version_negotiation() {
    assert!(validator::supports_version(1, 1));
    assert!(!validator::supports_version(1, 2));
}

#[test]
fn test_json_roundtrip() {
    let frame = IntentFrame::with_ids(123, 456);
    let json = serde::to_json(&frame).unwrap();
    let restored = serde::from_json(&json).unwrap();
    assert_eq!(frame.meta.intent_id, restored.meta.intent_id);
}

#[test]
fn test_c_frame_roundtrip() {
    let frame = IntentFrame::with_ids(123, 456);
    let c_frame = ffi::to_c_frame(&frame);
    let restored = ffi::from_c_frame(&c_frame);
    assert_eq!(frame.meta.intent_id, restored.meta.intent_id);
}

#[test]
fn test_manager_validation() {
    let manager = manager::IntentIRManager::new();
    let frame = IntentFrame::with_ids(123, 456);
    assert!(manager.validate_and_store(frame).is_ok());
    assert!(manager.is_valid());
}

#[test]
fn test_manager_snapshot() {
    let manager = manager::IntentIRManager::new();
    let frame = IntentFrame::with_ids(123, 456);
    manager.validate_and_store(frame).unwrap();
    
    let snapshot = manager.get_snapshot();
    assert_eq!(snapshot.meta.intent_id, 123);
}
