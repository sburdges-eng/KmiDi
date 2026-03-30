//! Unit tests for Intent IR validator

#![cfg(test)]

use crate::types::*;
use crate::validator::*;
use crate::builder::*;

#[test]
fn test_valid_intent_frame() {
    let frame = IntentFrame::default();
    assert!(validate_intent_frame(&frame).is_ok());
}

#[test]
fn test_invalid_version() {
    let mut frame = IntentFrame::default();
    frame.meta.ir_version = 999;  // Invalid version
    assert!(validate_intent_frame(&frame).is_err());
}

#[test]
fn test_clamp_valence() {
    let mut frame = IntentFrame::default();
    frame.emotion.valence = 2.0;  // Out of range
    clamp_intent_frame(&mut frame);
    assert!(frame.emotion.valence <= 1.0);
    assert!(frame.emotion.valence >= -1.0);
}

#[test]
fn test_clamp_arousal() {
    let mut frame = IntentFrame::default();
    frame.emotion.arousal = -0.5;  // Out of range
    clamp_intent_frame(&mut frame);
    assert!(frame.emotion.arousal >= 0.0);
    assert!(frame.emotion.arousal <= 1.0);
}

#[test]
fn test_clamp_tempo_bias() {
    let mut frame = IntentFrame::default();
    frame.music.tempo_bias = 5.0;  // Out of range
    clamp_intent_frame(&mut frame);
    assert!(frame.music.tempo_bias >= -1.0);
    assert!(frame.music.tempo_bias <= 1.0);
}

#[test]
fn test_builder_validation() {
    let builder = IntentFrameBuilder::new()
        .with_emotion(0.5, 0.7, 0.6, -1, 0.8, 0.9)
        .with_musical_intent(0.3, 0.6, 0.5, 0.4, 0.5, 1, 0.7, 0.6, 0.5, 0.6);

    assert!(builder.build().is_ok());
}

#[test]
fn test_builder_invalid_values() {
    let builder = IntentFrameBuilder::new()
        .with_emotion(2.0, 0.7, 0.6, -1, 0.8, 0.9);  // Invalid valence

    // Builder should clamp and still validate
    let result = builder.build();
    // Should succeed after clamping
    assert!(result.is_ok());
}

#[test]
fn test_version_supported() {
    assert!(version_supported(1));
    assert!(!version_supported(0));
    assert!(!version_supported(2));
}

#[test]
fn test_time_scope_validation() {
    let mut frame = IntentFrame::default();
    frame.time.start_bar = 5;
    frame.time.end_bar = 3;  // Invalid: end < start
    assert!(validate_intent_frame(&frame).is_err());

    frame.time.end_bar = 10;  // Valid
    assert!(validate_intent_frame(&frame).is_ok());

    frame.time.end_bar = -1;  // Open-ended is valid
    assert!(validate_intent_frame(&frame).is_ok());
}
