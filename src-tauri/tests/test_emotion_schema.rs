use serde_json;
use std::fs;
use std::path::PathBuf;

use idaw_lib::generated::emotion::{EmotionState, EmotionTag};

fn fixture_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // src-tauri -> project root
    p.push("tests/fixtures/intent");
    p
}

fn try_parse(name: &str) -> Result<EmotionState, serde_json::Error> {
    let path = fixture_dir().join(name);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", name, e));
    serde_json::from_str(&text)
}

#[test]
fn valid_neutral() {
    let e = try_parse("emotion_valid_neutral.json").expect("Should parse");
    assert!((e.valence - 0.0).abs() < f64::EPSILON);
    assert!((e.arousal - 0.5).abs() < f64::EPSILON);
    assert!(e.tags.is_empty());
    assert!(e.validate().is_ok());
}

#[test]
fn valid_excited() {
    let e = try_parse("emotion_valid_excited.json").expect("Should parse");
    assert_eq!(e.tags.len(), 2);
    assert!(e.tags.contains(&EmotionTag::Bright));
    assert!(e.tags.contains(&EmotionTag::Drive));
    assert!(e.validate().is_ok());
}

#[test]
fn valid_sad() {
    let e = try_parse("emotion_valid_sad.json").expect("Should parse");
    assert!(e.valence < 0.0);
    assert!(e.tags.contains(&EmotionTag::Cold));
    assert!(e.validate().is_ok());
}

#[test]
fn valid_max_tags() {
    let e = try_parse("emotion_valid_max_tags.json").expect("Should parse");
    assert_eq!(e.tags.len(), 3);
    assert!(e.validate().is_ok());
}

#[test]
fn valid_no_tags() {
    let e = try_parse("emotion_valid_no_tags.json").expect("Should parse");
    assert!(e.tags.is_empty());
    assert!(e.validate().is_ok());
}

#[test]
fn invalid_tag_unknown() {
    assert!(try_parse("emotion_invalid_tag_unknown.json").is_err());
}

#[test]
fn invalid_extra_field() {
    let result = try_parse("emotion_invalid_extra_field.json");
    assert!(result.is_err(), "Should reject extra fields via deny_unknown_fields");
}

#[test]
fn invalid_valence_oob_rejected_by_validate() {
    let e = try_parse("emotion_invalid_valence_oob.json").expect("serde parses it");
    assert!(e.validate().is_err());
}

#[test]
fn invalid_too_many_tags_rejected_by_validate() {
    let e = try_parse("emotion_invalid_too_many_tags.json").expect("serde parses it");
    assert!(e.validate().is_err());
}
