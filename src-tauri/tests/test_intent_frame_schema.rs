use std::fs;
use std::path::PathBuf;

use idaw_lib::generated::intent_frame::IntentFrame;

fn fixture_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.push("tests/fixtures/intent");
    p
}

fn try_parse(name: &str) -> Result<IntentFrame, serde_json::Error> {
    let path = fixture_dir().join(name);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", name, e));
    serde_json::from_str(&text)
}

#[test]
fn valid_default() {
    let f = try_parse("frame_valid_default.json").expect("Should parse");
    assert_eq!(f.meta.schema_version, 1);
    assert_eq!(f.timestamp_ms, 0);
    assert!(f.dsp_targets.stale);
    assert!(f.validate().is_ok());
}

#[test]
fn valid_full() {
    let f = try_parse("frame_valid_full.json").expect("Should parse");
    assert_eq!(f.meta.intent_id, 42);
    assert_eq!(f.timestamp_ms, 5000);
    assert!(!f.dsp_targets.stale);
    assert!((f.dsp_targets.filter_cutoff_confidence - 0.9).abs() < 0.01);
    assert!(f.validate().is_ok());
}

#[test]
fn valid_ml_audio() {
    let f = try_parse("frame_valid_ml_audio.json").expect("Should parse");
    assert_eq!(f.provenance.source, 3);
    assert!(f.dsp_targets.stale);
    assert!(f.validate().is_ok());
}

#[test]
fn invalid_version_rejected_by_validate() {
    let f = try_parse("frame_invalid_version.json").expect("serde parses it");
    assert!(f.validate().is_err());
}

#[test]
fn invalid_tempo_oob_rejected_by_validate() {
    let result = try_parse("frame_invalid_tempo_oob.json");
    if let Ok(f) = result {
        assert!(f.validate().is_err());
    }
}

#[test]
fn invalid_time_scope_rejected_by_validate() {
    let f = try_parse("frame_invalid_time_scope.json").expect("serde parses it");
    assert!(f.validate().is_err());
}

#[test]
fn invalid_extra_field() {
    let result = try_parse("frame_invalid_extra_field.json");
    assert!(result.is_err(), "deny_unknown_fields should reject");
}
