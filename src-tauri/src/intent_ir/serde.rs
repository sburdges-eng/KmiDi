//! Intent IR Serialization - JSON and binary formats

use super::*;
use serde_json;

/// Serialize IntentFrame to JSON
pub fn to_json(frame: &IntentFrame) -> Result<String, serde_json::Error> {
    serde_json::to_string(frame)
}

/// Serialize IntentFrame to pretty JSON
pub fn to_json_pretty(frame: &IntentFrame) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(frame)
}

/// Deserialize IntentFrame from JSON
pub fn from_json(json: &str) -> Result<IntentFrame, serde_json::Error> {
    serde_json::from_str(json)
}

/// Serialize IntentFrame to JSON bytes
pub fn to_json_bytes(frame: &IntentFrame) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(frame)
}

/// Deserialize IntentFrame from JSON bytes
pub fn from_json_bytes(bytes: &[u8]) -> Result<IntentFrame, serde_json::Error> {
    serde_json::from_slice(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_json_roundtrip() {
        let frame = IntentFrame::with_ids(123, 456);
        let json = to_json(&frame).unwrap();
        let deserialized = from_json(&json).unwrap();
        assert_eq!(frame.meta.intent_id, deserialized.meta.intent_id);
        assert_eq!(frame.meta.session_id, deserialized.meta.session_id);
    }
    
    #[test]
    fn test_json_pretty() {
        let frame = IntentFrame::default();
        let json = to_json_pretty(&frame).unwrap();
        assert!(json.contains("\"ir_version\""));
    }
}
