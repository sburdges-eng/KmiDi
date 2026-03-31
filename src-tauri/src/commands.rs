use serde::{Deserialize, Serialize};
use tauri::command;
use crate::bridge::kelly_ffi::{get_kelly_brain_manager, IntentResult, GeneratedMidi, EmotionState, RTState, RTParamTarget, KellyError};

#[derive(Debug, Serialize, Deserialize)]
pub struct EmotionalIntent {
    pub core_wound: Option<String>,
    pub core_desire: Option<String>,
    pub emotional_intent: String,
    pub technical: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub intent: EmotionalIntent,
    pub output_format: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InterrogateRequest {
    pub message: String,
    pub session_id: Option<String>,
    pub context: Option<serde_json::Value>,
}

// =============================================================================
// Error Conversion
// =============================================================================

impl From<KellyError> for String {
    fn from(error: KellyError) -> Self {
        format!("KellyBrain Error: {}", error.message)
    }
}

// =============================================================================
// C++ KellyBrain Commands (Primary)
// =============================================================================

#[command]
pub async fn kelly_brain_initialize(data_path: String) -> Result<bool, String> {
    let manager = get_kelly_brain_manager();
    match manager.initialize(&data_path) {
        Ok(()) => Ok(true),
        Err(e) => Err(e.into()),
    }
}

#[command]
pub async fn kelly_brain_is_initialized() -> Result<bool, String> {
    let manager = get_kelly_brain_manager();
    Ok(manager.is_initialized())
}

#[command]
pub async fn kelly_brain_from_text(text: String) -> Result<IntentResult, String> {
    let manager = get_kelly_brain_manager();
    manager.with_brain(|brain| brain.from_text(&text))
        .map_err(|e| e.into())
}

#[command]
pub async fn kelly_brain_from_emotion(emotion_name: String, intensity: f32) -> Result<IntentResult, String> {
    let manager = get_kelly_brain_manager();
    manager.with_brain(|brain| brain.from_emotion(&emotion_name, intensity))
        .map_err(|e| e.into())
}

#[command]
pub async fn kelly_brain_generate_midi(intent: IntentResult, bars: i32) -> Result<GeneratedMidi, String> {
    let manager = get_kelly_brain_manager();
    manager.with_brain(|brain| brain.generate_midi(&intent, bars))
        .map_err(|e| e.into())
}

#[command]
pub async fn kelly_brain_generate_midi_with_params(
    intent: IntentResult, 
    bars: i32, 
    bpm: i32, 
    key_signature: String
) -> Result<GeneratedMidi, String> {
    let manager = get_kelly_brain_manager();
    manager.with_brain(|brain| brain.generate_midi_with_params(&intent, bars, bpm, &key_signature))
        .map_err(|e| e.into())
}

#[command]
pub async fn kelly_brain_get_emotion_state() -> Result<EmotionState, String> {
    let manager = get_kelly_brain_manager();
    manager.with_brain(|brain| brain.get_emotion_state())
        .map_err(|e| e.into())
}

#[command]
pub async fn kelly_brain_set_emotion_parameters(valence: f32, arousal: f32, dominance: f32) -> Result<bool, String> {
    let manager = get_kelly_brain_manager();
    match manager.with_brain(|brain| brain.set_emotion_parameters(valence, arousal, dominance)) {
        Ok(()) => Ok(true),
        Err(e) => Err(e.into()),
    }
}

#[command]
pub async fn kelly_brain_get_available_emotions() -> Result<serde_json::Value, String> {
    let manager = get_kelly_brain_manager();
    manager.with_brain(|brain| brain.get_available_emotions())
        .map_err(|e| e.into())
}

#[command]
pub async fn kelly_brain_get_version() -> Result<String, String> {
    Ok(crate::bridge::kelly_ffi::KellyBrain::get_version())
}

// =============================================================================
// Real-Time State Commands (Phase 3 — Direct C FFI, no Python)
// =============================================================================

#[allow(dead_code)]
#[command]
pub async fn kelly_brain_get_rt_state() -> Result<RTState, String> {
    let manager = get_kelly_brain_manager();
    manager.with_brain(|brain| brain.get_rt_state())
        .map_err(|e| e.into())
}

#[allow(dead_code)]
#[command]
pub async fn kelly_brain_push_rt_param(target: i32, param_index: u8, value: f32) -> Result<bool, String> {
    let manager = get_kelly_brain_manager();
    let rt_target = match target {
        0 => RTParamTarget::Bpm,
        1 => RTParamTarget::Emotion,
        2 => RTParamTarget::Intent,
        3 => RTParamTarget::TrackParam,
        4 => RTParamTarget::Transport,
        _ => return Err("Invalid RT param target".to_string()),
    };
    match manager.with_brain(|brain| brain.push_rt_param(rt_target, param_index, value)) {
        Ok(()) => Ok(true),
        Err(e) => Err(e.into()),
    }
}

// =============================================================================
// Python HTTP API Commands (Fallback/Compatibility)
// =============================================================================

#[command]
pub async fn generate_music(request: GenerateRequest) -> Result<serde_json::Value, String> {
    // Use direct FFI, not Tauri commands, to avoid circular dependency
    let manager = get_kelly_brain_manager();

    if manager.is_initialized() {
        let midi_result = manager.with_brain(|brain| {
            let intent = brain.from_text(&request.intent.emotional_intent)?;
            brain.generate_midi(&intent, 8)
        }).map_err(String::from)?;

        return Ok(serde_json::to_value(midi_result).unwrap());
    }
    
    // Fallback to Python HTTP API
    crate::bridge::musicbrain::generate(request)
        .await
        .map_err(|e| e.to_string())
}

#[command]
pub async fn interrogate(request: InterrogateRequest) -> Result<serde_json::Value, String> {
    // Use direct FFI, not Tauri commands, to avoid circular dependency
    let manager = get_kelly_brain_manager();

    if manager.is_initialized() {
        let intent_result = manager.with_brain(|brain| {
            brain.from_text(&request.message)
        }).map_err(String::from)?;

        return Ok(serde_json::to_value(intent_result).unwrap());
    }
    
    // Fallback to Python HTTP API
    crate::bridge::musicbrain::interrogate(request)
        .await
        .map_err(|e| e.to_string())
}

#[command]
pub async fn get_emotions() -> Result<serde_json::Value, String> {
    // Use direct FFI, not Tauri commands, to avoid circular dependency
    let manager = get_kelly_brain_manager();

    if manager.is_initialized() {
        return manager.with_brain(|brain| brain.get_available_emotions())
            .map_err(String::from);
    }
    
    // Fallback to Python HTTP API
    crate::bridge::musicbrain::get_emotions()
        .await
        .map_err(|e| e.to_string())
}

#[command]
pub async fn get_humanizer_config() -> Result<serde_json::Value, String> {
    // For now, only Python API supports this
    crate::bridge::musicbrain::get_humanizer_config()
        .await
        .map_err(|e| e.to_string())
}

#[command]
pub async fn set_user_lyrics(lyrics: String) -> Result<serde_json::Value, String> {
    // For now, only Python API supports this
    crate::bridge::musicbrain::set_lyrics(lyrics)
        .await
        .map_err(|e| e.to_string())
}

#[command]
pub async fn get_user_lyrics() -> Result<serde_json::Value, String> {
    // For now, only Python API supports this
    crate::bridge::musicbrain::get_lyrics()
        .await
        .map_err(|e| e.to_string())
}
