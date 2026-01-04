use serde::{Deserialize, Serialize};
use tauri::command;

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

#[command]
pub async fn generate_music(request: GenerateRequest) -> Result<serde_json::Value, String> {
    crate::bridge::musicbrain::generate(request)
        .await
        .map_err(|e| e.to_string())
}

#[command]
pub async fn interrogate(request: InterrogateRequest) -> Result<serde_json::Value, String> {
    crate::bridge::musicbrain::interrogate(request)
        .await
        .map_err(|e| e.to_string())
}

#[command]
pub async fn get_emotions() -> Result<serde_json::Value, String> {
    crate::bridge::musicbrain::get_emotions()
        .await
        .map_err(|e| e.to_string())
}

#[command]
pub async fn get_humanizer_config() -> Result<serde_json::Value, String> {
    crate::bridge::musicbrain::get_humanizer_config()
        .await
        .map_err(|e| e.to_string())
}

#[command]
pub async fn set_user_lyrics(lyrics: String) -> Result<serde_json::Value, String> {
    crate::bridge::musicbrain::set_lyrics(lyrics)
        .await
        .map_err(|e| e.to_string())
}

#[command]
pub async fn get_user_lyrics() -> Result<serde_json::Value, String> {
    crate::bridge::musicbrain::get_lyrics()
        .await
        .map_err(|e| e.to_string())
}
