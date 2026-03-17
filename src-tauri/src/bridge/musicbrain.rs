use std::env;

use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};

use crate::commands::{GenerateRequest, InterrogateRequest};

fn api_base_url() -> String {
    env::var("MUSIC_BRAIN_API_URL")
        .or_else(|_| env::var("KMIDI_API_URL"))
        .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string())
        .trim_end_matches('/')
        .to_string()
}

async fn send_json(request: reqwest::RequestBuilder) -> Result<Value, String> {
    let response = request
        .send()
        .await
        .map_err(|e| format!("MusicBrain request failed: {e}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| format!("MusicBrain response read failed: {e}"))?;

    if !status.is_success() {
        return Err(format!("MusicBrain API {}: {}", status, body));
    }

    serde_json::from_str(&body).map_err(|e| format!("MusicBrain JSON parse failed: {e}"))
}

async fn post_json<T: Serialize>(path: &str, payload: &T) -> Result<Value, String> {
    let client = Client::new();
    let url = format!("{}/{}", api_base_url(), path.trim_start_matches('/'));
    send_json(client.post(url).json(payload)).await
}

async fn get_json(path: &str) -> Result<Value, String> {
    let client = Client::new();
    let url = format!("{}/{}", api_base_url(), path.trim_start_matches('/'));
    send_json(client.get(url)).await
}

pub async fn generate(request: GenerateRequest) -> Result<Value, String> {
    post_json("/generate", &request).await
}

pub async fn interrogate(request: InterrogateRequest) -> Result<Value, String> {
    post_json("/interrogate", &request).await
}

pub async fn get_emotions() -> Result<Value, String> {
    get_json("/emotions").await
}

pub async fn get_humanizer_config() -> Result<Value, String> {
    get_json("/config/humanizer").await
}

pub async fn set_lyrics(lyrics: String) -> Result<Value, String> {
    let payload = json!({
        "lyrics": lyrics,
        "source": "user"
    });
    post_json("/lyrics", &payload).await
}

pub async fn get_lyrics() -> Result<Value, String> {
    get_json("/lyrics").await
}
