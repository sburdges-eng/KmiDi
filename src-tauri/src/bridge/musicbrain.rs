use crate::commands::{GenerateRequest, InterrogateRequest};
use reqwest;
use serde_json::Value;

const MUSIC_BRAIN_API: &str = "http://127.0.0.1:8000";

pub async fn generate(request: GenerateRequest) -> Result<Value, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let res = client
        .post(format!("{}/generate", MUSIC_BRAIN_API))
        .json(&request)
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(res)
}

pub async fn interrogate(request: InterrogateRequest) -> Result<Value, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let res = client
        .post(format!("{}/interrogate", MUSIC_BRAIN_API))
        .json(&request)
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(res)
}

pub async fn get_emotions() -> Result<Value, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let res = client
        .get(format!("{}/emotions", MUSIC_BRAIN_API))
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(res)
}

pub async fn get_humanizer_config() -> Result<Value, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let res = client
        .get(format!("{}/config/humanizer", MUSIC_BRAIN_API))
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(res)
}
