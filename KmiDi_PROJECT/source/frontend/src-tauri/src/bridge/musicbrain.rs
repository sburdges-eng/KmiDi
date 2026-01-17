use crate::commands::{GenerateRequest, InterrogateRequest};
use reqwest;
use serde_json::{json, Value};
use std::time::Duration;

const MUSIC_BRAIN_API: &str = "http://127.0.0.1:8000";

fn client() -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
}

pub async fn generate(request: GenerateRequest) -> Result<Value, Box<dyn std::error::Error>> {
    let client = client()?;
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
    let client = client()?;
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
    let client = client()?;
    let res = client
        .get(format!("{}/emotions", MUSIC_BRAIN_API))
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(res)
}

pub async fn get_humanizer_config() -> Result<Value, Box<dyn std::error::Error>> {
    let client = client()?;
    let res = client
        .get(format!("{}/config/humanizer", MUSIC_BRAIN_API))
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(res)
}

pub async fn set_lyrics(lyrics: String) -> Result<Value, Box<dyn std::error::Error>> {
    let client = client()?;
    let res = client
        .post(format!("{}/lyrics", MUSIC_BRAIN_API))
        .json(&json!({ "lyrics": lyrics, "source": "user" }))
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(res)
}

pub async fn get_lyrics() -> Result<Value, Box<dyn std::error::Error>> {
    let client = client()?;
    let res = client
        .get(format!("{}/lyrics", MUSIC_BRAIN_API))
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(res)
}
