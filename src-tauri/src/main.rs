// Prevents additional console window on Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod bridge;
mod state;
mod events;
mod generated;
mod intent_ir;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{Emitter, State};
use tokio::sync::mpsc;

use commands::{
    // C++ KellyBrain commands
    kelly_brain_initialize,
    kelly_brain_is_initialized,
    kelly_brain_from_text,
    kelly_brain_from_emotion,
    kelly_brain_generate_midi,
    kelly_brain_generate_midi_with_params,
    kelly_brain_get_emotion_state,
    kelly_brain_set_emotion_parameters,
    kelly_brain_get_available_emotions,
    kelly_brain_get_version,
    // Legacy/fallback commands
    generate_music,
    interrogate,
    get_emotions,
    get_humanizer_config,
    set_user_lyrics,
    get_user_lyrics,
};

use state::{
    get_kelly_brain_state,
    subscribe_to_state_events,
    unsubscribe_from_state_events,
    get_subscriber_count,
};

use events::{
    add_event_listener,
    remove_event_listener,
    get_event_listener_count,
    emit_test_event,
};

#[derive(Clone, Serialize)]
struct GenerationProgress {
    status: String,
    percent: u8,
}

#[derive(Clone, Serialize)]
struct GenerationResult {
    ok: bool,
    error: Option<String>,
    midi: Option<crate::bridge::kelly_ffi::GeneratedMidi>,
}

/// Per-job cancellation: each job has its own flag so cancelling one does not affect others.
struct EngineJob {
    id: u64,
    intent_json: String,
    cancel: Arc<AtomicBool>,
}

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(0);

struct AppState {
    tx: mpsc::Sender<EngineJob>,
    active_jobs: Arc<Mutex<HashMap<u64, Arc<AtomicBool>>>>,
}

#[tauri::command]
async fn start_generation(
    intent_json: String,
    state: State<'_, AppState>,
) -> Result<u64, String> {
    if intent_json.is_empty() {
        return Err("Empty intent JSON".into());
    }
    if intent_json.len() > 64_000 {
        return Err("Intent exceeds limit".into());
    }
    if serde_json::from_str::<serde_json::Value>(&intent_json).is_err() {
        return Err("Fatal: React UI generated invalid JSON syntax".into());
    }

    let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::SeqCst);
    let cancel_flag = Arc::new(AtomicBool::new(false));
    state
        .active_jobs
        .lock()
        .unwrap()
        .insert(job_id, Arc::clone(&cancel_flag));

    state
        .tx
        .send(EngineJob {
            id: job_id,
            intent_json,
            cancel: cancel_flag,
        })
        .await
        .map_err(|e| format!("Engine queue full: {e}"))?;

    Ok(job_id)
}

#[tauri::command]
async fn cancel_generation(job_id: u64, state: State<'_, AppState>) -> Result<(), String> {
    if let Some(flag) = state.active_jobs.lock().unwrap().get(&job_id) {
        flag.store(true, Ordering::SeqCst);
    }
    Ok(())
}

fn main() {
    // Load environment variables from .env files
    dotenv::dotenv().ok();
    let (job_sender, mut job_receiver) = mpsc::channel::<EngineJob>(10);
    let active_jobs: Arc<Mutex<HashMap<u64, Arc<AtomicBool>>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let active_jobs_for_worker = Arc::clone(&active_jobs);

    tauri::Builder::default()
        .manage(AppState {
            tx: job_sender,
            active_jobs,
        })
        .invoke_handler(tauri::generate_handler![
            // C++ KellyBrain commands
            kelly_brain_initialize,
            kelly_brain_is_initialized,
            kelly_brain_from_text,
            kelly_brain_from_emotion,
            kelly_brain_generate_midi,
            kelly_brain_generate_midi_with_params,
            kelly_brain_get_emotion_state,
            kelly_brain_set_emotion_parameters,
            kelly_brain_get_available_emotions,
            kelly_brain_get_version,
            // State management commands
            get_kelly_brain_state,
            subscribe_to_state_events,
            unsubscribe_from_state_events,
            get_subscriber_count,
            // Event management commands
            add_event_listener,
            remove_event_listener,
            get_event_listener_count,
            emit_test_event,
            // Async generation queue command
            start_generation,
            cancel_generation,
            // Legacy/fallback commands
            generate_music,
            interrogate,
            get_emotions,
            get_humanizer_config,
            set_user_lyrics,
            get_user_lyrics,
        ])
        .setup(move |app| {
            // Initialize state and event managers
            let app_handle = app.handle().clone();
            state::initialize_state_manager(app_handle.clone());
            events::initialize_event_manager(app_handle.clone());

            // Dedicated async worker for generation jobs (captures job_receiver, active_jobs_for_worker).
            tauri::async_runtime::spawn(async move {
                events::start_event_forwarding_if_needed();
                while let Some(job) = job_receiver.recv().await {
                    // Remove from active_jobs when this job finishes (success or error)
                    let job_id = job.id;
                    let active_jobs = Arc::clone(&active_jobs_for_worker);

                    if job.cancel.load(Ordering::SeqCst) {
                        let _ = active_jobs.lock().unwrap().remove(&job_id);
                        continue;
                    }

                    let _ = app_handle.emit(
                        "gen-progress",
                        GenerationProgress {
                            status: "Validating intent...".to_string(),
                            percent: 10,
                        },
                    );

                    let intent = match serde_json::from_str::<
                        crate::generated::intent::CompleteSongIntentRequest,
                    >(&job.intent_json)
                    {
                        Ok(i) => i,
                        Err(e) => {
                            let _ = app_handle.emit(
                                "gen-result",
                                GenerationResult {
                                    ok: false,
                                    error: Some(format!("Invalid intent JSON: {e}")),
                                    midi: None,
                                },
                            );
                            let _ = active_jobs.lock().unwrap().remove(&job_id);
                            continue;
                        }
                    };

                    if job.cancel.load(Ordering::SeqCst) {
                        let _ = active_jobs.lock().unwrap().remove(&job_id);
                        continue;
                    }

                    let prompt = format!(
                        "{} | {} | {}",
                        intent.mood_primary, intent.genre, intent.core_desire
                    );

                    let manager = crate::bridge::kelly_ffi::get_kelly_brain_manager();
                    let result = if manager.is_initialized() {
                        manager.with_brain(|brain| {
                            let intent_ir = brain.from_text(&prompt)?;
                            brain.generate_midi(&intent_ir, 8)
                        })
                    } else {
                        Err(crate::bridge::kelly_ffi::KellyError {
                            code: crate::bridge::kelly_ffi::KellyErrorCode::InitializationFailed,
                            message: "KellyBrain is not initialized".to_string(),
                        })
                    };

                    let _ = active_jobs.lock().unwrap().remove(&job_id);

                    match result {
                        Ok(midi) => {
                            let _ = app_handle.emit(
                                "gen-progress",
                                GenerationProgress {
                                    status: "MIDI generation complete".to_string(),
                                    percent: 100,
                                },
                            );
                            let _ = app_handle.emit(
                                "gen-result",
                                GenerationResult {
                                    ok: true,
                                    error: None,
                                    midi: Some(midi),
                                },
                            );
                        }
                        Err(error) => {
                            let _ = app_handle.emit(
                                "gen-result",
                                GenerationResult {
                                    ok: false,
                                    error: Some(error.message),
                                    midi: None,
                                },
                            );
                        }
                    }
                }
            });
            
            // Start background tasks (use Tauri's runtime so reactor is running)
            tauri::async_runtime::spawn(async {
                state::start_background_tasks();
                events::start_event_tasks();
            });
            
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
