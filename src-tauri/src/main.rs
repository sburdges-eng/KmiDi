// Prevents additional console window on Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod bridge;
mod state;
mod events;
mod intent_ir;

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

fn main() {
    // Load environment variables from .env files
    dotenv::dotenv().ok();

    tauri::Builder::default()
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
            // Legacy/fallback commands
            generate_music,
            interrogate,
            get_emotions,
            get_humanizer_config,
            set_user_lyrics,
            get_user_lyrics,
        ])
        .setup(|app| {
            // Initialize state and event managers
            let app_handle = app.handle().clone();
            state::initialize_state_manager(app_handle.clone());
            events::initialize_event_manager(app_handle);
            
            // Start background tasks
            tokio::spawn(async {
                state::start_background_tasks();
                events::start_event_tasks();
            });
            
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
