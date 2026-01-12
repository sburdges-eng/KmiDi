// Prevents additional console window on Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod bridge;

use commands::{
    generate_music,
    interrogate,
    get_emotions,
    get_humanizer_config,
    set_user_lyrics,
    get_user_lyrics,
};

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            generate_music,
            interrogate,
            get_emotions,
            get_humanizer_config,
            set_user_lyrics,
            get_user_lyrics,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
