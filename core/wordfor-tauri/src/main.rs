#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod state;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(state::EngineState::default())
        .invoke_handler(tauri::generate_handler![
            commands::init_engine,
            commands::search,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
