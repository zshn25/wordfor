use std::path::PathBuf;
use tauri::State;
use wordfor_core::{Mode, SearchResult, WordForEngine};

use crate::state::EngineState;

#[derive(serde::Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub count: usize,
}

#[tauri::command]
pub fn init_engine(
    state: State<'_, EngineState>,
    data_dir: String,
    model_dir: Option<String>,
    mode: Option<String>,
) -> Result<String, String> {
    let mode = match mode.as_deref() {
        Some("lite") => Mode::Lite,
        Some("binary") => Mode::FullBinary,
        _ => Mode::Full, // default desktop mode
    };

    let data_path = PathBuf::from(&data_dir);
    let model_path = model_dir.map(PathBuf::from);

    let engine = WordForEngine::new(
        &data_path,
        model_path.as_deref(),
        None, // potion model dir (desktop uses full mode)
        mode,
    )
    .map_err(|e| format!("Failed to init engine: {}", e))?;

    let count = engine.data.count;
    let mut lock = state.0.lock().map_err(|e| e.to_string())?;
    *lock = Some(engine);

    Ok(format!("Engine loaded: {} entries, mode {:?}", count, mode))
}

#[tauri::command]
pub fn search(
    state: State<'_, EngineState>,
    query: String,
) -> Result<SearchResponse, String> {
    let lock = state.0.lock().map_err(|e| e.to_string())?;
    let engine = lock
        .as_ref()
        .ok_or_else(|| "Engine not initialized".to_string())?;

    let results = engine
        .search(&query)
        .map_err(|e| format!("Search failed: {}", e))?;

    let count = results.len();
    Ok(SearchResponse { results, count })
}
