//! iOS static library bridge for WordFor.
//!
//! Exposes C-ABI functions that can be called from Swift via a bridging header.
//! iOS defaults to Lite mode (potion encoder).
//! Swift handles CoreML/ONNX inference and passes the float32 vector to Rust.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::sync::Mutex;

use wordfor_core::{Mode, WordForEngine};

static ENGINE: Mutex<Option<WordForEngine>> = Mutex::new(None);

/// Initialize the engine. Call once at startup.
/// `data_dir`: path to directory containing words.json + .bin files.
/// `mode`: "full", "binary", or "lite" (default for iOS).
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn wordfor_init(data_dir: *const c_char, mode: *const c_char) -> i32 {
    let data_dir = match unsafe { CStr::from_ptr(data_dir) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let mode_str = match unsafe { CStr::from_ptr(mode) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let mode = match mode_str {
        "full" => Mode::Full,
        "binary" => Mode::FullBinary,
        _ => Mode::Lite, // iOS default
    };

    match WordForEngine::new(Path::new(data_dir), None, None, mode) {
        Ok(engine) => {
            if let Ok(mut lock) = ENGINE.lock() {
                *lock = Some(engine);
                0
            } else {
                -1
            }
        }
        Err(_) => -1,
    }
}

/// Search using a pre-computed query vector.
/// `qvec`: pointer to float32 array of query embedding.
/// `qvec_len`: number of floats in qvec.
/// Returns a JSON string (caller must free with `wordfor_free_string`).
/// Returns null on error.
#[no_mangle]
pub extern "C" fn wordfor_search_vector(qvec: *const f32, qvec_len: usize) -> *mut c_char {
    let qvec = unsafe { std::slice::from_raw_parts(qvec, qvec_len) };

    let lock = match ENGINE.lock() {
        Ok(l) => l,
        Err(_) => return std::ptr::null_mut(),
    };
    let engine = match lock.as_ref() {
        Some(e) => e,
        None => return std::ptr::null_mut(),
    };

    match engine.search_with_vector(qvec) {
        Ok(results) => match serde_json::to_string(&results) {
            Ok(json) => match CString::new(json) {
                Ok(cs) => cs.into_raw(),
                Err(_) => std::ptr::null_mut(),
            },
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Get the number of loaded entries. Returns 0 if engine not initialized.
#[no_mangle]
pub extern "C" fn wordfor_entry_count() -> u32 {
    ENGINE
        .lock()
        .ok()
        .and_then(|l| l.as_ref().map(|e| e.data.count as u32))
        .unwrap_or(0)
}

/// Free a string returned by `wordfor_search_vector`.
#[no_mangle]
pub extern "C" fn wordfor_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}
