use std::sync::Mutex;
use wordfor_core::WordForEngine;

pub struct EngineState(pub Mutex<Option<WordForEngine>>);

impl Default for EngineState {
    fn default() -> Self {
        Self(Mutex::new(None))
    }
}
