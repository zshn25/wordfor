use serde::{Deserialize, Serialize};

/// A single dictionary entry as stored in words.json.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WordEntry {
    /// Word variants (primary headword is w[0])
    pub w: Vec<String>,
    /// Definition text
    pub d: String,
    /// Part of speech (may be empty)
    #[serde(default)]
    pub p: String,
    /// Quality score (default 1.0)
    #[serde(default = "default_quality")]
    pub q: f32,
    /// Synonyms
    #[serde(default)]
    pub s: Vec<String>,
    /// Hidden flag (1 = Wiktionary sense, don't show def to user)
    #[serde(default)]
    pub h: Option<u8>,
}

fn default_quality() -> f32 {
    1.0
}

/// A search result group after dedup/stem grouping.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub words: Vec<String>,
    pub synonyms: Vec<String>,
    pub definitions: Vec<ResultDef>,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResultDef {
    pub definition: String,
    pub part_of_speech: String,
    pub score: f32,
}

/// ITQ calibration data.
pub struct ItqCalibration {
    /// Centering vector, length = dims
    pub mean: Vec<f32>,
    /// Rotation matrix, dims x dims, row-major flattened
    pub rotation: Vec<f32>,
}

/// Quantization ranges for a dimension set.
pub struct QuantRanges {
    /// Per-dimension minimum
    pub min: Vec<f32>,
    /// Per-dimension range (max - min)
    pub scale: Vec<f32>,
}
