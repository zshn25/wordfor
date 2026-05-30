use crate::config::*;
use crate::types::*;
use anyhow::{Context, Result};
use std::path::Path;

/// All loaded data needed for search.
pub struct WordForData {
    pub entries: Vec<WordEntry>,
    pub count: usize,

    // Full mode
    pub full_binary: Vec<u8>,
    pub full_int3: Option<Vec<u8>>,
    pub full_int3_ranges: Option<QuantRanges>,
    pub full_int4: Option<Vec<u8>>,
    pub full_int8: Option<Vec<u8>>,
    pub full_ranges: QuantRanges,
    pub itq: Option<ItqCalibration>,

    // Lite mode
    pub potion_int4: Option<Vec<u8>>,
    pub potion_ranges: Option<QuantRanges>,
}

impl WordForData {
    /// Load all data from a directory containing words.json and embedding files.
    pub fn load(data_dir: &Path) -> Result<Self> {
        let words_path = data_dir.join("words.json");
        let words_raw =
            std::fs::read_to_string(&words_path).context("reading words.json")?;
        let entries: Vec<WordEntry> =
            serde_json::from_str(&words_raw).context("parsing words.json")?;
        let count = entries.len();

        let full_binary = std::fs::read(data_dir.join("embeddings_binary.bin"))
            .context("reading embeddings_binary.bin")?;

        let itq = load_itq(data_dir)?;

        let full_ranges = load_ranges(data_dir, "embeddings_ranges.bin", FULL_DIMS)
            .context("reading full-mode ranges")?;

        let (full_int3, full_int3_ranges) = match load_int3(data_dir) {
            Ok((data, ranges)) => (Some(data), Some(ranges)),
            Err(_) => (None, None),
        };

        let full_int4 = std::fs::read(data_dir.join("embeddings_int4.bin")).ok();
        let full_int8 = std::fs::read(data_dir.join("embeddings_int8.bin")).ok();

        let potion_int4 =
            std::fs::read(data_dir.join("embeddings_potion_int4.bin")).ok();
        let potion_ranges =
            load_ranges(data_dir, "embeddings_potion_ranges.bin", LITE_DIMS).ok();

        Ok(Self {
            entries,
            count,
            full_binary,
            full_int3,
            full_int3_ranges,
            full_int4,
            full_int8,
            full_ranges,
            itq,
            potion_int4,
            potion_ranges,
        })
    }
}

fn load_itq(data_dir: &Path) -> Result<Option<ItqCalibration>> {
    let path = data_dir.join("embeddings_itq.bin");
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path)?;
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mean = floats[..FULL_DIMS].to_vec();
    let rotation = floats[FULL_DIMS..FULL_DIMS + FULL_DIMS * FULL_DIMS].to_vec();
    Ok(Some(ItqCalibration { mean, rotation }))
}

fn load_ranges(data_dir: &Path, filename: &str, dims: usize) -> Result<QuantRanges> {
    let bytes = std::fs::read(data_dir.join(filename))?;
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(QuantRanges {
        min: floats[..dims].to_vec(),
        scale: floats[dims..dims * 2].to_vec(),
    })
}

fn load_int3(data_dir: &Path) -> Result<(Vec<u8>, QuantRanges)> {
    let data = std::fs::read(data_dir.join("embeddings_int3.bin"))?;
    let ranges = load_ranges(data_dir, "embeddings_int3_ranges.bin", FULL_DIMS)?;
    Ok((data, ranges))
}
