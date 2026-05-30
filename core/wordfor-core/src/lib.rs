pub mod config;
pub mod data;
pub mod query;
pub mod ranking;
pub mod scoring;
pub mod types;

use anyhow::Result;
use std::path::Path;

pub use data::WordForData;
pub use types::{SearchResult, WordEntry};

/// Operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// ONNX encoder + binary + int3 rerank (desktop default)
    Full,
    /// ONNX encoder + binary-only (mobile/low-memory)
    FullBinary,
    /// Potion encoder + int4 scoring (lite mode)
    Lite,
}

/// The main search engine. Holds all loaded data and model state.
pub struct WordForEngine {
    pub data: WordForData,
    #[cfg(feature = "onnx")]
    onnx_encoder: Option<query::OnnxEncoder>,
    #[cfg(feature = "potion")]
    potion_encoder: Option<query::PotionEncoder>,
    pub mode: Mode,
}

impl WordForEngine {
    /// Initialize the engine with data and model directories.
#[allow(unused_variables)]
    pub fn new(
        data_dir: &Path,
        onnx_model_dir: Option<&Path>,
        potion_model_dir: Option<&Path>,
        mode: Mode,
    ) -> Result<Self> {
        let data = WordForData::load(data_dir)?;

        #[cfg(feature = "onnx")]
        let onnx_encoder = if matches!(mode, Mode::Full | Mode::FullBinary) {
            onnx_model_dir
                .map(|dir| query::OnnxEncoder::load(dir))
                .transpose()?
        } else {
            None
        };

        #[cfg(feature = "potion")]
        let potion_encoder = if mode == Mode::Lite {
            potion_model_dir
                .map(|dir| query::PotionEncoder::load(dir))
                .transpose()?
        } else {
            None
        };

        Ok(Self {
            data,
            #[cfg(feature = "onnx")]
            onnx_encoder,
            #[cfg(feature = "potion")]
            potion_encoder,
            mode,
        })
    }

    /// Execute a search query. Returns ranked, deduped results.
    pub fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(vec![]);
        }

        let qvec = self.encode_query(query)?;
        self.search_with_vector(&qvec)
    }

    /// Search using a pre-computed query vector.
    /// This is the key API for platforms that handle ONNX inference natively.
    pub fn search_with_vector(&self, qvec: &[f32]) -> Result<Vec<SearchResult>> {
        let count = self.data.count;
        let mut scored = vec![0.0f32; count];

        match self.mode {
            Mode::Full => {
                if let (Some(int3), Some(int3_ranges)) =
                    (&self.data.full_int3, &self.data.full_int3_ranges)
                {
                    scoring::score_binary_rerank(
                        qvec,
                        &self.data.full_binary,
                        int3,
                        int3_ranges,
                        self.data.itq.as_ref(),
                        config::FULL_DIMS,
                        count,
                        config::RERANK_K,
                        &mut scored,
                    );
                } else {
                    scoring::score_hamming(
                        qvec,
                        &self.data.full_binary,
                        self.data.itq.as_ref(),
                        config::FULL_DIMS,
                        count,
                        &mut scored,
                    );
                }
            }
            Mode::FullBinary => {
                scoring::score_hamming(
                    qvec,
                    &self.data.full_binary,
                    self.data.itq.as_ref(),
                    config::FULL_DIMS,
                    count,
                    &mut scored,
                );
            }
            Mode::Lite => {
                if let (Some(int4), Some(ranges)) =
                    (&self.data.potion_int4, &self.data.potion_ranges)
                {
                    scoring::score_int4(
                        qvec,
                        int4,
                        ranges,
                        config::LITE_DIMS,
                        count,
                        &mut scored,
                    );
                }
            }
        }

        ranking::apply_quality_weights(&mut scored, &self.data.entries);
        let results =
            ranking::top_k(&scored, &self.data.entries, config::result_exclude());
        Ok(results)
    }

    #[allow(unused_variables)]
    fn encode_query(&self, query: &str) -> Result<Vec<f32>> {
        match self.mode {
            Mode::Full | Mode::FullBinary => {
                #[cfg(feature = "onnx")]
                if let Some(enc) = &self.onnx_encoder {
                    return enc.encode(query);
                }
                anyhow::bail!("ONNX encoder not available")
            }
            Mode::Lite => {
                #[cfg(feature = "potion")]
                if let Some(enc) = &self.potion_encoder {
                    return Ok(enc.encode(query));
                }
                anyhow::bail!("Potion encoder not available")
            }
        }
    }
}
