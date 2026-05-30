#[cfg(feature = "onnx")]
use crate::config::*;
#[allow(unused_imports)]
use anyhow::{Context, Result};

/// ONNX-based query encoder for full mode (mdbr-leaf-mt).
#[cfg(feature = "onnx")]
pub struct OnnxEncoder {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "onnx")]
impl OnnxEncoder {
    /// Load from a directory containing model_quantized.onnx + tokenizer.json.
    pub fn load(model_dir: &std::path::Path) -> Result<Self> {
        let tokenizer =
            tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;

        let onnx_dir = model_dir.join("onnx");
        let model_path = if onnx_dir.join("model_quantized.onnx").exists() {
            onnx_dir.join("model_quantized.onnx")
        } else {
            onnx_dir.join("model_q4f16.onnx")
        };

        let session = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .commit_from_file(&model_path)
            .context("loading ONNX model")?;

        Ok(Self { session, tokenizer })
    }

    /// Encode a query string to a FULL_DIMS-d float32 vector.
    pub fn encode(&self, query: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{}{}", FULL_QUERY_PREFIX, query);

        let encoding = self
            .tokenizer
            .encode(prefixed, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;

        let input_ids: Vec<i64> =
            encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let token_type_ids: Vec<i64> =
            encoding.get_type_ids().iter().map(|&t| t as i64).collect();
        let seq_len = input_ids.len();

        let input_ids_arr =
            ndarray::Array2::from_shape_vec((1, seq_len), input_ids)?;
        let attention_mask_arr =
            ndarray::Array2::from_shape_vec((1, seq_len), attention_mask)?;
        let token_type_ids_arr =
            ndarray::Array2::from_shape_vec((1, seq_len), token_type_ids)?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_arr,
            "attention_mask" => attention_mask_arr,
            "token_type_ids" => token_type_ids_arr,
        ]?)?;

        let emb_tensor = outputs
            .get("sentence_embedding")
            .context("missing sentence_embedding output")?;
        let emb_view = emb_tensor.try_extract_tensor::<f32>()?;
        let emb_full: Vec<f32> = emb_view.iter().copied().collect();

        // MRL truncation to FULL_DIMS + L2 re-normalize
        let mut vec = emb_full[..FULL_DIMS].to_vec();
        let norm = vec.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-32);
        for x in &mut vec {
            *x /= norm;
        }

        Ok(vec)
    }
}

/// Potion (Model2Vec) encoder for lite mode.
/// Currently a stub — will use model2vec-rs as a path dependency.
#[cfg(feature = "potion")]
pub struct PotionEncoder {
    embedding_matrix: Vec<f32>, // vocab_size x dims, row-major
    vocab: Vec<String>,
    dims: usize,
}

#[cfg(feature = "potion")]
impl PotionEncoder {
    /// Load from a directory containing model.safetensors + tokenizer.json.
    pub fn load(model_dir: &std::path::Path) -> Result<Self> {
        use half::f16;

        let safetensors_path = model_dir.join("wasm").join("model.safetensors");
        let vocab_path = model_dir.join("vocab.txt");

        // Load vocabulary
        let vocab_text = std::fs::read_to_string(&vocab_path)
            .context("reading vocab.txt")?;
        let vocab: Vec<String> = vocab_text.lines().map(|l| l.to_string()).collect();

        // Load safetensors (float16 embedding matrix)
        let st_data = std::fs::read(&safetensors_path)
            .context("reading model.safetensors")?;
        let tensors = safetensors::SafeTensors::deserialize(&st_data)
            .map_err(|e| anyhow::anyhow!("safetensors: {e}"))?;

        let emb_tensor = tensors
            .tensor("embedding")
            .or_else(|_| tensors.tensor("static_embedding"))
            .map_err(|e| anyhow::anyhow!("no embedding tensor: {e}"))?;

        let shape = emb_tensor.shape();
        let dims = shape[1];

        // Convert float16 to float32
        let f16_bytes = emb_tensor.data();
        let embedding_matrix: Vec<f32> = f16_bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();

        Ok(Self {
            embedding_matrix,
            vocab,
            dims,
        })
    }

    /// Encode a query using bag-of-words mean pooling.
    pub fn encode(&self, query: &str) -> Vec<f32> {
        let tokens = self.tokenize(query);
        if tokens.is_empty() {
            return vec![0.0; self.dims];
        }

        let mut sum = vec![0.0f32; self.dims];
        let mut count = 0;
        for token_id in &tokens {
            let id = *token_id;
            if id < self.vocab.len() {
                let offset = id * self.dims;
                for d in 0..self.dims {
                    sum[d] += self.embedding_matrix[offset + d];
                }
                count += 1;
            }
        }

        if count > 0 {
            let inv = 1.0 / count as f32;
            for d in 0..self.dims {
                sum[d] *= inv;
            }
        }

        // L2 normalize
        let norm = sum.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-32);
        for x in &mut sum {
            *x /= norm;
        }

        sum
    }

    /// Simple WordPiece tokenization.
    fn tokenize(&self, text: &str) -> Vec<usize> {
        let text = text.to_lowercase();
        let mut token_ids = Vec::new();

        // Build vocab lookup (lazy, should be cached in production)
        let vocab_map: std::collections::HashMap<&str, usize> = self
            .vocab
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();

        for word in text.split_whitespace() {
            let word: String = word
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect();
            if word.is_empty() {
                continue;
            }

            // Try whole word first
            if let Some(&id) = vocab_map.get(word.as_str()) {
                token_ids.push(id);
                continue;
            }

            // WordPiece: greedily match longest prefix, then subwords with ##
            let chars: Vec<char> = word.chars().collect();
            let mut start = 0;
            let mut is_first = true;
            while start < chars.len() {
                let mut end = chars.len();
                let mut found = false;
                while start < end {
                    let sub: String = if is_first {
                        chars[start..end].iter().collect()
                    } else {
                        format!("##{}", chars[start..end].iter().collect::<String>())
                    };
                    if let Some(&id) = vocab_map.get(sub.as_str()) {
                        token_ids.push(id);
                        start = end;
                        is_first = false;
                        found = true;
                        break;
                    }
                    end -= 1;
                }
                if !found {
                    // Unknown token, skip character
                    start += 1;
                    is_first = false;
                }
            }
        }

        token_ids
    }
}
