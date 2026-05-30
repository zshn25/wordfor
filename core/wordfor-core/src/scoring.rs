use crate::types::{ItqCalibration, QuantRanges};

/// Int8 dot product scoring. Matches app.js scoreInt8().
pub fn score_int8(
    qvec: &[f32],
    emb_data: &[u8],
    ranges: &QuantRanges,
    dims: usize,
    count: usize,
    out: &mut [f32],
) {
    let mut q_scaled = vec![0.0f32; dims];
    let mut q_offset: f32 = 0.0;
    for d in 0..dims {
        q_scaled[d] = qvec[d] * ranges.scale[d] / 255.0;
        q_offset += qvec[d] * ranges.min[d];
    }
    for i in 0..count {
        let mut dot = q_offset;
        let base = i * dims;
        for d in 0..dims {
            dot += q_scaled[d] * emb_data[base + d] as f32;
        }
        out[i] = dot;
    }
}

/// Int4 dot product scoring (2 nibbles per byte). Matches app.js scoreInt4().
pub fn score_int4(
    qvec: &[f32],
    int4_data: &[u8],
    ranges: &QuantRanges,
    dims: usize,
    count: usize,
    out: &mut [f32],
) {
    let mut q_scaled = vec![0.0f32; dims];
    let mut q_offset: f32 = 0.0;
    for d in 0..dims {
        q_scaled[d] = qvec[d] * ranges.scale[d] / 15.0;
        q_offset += qvec[d] * ranges.min[d];
    }
    let half_dims = dims >> 1;
    for i in 0..count {
        let mut dot = q_offset;
        let base = i * half_dims;
        for d in (0..dims).step_by(2) {
            let packed = int4_data[base + (d >> 1)];
            dot += q_scaled[d] * (packed >> 4) as f32;
            dot += q_scaled[d + 1] * (packed & 0x0F) as f32;
        }
        out[i] = dot;
    }
}

/// Int3 dot product scoring (8 dims per 3 bytes, big-endian).
/// Matches app.js scoreInt3(). Critical reranking kernel.
pub fn score_int3(
    qvec: &[f32],
    int3_data: &[u8],
    ranges: &QuantRanges,
    dims: usize,
    count: usize,
    out: &mut [f32],
) {
    let mut q_scaled = vec![0.0f32; dims];
    let mut q_offset: f32 = 0.0;
    for d in 0..dims {
        q_scaled[d] = qvec[d] * ranges.scale[d] / 7.0;
        q_offset += qvec[d] * ranges.min[d];
    }
    let bytes_per_entry = (dims * 3) >> 3;
    let n_groups = dims >> 3;
    for i in 0..count {
        let mut dot = q_offset;
        let base = i * bytes_per_entry;
        for g in 0..n_groups {
            let b = base + g * 3;
            let w = (int3_data[b] as u32) << 16
                | (int3_data[b + 1] as u32) << 8
                | int3_data[b + 2] as u32;
            let d8 = g << 3;
            dot += q_scaled[d8] * ((w >> 21) & 7) as f32;
            dot += q_scaled[d8 + 1] * ((w >> 18) & 7) as f32;
            dot += q_scaled[d8 + 2] * ((w >> 15) & 7) as f32;
            dot += q_scaled[d8 + 3] * ((w >> 12) & 7) as f32;
            dot += q_scaled[d8 + 4] * ((w >> 9) & 7) as f32;
            dot += q_scaled[d8 + 5] * ((w >> 6) & 7) as f32;
            dot += q_scaled[d8 + 6] * ((w >> 3) & 7) as f32;
            dot += q_scaled[d8 + 7] * (w & 7) as f32;
        }
        out[i] = dot;
    }
}

/// Binary Hamming scoring with ITQ rotation.
/// Matches app.js scoreHamming().
pub fn score_hamming(
    qvec: &[f32],
    bin_data: &[u8],
    itq: Option<&ItqCalibration>,
    dims: usize,
    count: usize,
    out: &mut [f32],
) {
    let bytes_per_entry = dims / 8;

    // Apply ITQ rotation if available
    let rotated: Vec<f32> = if let Some(cal) = itq {
        (0..dims)
            .map(|d| {
                let mut sum = 0.0f32;
                for k in 0..dims {
                    sum += (qvec[k] - cal.mean[k]) * cal.rotation[k * dims + d];
                }
                sum
            })
            .collect()
    } else {
        qvec.to_vec()
    };

    // Pack query sign bits (MSB first, matching JS: 128 >> bit)
    let mut q_bin = vec![0u8; bytes_per_entry];
    for b in 0..bytes_per_entry {
        let mut byte = 0u8;
        for bit in 0..8u8 {
            if rotated[b * 8 + bit as usize] > 0.0 {
                byte |= 128 >> bit;
            }
        }
        q_bin[b] = byte;
    }

    // Hamming scoring: similarity = (dims - 2*dist) / dims
    let dims_f = dims as f32;
    for i in 0..count {
        let base = i * bytes_per_entry;
        let mut dist: u32 = 0;
        for b in 0..bytes_per_entry {
            dist += (q_bin[b] ^ bin_data[base + b]).count_ones();
        }
        out[i] = (dims_f - 2.0 * dist as f32) / dims_f;
    }
}

/// Quickselect: rearranges indices so top-k by descending score are first.
fn nth_element_by(indices: &mut [usize], scores: &[f32], k: usize) {
    if indices.len() <= k {
        return;
    }
    let mut lo = 0;
    let mut hi = indices.len() - 1;
    let mut remaining = k;
    while lo < hi {
        let pivot_score = scores[indices[lo + ((hi - lo) >> 1)]];
        let mut i = lo;
        let mut j = hi;
        while i <= j {
            while scores[indices[i]] > pivot_score {
                i += 1;
            }
            while scores[indices[j]] < pivot_score {
                if j == 0 {
                    break;
                }
                j -= 1;
            }
            if i <= j {
                indices.swap(i, j);
                i += 1;
                if j == 0 {
                    break;
                }
                j -= 1;
            }
        }
        if j >= lo && j - lo + 1 >= remaining {
            hi = j;
        } else if i >= lo && i - lo <= remaining {
            remaining -= i - lo;
            lo = i;
        } else {
            break;
        }
    }
}

/// Two-stage scoring: binary Hamming first-pass + int3 reranking.
/// Matches app.js scoreBinaryRerank().
pub fn score_binary_rerank(
    qvec: &[f32],
    binary_data: &[u8],
    rerank_data: &[u8],
    ranges: &QuantRanges,
    itq: Option<&ItqCalibration>,
    dims: usize,
    count: usize,
    rerank_k: usize,
    out: &mut [f32],
) {
    // Stage 1: Hamming over all entries
    let mut hamming_scores = vec![0.0f32; count];
    score_hamming(qvec, binary_data, itq, dims, count, &mut hamming_scores);

    // Stage 2: Find top rerank_k by Hamming
    let mut indices: Vec<usize> = (0..count).collect();
    let k = rerank_k.min(count);
    nth_element_by(&mut indices, &hamming_scores, k);

    // Fill all with -inf, then overwrite reranked candidates
    for s in out.iter_mut().take(count) {
        *s = f32::NEG_INFINITY;
    }

    // Pre-compute query scaling for int3
    let mut q_scaled = vec![0.0f32; dims];
    let mut q_offset: f32 = 0.0;
    for d in 0..dims {
        q_scaled[d] = qvec[d] * ranges.scale[d] / 7.0;
        q_offset += qvec[d] * ranges.min[d];
    }
    let bytes_per_entry = (dims * 3) >> 3;
    let n_groups = dims >> 3;

    for j in 0..k {
        let idx = indices[j];
        let mut dot = q_offset;
        let base = idx * bytes_per_entry;
        for g in 0..n_groups {
            let b = base + g * 3;
            let w = (rerank_data[b] as u32) << 16
                | (rerank_data[b + 1] as u32) << 8
                | rerank_data[b + 2] as u32;
            let d8 = g << 3;
            dot += q_scaled[d8] * ((w >> 21) & 7) as f32;
            dot += q_scaled[d8 + 1] * ((w >> 18) & 7) as f32;
            dot += q_scaled[d8 + 2] * ((w >> 15) & 7) as f32;
            dot += q_scaled[d8 + 3] * ((w >> 12) & 7) as f32;
            dot += q_scaled[d8 + 4] * ((w >> 9) & 7) as f32;
            dot += q_scaled[d8 + 5] * ((w >> 6) & 7) as f32;
            dot += q_scaled[d8 + 6] * ((w >> 3) & 7) as f32;
            dot += q_scaled[d8 + 7] * (w & 7) as f32;
        }
        out[idx] = dot;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_int8_basic() {
        let dims = 4;
        let qvec = vec![0.5, -0.3, 0.8, 0.1];
        let ranges = QuantRanges {
            min: vec![-1.0, -1.0, -1.0, -1.0],
            scale: vec![2.0, 2.0, 2.0, 2.0],
        };
        // One entry: [128, 128, 128, 128] = all at midpoint (0.0)
        let emb_data = vec![128u8, 128, 128, 128];
        let mut out = vec![0.0f32; 1];
        score_int8(&qvec, &emb_data, &ranges, dims, 1, &mut out);
        // Each dim: qvec[d] * (128/255 * 2.0 + (-1.0)) ≈ qvec[d] * 0.00392
        // Sum should be close to 0
        assert!((out[0]).abs() < 0.1);
    }

    #[test]
    fn test_score_int8_known_values() {
        // 2 dims, 2 entries
        let dims = 2;
        let qvec = vec![1.0, 0.0]; // only first dim matters
        let ranges = QuantRanges {
            min: vec![0.0, 0.0],   // min=0
            scale: vec![1.0, 1.0], // scale=1
        };
        // Entry 0: [255, 0] -> first dim = 255/255*1.0+0.0 = 1.0
        // Entry 1: [0, 255] -> first dim = 0
        let emb_data = vec![255, 0, 0, 255];
        let mut out = vec![0.0f32; 2];
        score_int8(&qvec, &emb_data, &ranges, dims, 2, &mut out);
        // out[0] = 1.0*1.0 + 0.0*0.0 = 1.0 (approx, via q_offset + q_scaled * byte)
        // out[1] = 1.0*0.0 + 0.0*1.0 = 0.0
        assert!(out[0] > out[1], "Entry 0 should score higher");
    }

    #[test]
    fn test_score_int4_basic() {
        // 4 dims, 1 entry. Packed: 2 nibbles per byte -> 2 bytes per entry
        let dims = 4;
        let qvec = vec![1.0, 0.0, 1.0, 0.0];
        let ranges = QuantRanges {
            min: vec![0.0; 4],
            scale: vec![1.0; 4],
        };
        // 4 dims -> 2 bytes. Nibbles: [15,0, 15,0] = bytes [0xF0, 0xF0]
        let int4_data = vec![0xF0, 0xF0];
        let mut out = vec![0.0f32; 1];
        score_int4(&qvec, &int4_data, &ranges, dims, 1, &mut out);
        // dot = q_offset(0) + q_scaled[0]*15 + q_scaled[1]*0 + q_scaled[2]*15 + q_scaled[3]*0
        // q_scaled[d] = qvec[d] * scale / 15 = 1/15 for d=0,2
        // dot = (1/15)*15 + 0 + (1/15)*15 + 0 = 2.0
        assert!((out[0] - 2.0).abs() < 1e-5, "Expected ~2.0, got {}", out[0]);
    }

    #[test]
    fn test_score_int4_two_entries() {
        let dims = 4;
        let qvec = vec![1.0, 1.0, 1.0, 1.0];
        let ranges = QuantRanges {
            min: vec![0.0; 4],
            scale: vec![1.0; 4],
        };
        // Entry 0: all 15s -> [0xFF, 0xFF] => max dot
        // Entry 1: all 0s  -> [0x00, 0x00] => zero dot
        let int4_data = vec![0xFF, 0xFF, 0x00, 0x00];
        let mut out = vec![0.0f32; 2];
        score_int4(&qvec, &int4_data, &ranges, dims, 2, &mut out);
        assert!(out[0] > out[1], "All-max entry should beat all-zero: {} vs {}", out[0], out[1]);
        // out[0] = 4 * (1/15 * 15) = 4.0
        assert!((out[0] - 4.0).abs() < 1e-5);
        assert!(out[1].abs() < 1e-5);
    }

    #[test]
    fn test_score_int3_basic() {
        // 8 dims (1 group), 1 entry. 3 bytes per group.
        let dims = 8;
        let qvec = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // only first dim
        let ranges = QuantRanges {
            min: vec![0.0; 8],
            scale: vec![1.0; 8],
        };
        // Pack: val[0]=7, rest=0.
        // 3-bit per dim, 8 dims, big-endian 24-bit word:
        // bits: 111_000_000_000_000_000_000_000 = 0b111000... = 0xE00000
        // bytes: [0xE0, 0x00, 0x00]
        let int3_data = vec![0xE0, 0x00, 0x00];
        let mut out = vec![0.0f32; 1];
        score_int3(&qvec, &int3_data, &ranges, dims, 1, &mut out);
        // dot = q_offset(0) + q_scaled[0] * 7
        // q_scaled[0] = 1.0 * 1.0 / 7.0 = 1/7
        // dot = (1/7) * 7 = 1.0
        assert!((out[0] - 1.0).abs() < 1e-5, "Expected ~1.0, got {}", out[0]);
    }

    #[test]
    fn test_score_int3_all_dims() {
        // 8 dims, all values = 7 (max)
        let dims = 8;
        let qvec = vec![1.0; 8];
        let ranges = QuantRanges {
            min: vec![0.0; 8],
            scale: vec![1.0; 8],
        };
        // All 7s: 111_111_111_111_111_111_111_111 = 0xFFFFFF
        let int3_data = vec![0xFF, 0xFF, 0xFF];
        let mut out = vec![0.0f32; 1];
        score_int3(&qvec, &int3_data, &ranges, dims, 1, &mut out);
        // Each dim: (1/7)*7 = 1.0, 8 dims -> 8.0
        assert!((out[0] - 8.0).abs() < 1e-5, "Expected ~8.0, got {}", out[0]);
    }

    #[test]
    fn test_score_int3_packing_order() {
        // Verify bit packing: dim0 is highest 3 bits
        let dims = 8;
        // qvec = [1,0,0,0,0,0,0,1] -> only dim0 and dim7 matter
        let qvec = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let ranges = QuantRanges {
            min: vec![0.0; 8],
            scale: vec![1.0; 8],
        };
        // dim0=5 (101), dim7=3 (011), rest=0
        // bits: 101_000_000_000_000_000_000_011
        // = 0b10100000_00000000_00000011
        // = [0xA0, 0x00, 0x03]
        let int3_data = vec![0xA0, 0x00, 0x03];
        let mut out = vec![0.0f32; 1];
        score_int3(&qvec, &int3_data, &ranges, dims, 1, &mut out);
        // dot = (1/7)*5 + (1/7)*3 = 8/7 ≈ 1.1429
        let expected = 8.0 / 7.0;
        assert!((out[0] - expected).abs() < 1e-5, "Expected {}, got {}", expected, out[0]);
    }

    #[test]
    fn test_score_hamming_no_itq() {
        let dims = 8;
        // query: all positive -> bin = 0xFF
        let qvec = vec![1.0; 8];
        // one entry: all 1s = 0xFF -> distance = 0
        let bin_data = vec![0xFF];
        let mut out = vec![0.0f32; 1];
        score_hamming(&qvec, &bin_data, None, dims, 1, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6); // perfect match
    }

    #[test]
    fn test_score_hamming_opposite() {
        let dims = 8;
        // query: all positive -> bin = 0xFF
        let qvec = vec![1.0; 8];
        // entry: all 0s = 0x00 -> distance = 8
        let bin_data = vec![0x00];
        let mut out = vec![0.0f32; 1];
        score_hamming(&qvec, &bin_data, None, dims, 1, &mut out);
        // similarity = (8 - 2*8)/8 = -1.0
        assert!((out[0] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_score_hamming_half_match() {
        let dims = 8;
        let qvec = vec![1.0; 8]; // bin = 0xFF
        // entry: 0xF0 = upper 4 match, lower 4 don't -> dist=4
        let bin_data = vec![0xF0];
        let mut out = vec![0.0f32; 1];
        score_hamming(&qvec, &bin_data, None, dims, 1, &mut out);
        // similarity = (8 - 2*4)/8 = 0.0
        assert!((out[0]).abs() < 1e-6);
    }

    #[test]
    fn test_score_binary_rerank_basic() {
        // 8 dims, 3 entries, rerank_k=2
        let dims = 8;
        let count = 3;
        let qvec = vec![1.0; 8]; // bin = 0xFF

        // Binary data: entry 0=0xFF (best), entry 1=0xF0, entry 2=0x00 (worst)
        let binary_data = vec![0xFF, 0xF0, 0x00];

        // Int3 rerank data: 3 bytes per entry (8 dims)
        // Entry 0: all 7s = 0xFFFFFF
        // Entry 1: all 4s: 100_100_100_100_100_100_100_100 = 0x924924
        // Entry 2: all 0s
        let int3_data = vec![
            0xFF, 0xFF, 0xFF,   // entry 0: all max
            0x92, 0x49, 0x24,   // entry 1: all 4s
            0x00, 0x00, 0x00,   // entry 2: all 0
        ];
        let ranges = QuantRanges {
            min: vec![0.0; 8],
            scale: vec![1.0; 8],
        };

        let mut out = vec![0.0f32; 3];
        score_binary_rerank(&qvec, &binary_data, &int3_data, &ranges, None, dims, count, 2, &mut out);

        // Top-2 by Hamming are entries 0 and 1, reranked with int3
        // Entry 0 (all 7s) should score highest
        // Entry 2 (not in top-2) should remain -inf
        assert!(out[0] > out[1], "Entry 0 should beat entry 1: {} vs {}", out[0], out[1]);
        assert!(out[2] == f32::NEG_INFINITY, "Entry 2 should be -inf (not reranked)");
    }

    #[test]
    fn test_nth_element_by_basic() {
        let scores = vec![1.0f32, 5.0, 3.0, 4.0, 2.0];
        let mut indices: Vec<usize> = (0..5).collect();
        nth_element_by(&mut indices, &scores, 2);
        // Top-2 by descending score: indices for 5.0 and 4.0 (indices 1,3)
        let top2: Vec<usize> = indices[..2].to_vec();
        assert!(top2.contains(&1) && top2.contains(&3),
            "Top-2 should be indices 1 and 3, got {:?}", top2);
    }
}
