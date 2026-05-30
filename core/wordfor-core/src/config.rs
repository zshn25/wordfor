use std::collections::HashSet;
use std::sync::LazyLock;

pub const FULL_DIMS: usize = 384;
pub const LITE_DIMS: usize = 256;
pub const FULL_BINARY_BYTES: usize = FULL_DIMS / 8; // 48
pub const FULL_INT3_BYTES: usize = FULL_DIMS * 3 / 8; // 144
pub const FULL_INT4_BYTES: usize = FULL_DIMS / 2; // 192
pub const RERANK_K: usize = 500;
pub const TOP_K: usize = 30;
pub const SHOW_K: usize = 9;
pub const CANDIDATE_LIMIT: usize = TOP_K * 4; // 120

/// Query prefix for mdbr-leaf-mt (BERT model).
pub const FULL_QUERY_PREFIX: &str =
    "Represent this sentence for searching relevant passages: ";

/// Function words / articles / pronouns excluded from results.
/// Matches the RESULT_EXCLUDE set in app.js.
pub fn result_exclude() -> &'static HashSet<&'static str> {
    static SET: LazyLock<HashSet<&str>> = LazyLock::new(|| {
        [
            "a", "an", "the", "and", "but", "or", "nor", "for", "yet", "so",
            "at", "by", "from", "in", "into", "of", "on", "to", "up", "with",
            "be", "been", "being", "is", "are", "was", "were",
            "have", "has", "had", "do", "does", "did",
            "will", "would", "shall", "should", "may", "might", "must", "can", "could",
            "he", "she", "it", "they", "we", "i", "you",
            "his", "her", "its", "their", "our", "my", "your",
            "this", "that", "these", "those",
            "b", "c", "d", "e", "f", "g", "h", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "thing", "stuff", "sort", "kind", "type", "part", "bit", "lot",
        ]
        .into_iter()
        .collect()
    });
    &SET
}
