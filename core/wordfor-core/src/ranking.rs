use crate::config::*;
use crate::types::{ResultDef, SearchResult, WordEntry};
use std::collections::{HashMap, HashSet};

/// Suffix-stripping stemmer. Matches app.js stemWord().
pub fn stem_word(word: &str) -> Option<String> {
    let w: String = word
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .collect();
    if w.len() < 7 {
        return None;
    }

    const SUFFIXES: &[&str] = &[
        "ational", "ionally", "ically", "ation", "ition", "ness", "ment", "ible", "able",
        "ical", "ious", "eous", "ist", "ism", "ous", "ive", "ful", "ing", "ant", "ent",
        "ial", "ion", "ic", "al", "ly", "er", "ed", "ia",
    ];

    for suf in SUFFIXES {
        if w.ends_with(suf) && w.len() - suf.len() >= 5 {
            return Some(w[..w.len() - suf.len()].to_string());
        }
    }
    None
}

/// Apply quality weights: scored[i] *= entries[i].q
pub fn apply_quality_weights(scored: &mut [f32], entries: &[WordEntry]) {
    for (i, entry) in entries.iter().enumerate() {
        if i < scored.len() {
            scored[i] *= entry.q;
        }
    }
}

/// Build top-K search results with stem grouping and dedup.
/// Matches app.js topK().
pub fn top_k(
    scored: &[f32],
    entries: &[WordEntry],
    exclude: &HashSet<&str>,
) -> Vec<SearchResult> {
    let count = scored.len();

    // Sort indices by descending score
    let mut indices: Vec<usize> = (0..count).collect();
    indices.sort_unstable_by(|&a, &b| {
        scored[b]
            .partial_cmp(&scored[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let candidate_limit = CANDIDATE_LIMIT.min(indices.len());

    let mut groups: HashMap<String, SearchResult> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut word_to_group: HashMap<String, String> = HashMap::new();
    let mut stem_to_group: HashMap<String, String> = HashMap::new();

    for &idx in &indices[..candidate_limit] {
        let item_score = scored[idx];
        let entry = &entries[idx];
        let primary_lower = entry.w[0].to_lowercase();

        // Check if we have enough groups and this isn't mergeable
        if order.len() >= TOP_K && !groups.contains_key(&primary_lower) {
            let mut found = false;
            for w in &entry.w {
                if word_to_group.contains_key(&w.to_lowercase()) {
                    found = true;
                    break;
                }
            }
            if !found {
                for w in &entry.w {
                    if let Some(s) = stem_word(w) {
                        if stem_to_group.contains_key(&s) {
                            found = true;
                            break;
                        }
                    }
                }
            }
            if !found {
                break;
            }
        }

        // Find existing group via shared word
        let mut group_key: Option<String> = None;
        for w in &entry.w {
            if let Some(gk) = word_to_group.get(&w.to_lowercase()) {
                group_key = Some(gk.clone());
                break;
            }
        }
        // Try stem-based matching
        if group_key.is_none() {
            for w in &entry.w {
                if let Some(s) = stem_word(w) {
                    if let Some(gk) = stem_to_group.get(&s) {
                        group_key = Some(gk.clone());
                        break;
                    }
                }
            }
        }

        if let Some(ref gk) = group_key {
            if let Some(g) = groups.get_mut(gk) {
                if entry.h.is_none() && g.definitions.len() < 3 {
                    g.definitions.push(ResultDef {
                        definition: entry.d.clone(),
                        part_of_speech: entry.p.clone(),
                        score: item_score,
                    });
                }
                for w in &entry.w {
                    if !g.words.contains(w) {
                        g.words.push(w.clone());
                    }
                    word_to_group.insert(w.to_lowercase(), gk.clone());
                    if let Some(s) = stem_word(w) {
                        stem_to_group.insert(s, gk.clone());
                    }
                }
                for syn in &entry.s {
                    if !g.synonyms.contains(syn) {
                        g.synonyms.push(syn.clone());
                    }
                }
            }
        } else {
            if order.len() >= TOP_K {
                continue;
            }
            if exclude.contains(primary_lower.as_str()) {
                continue;
            }
            let defs = if entry.h.is_some() {
                vec![]
            } else {
                vec![ResultDef {
                    definition: entry.d.clone(),
                    part_of_speech: entry.p.clone(),
                    score: item_score,
                }]
            };
            let g = SearchResult {
                words: entry.w.clone(),
                synonyms: entry.s.clone(),
                definitions: defs,
                score: item_score,
            };
            groups.insert(primary_lower.clone(), g);
            order.push(primary_lower.clone());
            for w in &entry.w {
                word_to_group.insert(w.to_lowercase(), primary_lower.clone());
                if let Some(s) = stem_word(w) {
                    stem_to_group.insert(s, primary_lower.clone());
                }
            }
        }
    }

    order
        .iter()
        .filter_map(|k| groups.remove(k))
        .filter(|g| !g.definitions.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stem_word() {
        // "education" (9): "ation"(5) -> stem 4 < 5, skip; "ion"(3) -> stem 6 >= 5 -> "educat"
        assert_eq!(stem_word("education"), Some("educat".to_string()));
        // "nationally" (10): "ly"(2) -> stem 8 >= 5 -> "national"
        assert_eq!(stem_word("nationally"), Some("national".to_string()));
        // "running" (7): "ing"(3) -> stem 4 < 5, skip; no other match -> None
        assert_eq!(stem_word("running"), None);
        // Too short (< 7 chars)
        assert_eq!(stem_word("cat"), None);
        assert_eq!(stem_word("run"), None);
    }

    #[test]
    fn test_stem_word_min_stem() {
        // "acted" -> strip "ed" -> "act" (len 3 < 5) -> None
        assert_eq!(stem_word("abcdefed"), Some("abcdef".to_string()));
    }
}
