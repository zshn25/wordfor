# WordFor Project Memory

## Project Overview
- **WordFor** (wordfor.xyz): Free reverse dictionary, runs entirely in browser
- GitHub Pages hosting (CNAME: wordfor.xyz), purely static, no backend
- GoatCounter for analytics (privacy-friendly)
- License: CC-BY-NC-ND-4.0 (considering dual: CC-BY-SA for data, CC-BY-NC-ND for code)

## Architecture
- **Full mode**: mxbai-embed-large-v1 (1024d -> 384d MRL) teacher + mdbr-leaf-mt (22M) student
- **Scoring**: Binary (ITQ-calibrated) Hamming first-pass + int4 rerank of top-500 (default desktop); pure binary on mobile
- **Mobile**: iOS auto-detects to lite mode (ONNX WASM OOM for both q8 and q4f16). Android tries full mode with BINARY_ONLY.
- **Int4 reranking**: Half the file size of int8, BETTER MRR (0.6385 vs 0.6305, +1 H@1). Default rerank format.
- **Quantization formats available**: int8 (48B/d), int4 (192B/entry, packed nibbles), int3 (144B/entry, 8 dims/3-byte word), int2 (96B/entry, 4 dims/byte), binary (48B/entry, ITQ)
- **Word dedup**: Suffix-stripping stemmer (`stemWord()`) groups morphological variants in `topK()`, with definition-overlap guard (>=0.08 Jaccard) to prevent false merges
- **RESULT_EXCLUDE**: Hard set in app.js (~50 function words/articles/pronouns) excluded from topK() results
- **Hub word penalty**: Words appearing in >5%/15% of definitions get 0.90x/0.80x quality multiplier applied AFTER `q^0.1` dampening (guarded: q_idf > 0.95 bypasses penalty)
- **Lite mode**: distilled-mxbai (256d, bag-of-words) fine-tuned via knowledge distillation from mxbai-embed-large-v1, int4 scoring
- Pure cosine ranking + quality weights (keyword blending REMOVED per user request)
- **Wiktionary embedding enrichment**: Wiktionary senses encoded at build time, averaged with main entry embedding (up to 3 senses), then re-normalized. No text redistributed.
- **ITQ (Iterative Quantization)**: 50-iteration rotation matrix trained on 50K subsample, 0.2% bit-flip rate

## Key Files
- `wordfor/app.js` - Main frontend (search, rendering, quality weights, ITQ+rerank)
- `wordfor/build/export_embeddings.py` - Shared quantization (int8/int4/int3/int2) + potion model export utilities
- `wordfor/build/build_dictionary.py` - Dictionary build pipeline (5 PD sources + Moby Common + LLM augmented + Wiktionary enrichment + quality + full-mode embeddings only)
- `wordfor/build/finetune_potion.py` - Fine-tuning + evaluation (67-query test set, 6+ quant methods) + potion embedding export
- `wordfor/build/compare_eval.py` - Cross-method evaluation (auto-updates bench.html eval table)
- `wordfor/build/llm_augmented.json` - 61 curated LLM-written definitions (new words + improved terse entries), SOURCE_WEIGHT=0.95
- `wordfor/build/build.sh` - Unified build script (dictionary -> finetune -> export -> eval)
- `wordfor/bench.html` - Browser-side scoring + E2E benchmark page (E2E reduced to Lite only; main table compares 1-bit through 4-bit data quantization)
- `wordfor/about.html` - About page (how it works, tech details)
- `wordfor/data/words.json` - ~350K entries with `w[]` (variants), `s[]` (synonyms), `q` (quality)
- `wordfor/data/embeddings_itq.bin` - ITQ calibration (mean + rotation matrix, ~578 KB)

## Dictionary & Build-Time Sources
- OEWN 2025+ (CC BY 4.0) - primary, definitions redistributed
- Webster's 1913 (public domain) - definitions redistributed
- GCIDE Webster 1913 portion (public domain) - supplementary Webster entries, `parse_gcide.py` extracts PD-only
- Century Dictionary 1889-1911 (public domain) - 161K entries, `parse_century.py` parses from hupong Markdown. Fixed: strips contributor-attribution text (» / «) and ALL-CAPS section headings. Safety net also in `load_century_entries()`.
- Funk & Wagnalls 1908 - **REMOVED** (OCR too corrupt; funkandwagnalls.com is a domain squatter, can't crawl)
- Moby Thesaurus II (public domain, synonym enrichment -> stored in `s[]` field); 30K-root Thesaurus II
- Moby Part-of-Speech II (public domain, 233K words) - build-time coverage signal only
- Moby Words II COMMON.TXT (74,550 words in 2+ major dicts) - build-time quality signal, replaced F&W cross-source signal
- LLM-augmented definitions (CC0) - 61 curated definitions in `llm_augmented.json`: new words absent from PD sources (petrichor, hiraeth, saudade, pareidolia, misophonia, hygge, ikigai, etc.) + improved senses for terse entries. Injected AFTER merge_and_dedup, BEFORE compute_quality_scores. SOURCE_WEIGHT=0.95. Will appear in words.json on next full rebuild.
- Wiktionary via kaikki.org (CC-BY-SA 3.0): build-time quality signals + embedding enrichment ONLY, NOT redistributed.
- ConceptNet 5.7 (CC-BY-SA 4.0, build-time quality signals only, NOT redistributed)
- American Heritage Dictionary: PROPRIETARY, cannot use
- GCIDE GPL additions (WordNet/PJC): Fine-tuning only, not redistribution

## Evaluation Results — Deployed Models (67-query test set, Apr 2026 data)
| Method | Mode | MRR | Hit@1 | Hit@6 |
|--------|------|-----|-------|-------|
| Full binary+int4 rerank | Full (desktop) | 0.6234 | ~37 | ~53 |
| Full pure int4 | Full | 0.6242 | — | — |
| Full pure binary ITQ | Full (mobile) | 0.5765 | — | — |
| Potion fine-tuned int4 | Lite | 0.4815 | 26 | 40 |
| Potion base int4 | Lite (baseline) | 0.4290 | 22 | 38 |

Note: MRR dropped from 0.5353->0.4815 (lite) after dictionary expansion (F&W + per-sense entries adding noise).

## Fine-tuning (potion / lite mode)
- **Base model**: Knowledge-distilled from mxbai-embed-large-v1 -> 256d static model (Model2Vec distill)
  - Distilled base saved at `wordfor/build/potion-distilled-mxbai/`
  - IMPORTANT: Must convert float16 -> float32 before training (NaN gradients otherwise)
- **Fine-tuned model**: `wordfor/build/potion-potion-distilled-mxbai/final/`
- Sources: OEWN + Webster's + GCIDE + Century + Wiktionary overlap (5 sources)
- No SemHash dedup (near-duplicates kept as augmentation), only eval decontamination
- Distilled+FT improved over potion-base-8M FT: MRR 0.5128 -> 0.5353 (+4.4%) on pre-expansion dictionary
- Run with: `.venv/Scripts/python.exe finetune_potion.py --base-model ./potion-distilled-mxbai`
- CUDA venv at `wordfor/build/.venv` (has torch CUDA + bf16 support)
- After training: `--export --model potion-potion-distilled-mxbai/final` to regenerate, then `compare_eval.py`
- Lite mode inherently weaker on focusing queries (bag-of-words weights all words equally)

**CRITICAL: StaticModel vs SentenceTransformer normalization**
- `StaticModel.encode(normalize_embeddings=True)` does NOT normalize output (norms = 15-26, NOT 1.0)
- `SentenceTransformer.encode(normalize_embeddings=True)` DOES normalize correctly (norm = 1.0)
- `export_for_browser()` in finetune_potion.py: MUST use SentenceTransformer for DB encoding
- `compare_eval.py` `_potion_score()`: uses StaticModel for queries (ok — consistent per-query norm preserves ranking)
- Using StaticModel for DB encoding makes DB norms variable per entry, corrupting relative ranking (near-zero MRR)

Note: Full-mode and potion evals use different dictionaries/embeddings so MRR not directly comparable.
Higher-dim binary experiment (128d-384d): diminishing returns above ~192d, reranking matters more than extra bits.

## 1024d Binary Experiment (3K subsample, 62 valid queries)
- **Pure binary: 384d is optimal** (MRR=0.82, H@1=51), beats 768d (0.81) and 1024d (0.80)
- MRL concentrates semantic info in first dims, so more binary bits past 384 add noise
- **192d degrades significantly** (H@1=42) — not worth the 4MB savings
- 1024d binary+rerank is best (MRR=0.88) but at 193MB total, impractical for browser
- Conclusion: current 384d setup is near-optimal for size/quality tradeoff

## User Preferences
- "Don't use outlier filtering"
- "Whenever you change something, run yourself to make sure it works"
- "Data should be evolving and refining" (data-centric ML)
- "Let's stick to pure cosine ranking without keyword blending"
- Windows environment, cp1252 terminal (avoid Unicode arrows/special chars in print)
- "For data versioning, don't keep multiple copies" (prefers DVC/Git LFS, not archive copies)
- "If binary performance is almost similar to int8, use that directly" (prefer binary+rerank as default)

## Known Issues / Status
- `potion-wordnet/final/` has old potion-base-8M FT weights; current deployed model is `potion-potion-distilled-mxbai/final/`
- Previously-missing words (petrichor, glossophobia, athazagoraphobia, bibliophilia, etc.) are now in `llm_augmented.json` — will be in words.json after next rebuild
- words.json patched directly: 3 Century garbage entries fixed/removed (standard→clean def, foreign→clean def, critical garbage sense removed). Embeddings still use old vectors until rebuild.
- Compound word dominance: "art teacher" outranks "teacher"
- Antonym confusion: bag-of-words can't distinguish "fear" from "fearless"
- ConceptNet NOT viable as supplement data source (no real definitions, mostly multi-word phrases)
- words.json is per-sense format: 350K entries with 74K duplicated primary headwords (e.g., "cut" appears 53x). Potion embeddings must be sliced to min(n_json, n_emb) before reshape due to append-at-end entries.
- Lite mode MRR regression (0.5353->0.4815) after dictionary expansion with per-sense structure
