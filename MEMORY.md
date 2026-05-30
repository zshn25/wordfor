# WordFor Project Memory

## Project Overview
- **WordFor** (wordfor.xyz): Free reverse dictionary, runs entirely in browser
- GitHub Pages hosting (CNAME: wordfor.xyz), purely static, no backend
- GoatCounter for analytics (privacy-friendly)
- License: CC-BY-NC-ND-4.0 (considering dual: CC-BY-SA for data, CC-BY-NC-ND for code)

## Architecture
- **Full mode**: mxbai-embed-large-v1 (1024d -> 384d MRL) teacher + mdbr-leaf-mt (22M) student
- **Scoring**: Binary (ITQ-calibrated) Hamming first-pass + int3 rerank of top-500 (default desktop); pure binary on mobile
- **Mobile**: iOS auto-detects to lite mode (ONNX WASM OOM for both q8 and q4f16). Android tries full mode with BINARY_ONLY.
- **Int3 reranking**: Best MRR (0.6437), beats int4 (0.6088) and int8. Default rerank format. 8 dims packed per 3 bytes.
- **Quantization formats available**: int8 (48B/d), int4 (192B/entry, packed nibbles), int3 (144B/entry, 8 dims/3-byte word), int2 (96B/entry, 4 dims/byte), binary (48B/entry, ITQ)
- **Word dedup**: Suffix-stripping stemmer (`stemWord()`) groups morphological variants in `topK()`, with definition-overlap guard (>=0.08 Jaccard) to prevent false merges
- **RESULT_EXCLUDE**: Hard set in app.js (~50 function words/articles/pronouns) excluded from topK() results
- **Hub word penalty**: Words appearing in >5%/15% of definitions get 0.90x/0.80x quality multiplier applied AFTER `q^0.1` dampening (guarded: q_idf > 0.95 bypasses penalty)
- **Lite mode**: distilled-mxbai (256d, bag-of-words) fine-tuned via knowledge distillation from mxbai-embed-large-v1, int4 scoring
- Pure cosine ranking + quality weights (keyword blending REMOVED per user request)
- **Hidden Wiktionary entries**: Wiktionary senses for existing words added as separate entries with `h:1` flag. Participate in scoring but defs not shown to users. Replaces old averaging approach (which degraded results).
- **ITQ (Iterative Quantization)**: 50-iteration rotation matrix trained on 50K subsample, 0.2% bit-flip rate

## Key Files
- `wordfor/app.js` - Main frontend (search, rendering, quality weights, ITQ+rerank)
- `wordfor/build/export_embeddings.py` - Shared quantization (int8/int4/int3/int2) + potion model export utilities
- `wordfor/build/build_dictionary.py` - Dictionary build pipeline (5 PD sources + Moby Common + LLM augmented + Wiktionary enrichment + quality + full-mode embeddings only)
- `wordfor/build/finetune_potion.py` - Fine-tuning + evaluation (67-query test set, 6+ quant methods) + potion embedding export (int4 + int3 + binary ITQ)
- `wordfor/build/compare_eval.py` - Cross-method evaluation incl. binary+int3 rerank and pure int8 (auto-updates bench.html eval table)
- `wordfor/build/llm_augmented.json` - 61 curated LLM-written definitions (new words + improved terse entries), SOURCE_WEIGHT=0.95
- `wordfor/build/build.sh` - Unified build script (dictionary -> finetune -> export -> eval)
- `wordfor/bench.html` - Browser-side scoring + E2E benchmark page (E2E: Full q8 WASM, binary+int3 rerank, binary-only, WebGPU q4f16, Lite JS/WASM; scoring table: 1-bit through int8 + rerank combos)
- `wordfor/about.html` - About page (how it works, tech details)
- `wordfor/sw.js` - Service worker (v15): shell stale-while-revalidate, data cache-first, cross-origin model caching, offline navigation fallback
- `wordfor/manifest.json` - PWA manifest with PNG icons, share_target (?q= param), shortcuts, widgets (Edge/Win11 experimental)
- `wordfor/android-chrome-192x192.png`, `wordfor/android-chrome-512x512.png` - Purple rounded-rect "W" PNG icons (Georgia Bold)
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
- Chambers's Twentieth Century Dictionary 1908 (public domain, Project Gutenberg 37683/38538/38699/38700) - `parse_chambers1908.py` -> chambers1908.json (32,227 entries; 32,120 after build filter). Definitions redistributed; etymology kept in `ety` build-time only. Visible source. Integrated into build_dictionary.py (load_chambers_entries, merge_and_dedup, quality cross-source, provenance, src bitmask).

## Source policy & license enforcement (added 2026)
- `build/sources.yaml` + `build/source_policy.py`: typed registry of 18 classic sources; stable alphabetical source-bit-index; per-entry `src` bitmask written to words.json; `data/source_manifest.json` decodes it client-side.
- `build/audit_sources.py`: fails build if a visible (h!=1) entry carries a non-redistributable / GPL / CC-BY-SA / OED source. PASS on current data (576,405 entries: 378,631 visible, 197,774 hidden).
- **Modern lexical pipeline** (`build/modern_sources.yaml` + `modern_source_policy.py`): 15 modern HF/API sources classified by role (visible_ok / hidden_only / signal_only / api_lookup_only / do_not_copy / blocked) and zone (core / sharealike / research). Only `wordfor_generated_cc0` (original CC0 generated defs) is visible. Wiktionary/Wiktextract/OpenGloss=hidden, MongoDB/proshady/Oxford/Urban=do_not_copy (repo Apache/MIT label != text license).
  - Parsers: `parse_hf_opengloss.py`, `parse_hf_wiktionary_sqlite.py`, `parse_hf_wiktextract.py`, `parse_hf_english_valid_words.py`, `parse_hf_modern_definitions.py`, `parse_hf_slang.py` (all graceful-skip offline; `hf_common.py` drops gloss text unless source may_provide_hidden_text).
  - `modern_candidate_terms.py` -> modern_candidates.jsonl (display_definition_allowed always false). `generate_modern_definitions.py` prepare/ingest with n-gram+Jaccard copy-guard, CC0 defs q0.65-0.80.
  - `audit_modern_sources.py`: fails build if any restricted source contributes visible copied text (4 checks, self-test + live PASS). Wired into build.sh Step 5 alongside audit_sources.py.
  - `source_license_report.md`: zone/role reference. `eval_modern_ablation.py`: ablation matrix (license-safety + coverage columns; MRR pending per-config embeddings).

## Source expansion + verification (added 2026-05)
- `build/sources.yaml` now classifies every source: `source_class` = CORE_VISIBLE_OK / MAYBE_CORE_AFTER_VERIFICATION / NOT_CORE_COMPATIBLE, plus optional `version`, `checksum_sha256`, `jurisdiction_caution`, `provenance_audited`, `verification_status`. 23 sources total.
  - Added: `allens_synonyms` + `lewis_short` (MAYBE_CORE, research-only until edition/license proof); `freedict` + `reta_vortaro` + `onelook` (NOT_CORE_COMPATIBLE, blocked role). Roget1911 + Lewis&Short marked `jurisdiction_caution: true`.
  - `roget1911_pg` / `vulgar_tongue_1811` stay hidden-by-default; visible only after audit / profanity filter.
- `build/verify_sources.py` -> `data/source_checksums.json`: sha256 + byte size of all 13 ingested source files (oewn 11.3MB, webster 9.0MB, gcide_webster 5.7MB + tar 18.9MB, century 19.4MB, chambers 7.3MB, moby 24.9MB, kaikki 457MB, conceptnet 498MB, ...). `--check` flag fails on checksum/status drift. Wired into build.sh Step 1b.
- Honest status: Part B per-source Chambers-style isolated re-encode + compare_eval reports (Roget/Vulgar/Allen/OEWN-update/Moby/Century) are NOT yet run -- each needs a ~50-min GPU encode. Framework (verify_sources, compare_eval, eval_modern_ablation) is ready; eval_*_report.md files pending real runs.

## Lemma / word-family canonicalization (added 2026-05)
- `build/build_lemma_families.py` moves the old JS suffix-stemmer into the build. Outputs (visible, from words.json visible vocab = 200,863 words):
  - `data/forms_to_lemma.json` (287 KB, 12,725 collapses) -- inflected form -> canonical lemma, runtime lookup.
  - `data/lemma_families.json` (4.3 MB, 27,541 families) -- inflectional + derivational_related + antonym_or_negative_prefix members per lemma.
  - `data/lemma_family_provenance.json` -- per-link evidence (source_id, license_class, evidence_type, visible_allowed, confidence).
- Evidence (all core-compatible/owned): `wordfor_irregular_table` (curated CC0, high conf), `wordfor_morph_rules` (regular morphology, CC0, medium), `core_vocab_membership` (medium). Collapse needs >=1 high OR >=2 independent medium + >=1 visible-compatible.
- SAFE collapses only: plurals (cats->cat, boxes->box, mice->mouse), verb forms (running->run, went->go, walked->walk), comparatives (happier/happiest->happy, better->good). NEVER strip negative/derivational prefixes (unhappy != happy, impossible != possible, invaluable != valuable). Derivational (-ness/-ity/...) recorded as related, NOT collapsed. Ambiguous forms (>=2 lemmas) -> uncertain, no collapse. Real run: 0 unsafe prefix collapses, 10,222 negative-prefix relations recorded, 851 uncertain.
- Runtime: `app.js` `stemWord()` removed; `canonicalLemma()` looks up `forms_to_lemma.json` (loaded in `loadWordList`, non-fatal if absent); `topK` groups by canonical lemma (`lemmaToGroup`). Original typed form preserved for display.
- `audit_sources.py` extended: fails on negative-prefix collapse (unX->X etc.), self-map, or a visible collapse whose provenance is only non-compatible. Self-test + live PASS. Wired into build.sh Step 1c + audit.
- `LICENSE_SOURCES.md` (repo root): full source license/visibility table + lemma provenance table.

## Evaluation Results — Deployed Models (67-query test set, post-Chambers 576,405 entries)
| Method | Mode | MRR | Hit@1 | Hit@6 |
|--------|------|-----|-------|-------|
| Full pure int8 | Full | 0.6439 | 36 | 55 |
| Full binary+int3 rerank | Full (desktop) | 0.6346 | 35 | 52 |
| Full pure int3 | Full | 0.6348 | 35 | 52 |
| Full pure int4 | Full | 0.6339 | 36 | 52 |
| Full binary+int4 rerank | Full | 0.6337 | 36 | 52 |
| Full pure binary ITQ | Full (mobile) | 0.6102 | 34 | 51 |
| Full pure int2 | Full | 0.6100 | 35 | 45 |
| Potion fine-tuned int4 | Lite | 0.5701 | 34 | 43 |
| Potion base int4 | Lite (baseline) | 0.5003 | 27 | 41 |

Chambers 1908 added 29,953 visible entries (+30,431 total, 81.0->84.9 MB). Mobile
binary +0.047 MRR and int2 +0.043 are the biggest gains; int8 best-ever (0.6439,
H@6 55). Desktop default (binary+int3) dipped -0.009 MRR / 2 queries (noise).
KEPT. Full before/after in build/eval_chambers_report.md, baseline in
build/eval_baseline_prechambers.txt.

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
- Fav color: purple. UI palette: purple primary, orange/green secondary
- Dark mode: automatic based on system theme (prefers-color-scheme)

## Known Issues / Status
- `potion-wordnet/final/` has old potion-base-8M FT weights; current deployed model is `potion-potion-distilled-mxbai/final/`
- Previously-missing words (petrichor, glossophobia, athazagoraphobia, bibliophilia, etc.) are now in `llm_augmented.json` — will be in words.json after next rebuild
- words.json patched directly: 3 Century garbage entries fixed/removed. Embeddings still use old vectors until rebuild.
- Compound word dominance: "art teacher" outranks "teacher"
- Antonym confusion: bag-of-words can't distinguish "fear" from "fearless"
- words.json is per-sense format: 350K entries with 74K duplicated primary headwords. Potion embeddings must be sliced to min(n_json, n_emb) before reshape.

## PWA
- PNG icons at `/android-chrome-192x192.png` and `/android-chrome-512x512.png` (purpose "any" + "maskable")

## Release Session (2026-05-30): source policy, lemma families, sharding, Tauri/CI, MCP
- **Versioning**: `sw.js` CACHE_NAME -> `wordfor-v16`; `app.js` DATA_VERSION -> `v3` (busts CDN cache for regenerated data); `sitemap.xml` lastmod 2026-05-30.
- **Source policy / provenance**: `data/source_manifest.json` (18 sources, alphabetical bit_index 0-17), `data/source_checksums.json`, `data/provenance.json`, `LICENSE_SOURCES.md`. words.json schema adds `src` bitmask. wiktionary (bit 17) ingested but hidden from per-source labels where `redistribute_text=false`.
- **Lemma families**: build-time maps `data/forms_to_lemma.json`, `data/lemma_families.json`, `data/lemma_family_provenance.json` replace runtime stemming. app.js consumes forms_to_lemma; non-fatal if absent.
- **Asset sharding**: `assets/shard-loader.js` (window.ShardLoader, cache-first, monolithic fallback) + `asset_hosting_strategy.md`. app.js guards every shard call with `if (window.ShardLoader && await isSharded(...))` -> graceful monolith fallback (one cached manifest 404 in prod, expected). `data/shards/` + `assets/model-manifest.json` are LOCAL-ONLY (gitignored, regenerable).
- **int4 untracked**: `data/embeddings_int4.bin` (105 MB, unused) was the only >=100 MiB GitHub push blocker -> `git rm --cached` + gitignored. `build/check_asset_sizes.py` now hard-fails only on >=100 MiB or shards >25 MiB; 10-100 MB monoliths are non-fatal advisories (`--strict` to enforce).
- **Tauri/Rust**: `core/` workspace (wordfor-core PASS; wordfor-tauri NOT RUN locally due to mixed GNU/MSVC toolchain - see `rust_tauri_eval.md`). CI: `.github/workflows/build-desktop.yml` (4-target matrix). NOTE: the workflow's asset-size-guard step calls `build/check_asset_sizes.py`, which is GITIGNORED and absent from the repo - must inline the check or vendor the script before that job can pass.
- **CRITICAL repo fact**: `.gitignore` has `build/*` -> the ENTIRE `build/` dir (all parsers/audit/build scripts) is untracked. Only data outputs (`data/*.json`) and root docs are committable. `git ls-files build` = 0.
- **MCP server**: `wordfor-mcp/` (TypeScript, @modelcontextprotocol/sdk + zod). 6 tools (reverse_lookup, search_word, explain_ranking, get_word_family, get_sources, health), 4 resources, 4 prompts. Stdio default; `--http` -> StreamableHTTP with per-IP rate limit. Lexical engine (no model download). `npm run build` + `test/smoke.mjs` PASS (HEALTH entries:576405 sources:18). Docs in `docs/mcp.md`; status in `mcp_publish_checklist.md` (NOT published).
- **Eval status**: `remaining_eval_work.md` lists all GPU evals = NOT RUN (Funk&Wagnalls, Ordway, Roget1911, Vulgar1811, modern ablation) with exact commands/baselines/output paths. int3 size guard: 144 bytes/entry, 576405 -> 79.2 MiB, max 728177 entries under 100 MiB (headroom ~152K / 26%).
- **Platform verification**: `platform_verification.md` per-target (macOS x64/arm64, Windows x64, Linux x64) = CI PENDING, artifacts UNKNOWN, runtime/search NOT RUN (cannot run until pushed to remote).
- **Commit author (MANDATORY)**: `git commit --author="Zeeshan Khan Suri <zshn25@gmail.com>"` - NO Claude/Copilot attribution or co-author trailers anywhere.
- Service worker `sw.js` (v15): precaches app shell, cache-first for data/model, cross-origin model caching (HuggingFace CDN)
- Share target: `?q={text}` — receiving shared text opens WordFor with that query
- Widget: Adaptive Cards template at `/widget-search.json` (experimental Edge/Win11)
- Potion export now generates: int4 + int3 + binary (ITQ-calibrated) embeddings

## Rust Core (Native Apps)
- Workspace at `wordfor/core/` with 4 crates: wordfor-core, wordfor-tauri, wordfor-android, wordfor-ios
- `wordfor-core`: All 5 scoring kernels (int8, int4, int3, hamming, binary_rerank), ranking/dedup, data loading, query encoding
- 14 unit tests pass (scoring + stemmer parity with app.js)
- Features: `onnx` (ort crate), `potion` (tokenizers+safetensors), `mmap` (memmap2)
- ONNX Runtime: no prebuilt for x86_64-pc-windows-gnu; needs MSVC target or custom build
- **Build toolchain**: Requires WinLibs MinGW on PATH for dlltool/windres/gcc
  - Install: `winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT.LLVM`
  - PATH: append `/c/Users/z.suri/AppData/Local/Microsoft/WinGet/.../mingw64/bin`
  - CC/AR/linker configured in `.cargo/config.toml`
- Tauri desktop app: compiles, uses `web-dist/` copy of frontend files, bundles data via resources
- Android/iOS: stub crates (cdylib/staticlib), UniFFI bindings planned

