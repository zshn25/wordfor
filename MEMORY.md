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
- **Word dedup**: Suffix-stripping stemmer (`stemWord()`) groups morphological variants in `topK()`
- **Lite mode**: distilled-mxbai (256d, bag-of-words) fine-tuned via knowledge distillation from mxbai-embed-large-v1, int8 scoring
- Pure cosine ranking + quality weights (keyword blending REMOVED per user request)
- **Wiktionary**: Build-time only (quality signals). NOT redistributed at runtime (CC-BY-SA ShareAlike risk).
- **ITQ (Iterative Quantization)**: 50-iteration rotation matrix trained on 50K subsample, 0.2% bit-flip rate

## Key Files
- `wordfor/app.js` - Main frontend (search, rendering, quality weights, ITQ+rerank)
- `wordfor/build/build_dictionary.py` - Dictionary build pipeline (4 sources + quality)
- `wordfor/build/finetune_potion.py` - Fine-tuning + evaluation (67-query test set, 6+ quant methods)
- `wordfor/build/compare_eval.py` - Cross-method evaluation (potion vs full-mode binary, 67-query set)
- `wordfor/bench.html` - Browser-side scoring + E2E benchmark page (6 configs: lite, lite-wasm, full-wasm, full-binary-rerank, full-binary-only, full-webgpu)
- `wordfor/about.html` - About page (how it works, tech details)
- `wordfor/data/words.json` - ~176K entries with `q` quality weights
- `wordfor/data/embeddings_itq.bin` - ITQ calibration (mean + rotation matrix, ~578 KB)

## Dictionary & Build-Time Sources
- OEWN 2025+ (CC BY 4.0) - primary, definitions redistributed
- Webster's 1913 (public domain) - definitions redistributed
- GCIDE Webster 1913 portion (public domain) - supplementary Webster entries, `parse_gcide.py` extracts PD-only
- Century Dictionary 1889-1911 (public domain) - 161K entries, `parse_century.py` parses from hupong Markdown
- Moby Thesaurus (public domain, synonym enrichment)
- Wiktionary via kaikki.org (CC-BY-SA 3.0): build-time quality + fine-tuning ONLY, NOT redistributed
- ConceptNet 5.7 (CC-BY-SA 4.0, build-time quality signals only, NOT redistributed)
- American Heritage Dictionary: PROPRIETARY, cannot use
- GCIDE GPL additions (WordNet/PJC): Fine-tuning only, not redistribution

## Evaluation Results — Deployed Models (67-query test set, Apr 2026 data)
| Method | Mode | MRR | Hit@1 | Hit@6 |
|--------|------|-----|-------|-------|
| Full binary+int4 rerank | Full (desktop) | 0.6375 | 37 | 53 |
| Full pure binary ITQ | Full (mobile) | 0.5401 | 28 | 46 |
| Potion fine-tuned int8 | Lite | 0.5353 | 30 | 43 |
| Potion base int8 | Lite (baseline) | 0.4621 | 25 | 38 |

## Fine-tuning (potion / lite mode)
- **Base model**: Knowledge-distilled from mxbai-embed-large-v1 -> 256d static model (Model2Vec distill)
  - Distilled base saved at `wordfor/build/potion-distilled-mxbai/`
  - IMPORTANT: Must convert float16 -> float32 before training (NaN gradients otherwise)
- **Fine-tuned model**: `wordfor/build/potion-potion-distilled-mxbai/final/`
- Sources: OEWN (115K) + Webster's (70K) + Wiktionary overlap (308K) = 493K entries, 2.3M pairs
- No SemHash dedup (near-duplicates kept as augmentation), only eval decontamination
- Distilled+FT improved over potion-base-8M FT: MRR 0.5128 -> 0.5353 (+4.4%)
- Run with: `.venv/Scripts/python.exe finetune_potion.py [--base-model ./potion-distilled-mxbai]`
- CUDA venv at `wordfor/build/.venv` (has torch CUDA + bf16 support)
- After training: `--export` to regenerate potion embeddings, then `compare_eval.py` to evaluate

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

## Known Issues
- `potion-wordnet/final/` has old potion-base-8M FT weights; current deployed model is `potion-potion-distilled-mxbai/final/`
- 4 MISSes: athazagoraphobia, bibliophilia, petrichor, glossophobia (not in dictionary)
- Compound word dominance: "art teacher" outranks "teacher"
- Antonym confusion: bag-of-words can't distinguish "fear" from "fearless"
- ConceptNet NOT viable as supplement data source (no real definitions, mostly multi-word phrases)
