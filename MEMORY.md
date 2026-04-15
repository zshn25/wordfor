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
- **Word dedup**: Suffix-stripping stemmer (`stemWord()`) groups morphological variants in `topK()`, with definition-overlap guard (>=0.08 Jaccard) to prevent false merges
- **Lite mode**: distilled-mxbai (256d, bag-of-words) fine-tuned via knowledge distillation from mxbai-embed-large-v1, int4 scoring
- Pure cosine ranking + quality weights (keyword blending REMOVED per user request)
- **Wiktionary**: Build-time only (quality signals + fine-tuning). NOT redistributed at runtime (CC-BY-SA ShareAlike risk).
- **ITQ (Iterative Quantization)**: 50-iteration rotation matrix trained on 50K subsample, 0.2% bit-flip rate

## Key Files
- `wordfor/app.js` - Main frontend (search, rendering, quality weights, ITQ+rerank)
- `wordfor/build/export_embeddings.py` - Shared quantization (int4/int8) + potion model export utilities
- `wordfor/build/build_dictionary.py` - Dictionary build pipeline (4 sources + quality + full-mode embeddings only)
- `wordfor/build/finetune_potion.py` - Fine-tuning + evaluation (67-query test set) + potion embedding export
- `wordfor/build/compare_eval.py` - Cross-method evaluation (auto-updates bench.html eval table)
- `wordfor/build/build.sh` - Unified build script (dictionary -> finetune -> export -> eval)
- `wordfor/bench.html` - Browser-side scoring + E2E benchmark page
- `wordfor/about.html` - About page (how it works, tech details)
- `wordfor/data/words.json` - ~350K entries with `w[]` (variants), `s[]` (synonyms), `q` (quality)
- `wordfor/data/embeddings_itq.bin` - ITQ calibration (mean + rotation matrix, ~578 KB)

## Dictionary & Build-Time Sources
- OEWN 2025+ (CC BY 4.0) - primary, definitions redistributed
- Webster's 1913 (public domain) - definitions redistributed
- GCIDE Webster 1913 portion (public domain) - supplementary Webster entries, `parse_gcide.py` extracts PD-only
- Century Dictionary 1889-1911 (public domain) - 161K entries, `parse_century.py` parses from hupong Markdown
- Moby Thesaurus (public domain, synonym enrichment -> stored in `s[]` field, separate from `w[]` variants)
- Wiktionary via kaikki.org (CC-BY-SA 3.0): build-time quality + fine-tuning ONLY, NOT redistributed
- ConceptNet 5.7 (CC-BY-SA 4.0, build-time quality signals only, NOT redistributed)

## Fine-tuning (potion / lite mode)
- **Base model**: Knowledge-distilled from mxbai-embed-large-v1 -> 256d static model (Model2Vec distill)
  - Distilled base saved at `wordfor/build/potion-distilled-mxbai/`
- **Fine-tuned model**: `wordfor/build/potion-potion-distilled-mxbai/final/`
- Sources: OEWN + Webster's + GCIDE + Century + Wiktionary overlap
- Run with: `wordfor/build/.venv/Scripts/python.exe finetune_potion.py --base-model ./potion-distilled-mxbai`
- After training: `--export` to regenerate potion embeddings, then `compare_eval.py` to evaluate
- CUDA venv at `wordfor/build/.venv` (has torch CUDA + bf16 support)

## Known Issues
- `potion-wordnet/final/` has old potion-base-8M FT weights; current deployed model is `potion-potion-distilled-mxbai/final/`
- 4 MISSes: athazagoraphobia, bibliophilia, petrichor, glossophobia (not in dictionary)
- Compound word dominance: "art teacher" outranks "teacher"
- Antonym confusion: bag-of-words can't distinguish "fear" from "fearless"
