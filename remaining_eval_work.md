# Remaining evaluation work (GPU)

Every ranking/quality evaluation that has **NOT** been run, with the exact command, the
input baseline, the output report path, and status. Nothing here is fabricated.

**Honesty gate:** source expansion is **NOT** considered fully evaluated until the reports
below exist with real `compare_eval.py` numbers. The composition/license audits have been
run (see `remaining_source_work.md`), but no new GPU re-encode + ranking eval has been run
this session.

## Shared baseline

The current deployed baseline (post-Chambers, 576,405 entries) lives in
`build/eval_chambers_report.md` and the MEMORY.md eval table:

| Method | Mode | MRR | Hit@1 | Hit@6 |
|--------|------|----:|------:|------:|
| Full pure int8 | Full | 0.6439 | 36 | 55 |
| Full binary+int3 rerank | Full (desktop default) | 0.6346 | 35 | 52 |
| Full pure binary ITQ | Full (mobile) | 0.6102 | 34 | 51 |
| Potion fine-tuned int4 | Lite | 0.5701 | 34 | 43 |

Before any new-source eval, snapshot a fresh baseline so deltas isolate the new source:

```powershell
cd d:\projects\word_for\wordfor\build
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe compare_eval.py --verbose > eval_baseline_postchambers.txt
```
Status: **NOT RUN** (baseline snapshot file not yet generated this session).

## Per-source GPU evals (all NOT RUN)

Each is a one-source-at-a-time isolated ingest → ~50-min GPU re-encode → eval, per the
Chambers protocol in `remaining_source_work.md` §3.

| # | Source | Exact command (after isolated ingest in `build/`) | Input baseline | Output report | Status |
|---|--------|---------------------------------------------------|----------------|---------------|--------|
| 1 | Funk & Wagnalls (already parsed: `funk_wagnalls.json`) | `build_dictionary.py` → `finetune_potion.py --export --base-model ./potion-distilled-mxbai` → `export_embeddings.py` → `compare_eval.py --verbose > eval_funkwagnalls_after.txt` | `eval_baseline_postchambers.txt` | `build/eval_funkwagnalls_report.md` | **NOT RUN** |
| 2 | Ordway *Synonyms and Antonyms* 1913 | (build `parse_ordway.py` first) then same chain → `eval_ordway_after.txt` | `eval_baseline_postchambers.txt` | `build/eval_ordway_report.md` | **NOT RUN** (parser not built) |
| 3 | Roget 1911 (as build-time synonym signal) | same chain → `eval_roget1911_after.txt` | `eval_baseline_postchambers.txt` | `build/eval_roget1911_report.md` | **NOT RUN** (parser not built) |
| 4 | Vulgar Tongue 1811 (profanity-filtered, opt-in pack) | filter → same chain → `eval_vulgar1811_after.txt` | `eval_baseline_postchambers.txt` | `build/eval_vulgar1811_report.md` | **NOT RUN** (filter + parser not built) |
| 5 | Modern ablation MRR (per-config embeddings) | `eval_modern_ablation.py` (currently emits license-safety + coverage only; MRR needs per-config embeddings) | post-Chambers | `build/eval_modern_ablation_report.md` | **NOT RUN** (MRR columns pending per-config encode) |

## Blocked evals (cannot run until upstream prerequisites met)

| Source | Blocker | Output report | Status |
|--------|---------|---------------|--------|
| Allen's Synonyms & Antonyms | No verified PD edition/license; `redistribute_text: false` | `build/eval_allen_report.md` | **NOT RUN — BLOCKED** |
| Lewis & Short | No PD-scan OCR pipeline built; jurisdiction_caution | `build/eval_lewis_short_report.md` | **NOT RUN — BLOCKED** |

## Evals explicitly NOT needed (source unchanged)
- `eval_oewn_update_report.md` — OEWN edition unchanged (2025).
- `eval_century_update_report.md` — Century unchanged (137,351 entries).
- `eval_moby_update_report.md` — Moby unchanged (build-time enrichment only).

## int3 size guard (relevant to new-source evals)

Each new visible source grows `embeddings_int3.bin` (the desktop rerank format) at
**144 bytes/entry**. Measured headroom before the 100 MiB GitHub hard limit:

- current: 576,405 entries → 83,002,320 B (79.2 MiB)
- max under 100 MiB: **728,177 entries** → headroom **151,772 entries (~26%)**

So a single moderate source (e.g. F&W ~40-90k entries) stays well under 100 MiB. Even if
int3 later crosses 100 MiB, `build/split_assets.py` already shards it into ≤15 MiB pieces
(loaded via `assets/shard-loader.js`), and `build/check_asset_sizes.py` BLOCKS any tracked
blob ≥100 MiB. Re-run `python build/check_asset_sizes.py --git` after every new-source build.

## Reproducibility / honesty note
- No GPU re-encode and no `compare_eval.py` on a new source were run this session.
- Do not claim source expansion is fully evaluated until reports 1–5 exist with real numbers.
