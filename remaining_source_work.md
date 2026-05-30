# Remaining source work

Grounded by `build/audit_source_composition.py` (read-only, run 2026-05-30 against
the live `data/words.json`, 576,405 entries). This file separates what is *actually
ingested* from what is only *registered*, and lists the exact remaining steps. No
GPU eval below was fabricated; pending items are marked **NOT RUN** with reproducible
commands.

## 1. Compatible sources actually ingested into the visible core

Verified by decoding the per-entry `src` bitmask against `data/source_manifest.json`:

| Source | Visible entries | Status |
|--------|----------------:|--------|
| Open English WordNet 2025+ (`oewn`) | 110,283 | INGESTED, visible |
| Century Dictionary (`century`) | 137,351 | INGESTED, visible |
| Webster 1913 (`webster1913`) | 61,419 | INGESTED, visible |
| GCIDE Webster-only PD subset (`gcide_pd_webster`) | 39,564 | INGESTED, visible |
| Chambers 1908 (`chambers1908`) | 29,953 | INGESTED, visible |
| LLM-augmented CC0 (`llm_augmented`) | 61 | INGESTED, visible |

Total visible core ≈ 378,631 entries. Hidden build-time layer: `wiktionary` 197,774
(CC-BY-SA, **hidden**, used as a quality/coverage signal only — never surfaced).

### Not yet ingested but CORE_VISIBLE_OK candidates (the real remaining work)

| Source | Bitmask entries | Decision |
|--------|----------------:|----------|
| Roget 1911 (`roget1911_pg`) | 0 | **PENDING** — registered, not ingested |
| 1811 Vulgar Tongue (`vulgar_tongue_1811`) | 0 | **PENDING** — registered, not ingested |
| Moby Thesaurus (`moby_thesaurus`) | 0 as own visible entries | Build-time synonym enrichment only; the `s` synonym arrays in `words.json` are populated, but Moby contributes **no standalone visible headwords**. Adding Moby as standalone headwords is OPTIONAL and low-value (thesaurus, not definitions). |
| Moby POS / Moby Words (`moby_extra`) | 0 | Build-time signal only; `redistribute_text: false`. Keep out of visible core. |

> Correction vs earlier notes: Moby is **not** currently a visible-headword source.
> It is used to enrich synonym (`s`) arrays at build time. This is fine and stays.

## 2. Registry-only / research-only (correctly NOT in visible core)

Verified 0 entries in live data:

| Source | Class | Why excluded |
|--------|-------|--------------|
| Allen's Synonyms & Antonyms (`allens_synonyms`) | MAYBE_CORE_AFTER_VERIFICATION | Edition/license unverified; `redistribute_text: false` |
| Lewis & Short (`lewis_short`) | MAYBE_CORE_AFTER_VERIFICATION | PD-scan-only; OCR pipeline not built; jurisdiction_caution |
| OED early fascicles (`oed1_fascicles`) | research-only | PD scans need per-fascicle verification |
| GCIDE full (`gcide_full`) | NOT_CORE_COMPATIBLE | GPL — would taint core if copied |
| FreeDict (`freedict`) | NOT_CORE_COMPATIBLE | blocked |
| Reta Vortaro (`reta_vortaro`) | NOT_CORE_COMPATIBLE | blocked |
| OneLook (`onelook`) | NOT_CORE_COMPATIBLE | aggregator, no data ingest |
| EDD (`edd`) | hidden/research | scan license verification pending |
| Wiktionary (`wiktionary`) | build-time signal | CC-BY-SA, hidden only |
| ConceptNet (`conceptnet`) | build-time signal | CC-BY-SA, not redistributed |
| CMUdict (`cmudict`) | build-time signal | pronunciation only |
| OPTED (`opted`) | PD (Webster-derived) | redundant with webster1913; not ingested |

**Do not add** FreeDict, Reta Vortaro, OneLook, full GCIDE GPL, Wiktionary, or
ConceptNet to the visible core. (Enforced by `audit_sources.py` + `audit_modern_sources.py`.)

## 3. Chambers-style add procedure for the pending CORE_VISIBLE_OK sources

For Roget 1911 and Vulgar Tongue 1811, follow the exact one-at-a-time protocol
already used for Chambers (see `eval_chambers_report.md`). **These have not been run
in this session** because each isolated re-encode is a ~50-minute GPU job and no GPU
run was performed. Reproducible steps per source:

```powershell
cd d:\projects\word_for\wordfor\build
$env:PYTHONIOENCODING='utf-8'

# 0. snapshot current eval as the baseline (already exists: eval_baseline_prechambers.txt
#    -> create a new baseline AFTER Chambers so deltas isolate the new source)
.\.venv\Scripts\python.exe compare_eval.py --verbose > eval_baseline_postchambers.txt

# 1. isolated ingest of ONE source only (parser already present where noted)
#    Roget:  parser TODO (parse_roget1911.py) reading the PG #10681 plaintext
#    Vulgar: parse the PG #5402 plaintext, FILTER profanity/slurs before ingest
#    Set the source default_visible=true in sources.yaml for that source only.
.\.venv\Scripts\python.exe build_dictionary.py            # rebuild words.json

# 2. rebuild embeddings (the ~50 min GPU step)
.\.venv\Scripts\python.exe finetune_potion.py --export --base-model ./potion-distilled-mxbai
.\.venv\Scripts\python.exe export_embeddings.py           # full + quantized bins

# 3. evaluate vs baseline
.\.venv\Scripts\python.exe compare_eval.py --verbose > eval_<source>_after.txt

# 4. audits (must PASS before keep)
.\.venv\Scripts\python.exe audit_sources.py
.\.venv\Scripts\python.exe audit_modern_sources.py
.\.venv\Scripts\python.exe verify_sources.py --check

# 5. write eval_<source>_report.md with the KEEP/DROP table (template below)
```

### Roget 1911 — special handling
Roget 1911 is a **thesaurus** (categories of related words), not definitions. Ingesting
it as visible *headwords* adds little; its real value is synonym/relatedness signal.
Recommended: treat Roget as a **build-time synonym signal** like Moby (enrich `s`
arrays, do not add standalone headwords). `jurisdiction_caution: true` is already set.
If standalone ingest is still wanted, run the protocol above and decide by eval.

### Vulgar Tongue 1811 — special handling
Must run a **profanity/slur filter** before ingest (the source contains slurs and
offensive cant). Keep `default_visible: false` until the filtered set is reviewed.
This is a small (~5k entries) historical-slang pack, best shipped as an **optional
pack**, not default-visible core.

## 4. Sources that must never enter core

FreeDict, Reta Vortaro, OneLook, full GCIDE (GPL), Wiktionary, ConceptNet. Confirmed
0 visible entries; guarded by the two audit scripts (both PASS as of this session).

## 5. Missing isolated GPU eval reports — status

| Report | Required when | Status |
|--------|---------------|--------|
| `eval_roget1911_report.md` | if Roget ingested as visible | **NOT RUN** — needs GPU re-encode (steps in §3) |
| `eval_vulgar1811_report.md` | if Vulgar Tongue ingested (filtered) | **NOT RUN** — needs GPU re-encode + profanity filter |
| `eval_oewn_update_report.md` | only if OEWN release changed | **NOT NEEDED** — current ingest is 2025-edition; no newer globalwordnet release ingested |
| `eval_moby_update_report.md` | only if Moby changed | **NOT NEEDED** — Moby unchanged; build-time enrichment only |
| `eval_century_update_report.md` | only if Century changed | **NOT NEEDED** — Century unchanged (137,351 entries) |
| `eval_allen_report.md` | only if a verified PD edition is ingested | **BLOCKED** — no verified PD edition ingested |
| `eval_lewis_short_report.md` | only if a verified PD OCR pipeline runs | **BLOCKED** — no OCR pipeline built |

### KEEP/DROP report template (fill with real `compare_eval.py` numbers only)

```
# WordFor <Source> integration -- evaluation report
## Data size
| Metric | Before | After | Delta |
| Total entries / Visible / words.json MB | ... |
## Ranking metrics (N-query test set) -- from compare_eval.py --verbose
| Config (binary/int2/int3/int4/int8, mobile+desktop) | Before MRR | After MRR | dMRR | H@1 | H@6 |
## Assessment + Decision: KEEP or DROP
## License audit: audit_sources.py + audit_modern_sources.py result
```

## 6. Public-domain synonym/antonym research pass (classification only — no ingest)

Classified by US public-domain status (pre-1929 published = PD in the US). **No text
was fetched or ingested.** Source proof (a specific scan URL + sha256) must be recorded
in `sources.yaml` + `verify_sources.py` before any ingest.

| Candidate | Pub. year | US PD? | Class | Action before ingest |
|-----------|----------:|:------:|-------|----------------------|
| Edith B. Ordway, *Synonyms and Antonyms* | 1913 | Yes (pre-1929) | MAYBE_CORE_AFTER_VERIFICATION | Locate a clean Project Gutenberg / Internet Archive plaintext; record URL + sha256; build `parse_ordway.py`; ingest isolated; eval. Good fit (antonyms are valuable and we have few). |
| Samuel Fallows, *Complete Dictionary of Synonyms and Antonyms* | 1898 | Yes (pre-1929) | MAYBE_CORE_AFTER_VERIFICATION | Only if a clean OCR/plaintext exists. Verify scan quality; many editions are noisy OCR. Record proof first. |
| Henley & Farmer, *Slang and its Analogues* | 1890–1904 | Yes (pre-1929) | research-only / optional pack | Contains heavy slurs/obscenity; would require the same profanity filter as Vulgar Tongue. Ship only as an opt-in slang pack, never default-visible. Verify volume-by-volume PD scan. |
| Funk & Wagnalls *Standard Dictionary* (already have `funk_wagnalls.json` in build/) | 1893–1913 eds. | Yes for pre-1929 eds. | MAYBE_CORE_AFTER_VERIFICATION | A parsed `funk_wagnalls.json` already exists in `build/` but is **not ingested** (0 visible entries). Verify the source edition is pre-1929 and the scan is clean, then run the Chambers protocol. This is the **highest-value next add** since it is already partially parsed. |

### Recommended order for the next ingest cycle
1. **Funk & Wagnalls** (already parsed locally; verify edition + license, then eval).
2. **Ordway antonyms** (fills the antonym gap; small, clean, high value).
3. **Roget 1911** as a build-time synonym signal (not standalone headwords).
4. **Vulgar Tongue 1811** as an opt-in filtered slang pack (not default core).

Each step: isolated ingest → re-encode → `compare_eval.py` → audits → KEEP/DROP report.

## Reproducibility / honesty note
- The composition audit (§1, §2) was actually run; numbers are live.
- No GPU re-encode, no `compare_eval.py` on a new source, and no new ingest were run
  this session. All such items are marked **NOT RUN / BLOCKED** above with exact commands.
