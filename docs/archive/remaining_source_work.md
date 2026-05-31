# Remaining source work

Grounded by `build/audit_source_composition.py` (read-only, run 2026-05-30 against
the live `data/words.json`, 576,405 entries). This file separates what is *actually
ingested* from what is only *registered*, and lists the exact remaining steps. No
GPU eval below was fabricated; pending items are marked **NOT RUN** with reproducible
commands.

## 0. 2026-05-31 source ingest cycle (Allen's, CMUdict, OpenGloss, MongoDB)

Four sources were taken end-to-end this cycle. Status against the DONE criteria
(declared in registry, parser wired, in `source_manifest.json`, audits pass):

| Source | License (verified) | Outcome | Where |
|--------|--------------------|---------|-------|
| Allen's Synonyms & Antonyms (F. Sturges Allen, 1920) | Public domain (pre-1924 US, PG #51155, OL6369686M) | **DONE** — ingested as synonym enrichment | `s[]` arrays |
| CMU Pronouncing Dictionary | BSD-2-Clause (CMU) | **DONE** — ingested as cross-source coverage/quality signal | `q` score |
| OpenGloss (`opengloss_v1`/`v13`) | CC-BY-4.0 | **BLOCKED** (visible text) — see §0.1 | n/a |
| MongoDB english-words-definitions | Apache-2.0 (repo label) | **BLOCKED** (visible text) — see §0.1 | n/a |

### Allen's — DONE (synonym word-lists, like Moby)
- Verified public domain: F. Sturges Allen, *Allen's Synonyms and Antonyms* (1920),
  pre-1924 US publication; Project Gutenberg #51155; OpenLibrary OL6369686M.
- `sources.yaml`: `redistribute_text: true`, `verification_status: verified`,
  `parser: build_dictionary.load_allens_synonyms`, bit index 0.
- Parser `download_allens` + `load_allens_synonyms` parse the `KEY:/SYN:/ANT:` block
  format into headword → single-word synonyms; `enrich_with_allens` merges them into
  the existing `s[]` field alongside Moby (no new visible headwords, no prose text).
- Live run: **6,254 headwords loaded; 27,083 entries gained synonyms.**
- Because synonyms are word-lists (not definition prose), Allen's does **not** appear
  in any visible entry's `src` bitmask — same precedent as Moby. This is correct and
  keeps `audit_sources.py` green.

### CMUdict — DONE (cross-source coverage signal)
- BSD-2-Clause (attribution required). Pronunciations are **not** shipped.
- `sources.yaml`: `parser: build_dictionary.load_cmudict`, `redistribute_text: false`,
  bit index 3.
- Parser `download_cmudict` + `load_cmudict` extract the headword set (126,052 words);
  `compute_quality_scores` adds CMUdict membership as one more cross-source agreement
  vote (alongside OEWN/Webster/GCIDE/Century/Chambers/Wiktionary/ConceptNet/Moby).
- Live run: **126,052 headwords loaded**, fed into the `q` cross-source term.

### 0.1 OpenGloss + MongoDB — BLOCKED (visible text), with exact reasons
Both are **modern** sources (governed by `build/modern_sources.yaml` +
`modern_source_policy.py`), not the classic `sources.yaml` core. They are blocked
for **visible definition text** for two independent reasons, either of which is
sufficient:

1. **Policy class.** `modern_sources.yaml` classifies `opengloss_v1`/`opengloss_v13`
   as `hidden_only` and `mongodb_words_definitions` as `do_not_copy`. The modern
   policy self-test and `audit_modern_sources.py` enforce that the **only** modern
   source whose text may be shown is `wordfor_generated_cc0`. Promoting either to
   visible would break `modern_source_policy.visible_text_sources() == ["wordfor_generated_cc0"]`.
2. **No visible-text generator.** The only sanctioned path to a *visible* modern
   definition is `generate_modern_definitions.py`, which is supposed to use an LLM to
   write **original CC0** glosses from the modern candidate terms. That file is
   **empty (0 bytes) / unimplemented**, and implementing it requires an LLM API key
   and a generation+QA budget that is not available in this environment. Until it
   exists, there is no legal way to surface OpenGloss/MongoDB definition prose.

**What is still possible and not blocked** (the "hidden embedding / word-without-meaning"
design the user described): OpenGloss (CC-BY) may contribute (a) candidate terms and
(b) hidden embeddings (`h=1`) so a rare word participates in scoring while only the
*word* is shown — never the imported gloss. MongoDB (`do_not_copy`) may contribute
candidate terms / coverage diffs only. These require running the modern HF pipeline
(`parse_hf_opengloss.py` → `modern_candidate_terms.py`) and wiring the resulting
hidden entries into `build_dictionary.py`. That ingestion is **NOT YET WIRED** (the
classic build does not read `modern_*.jsonl`), so it remains the concrete next step
for the modern workstream — distinct from the visible-text path, which stays BLOCKED.

---

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
| Allen's Synonyms & Antonyms (`allens_synonyms`) | MAYBE_CORE_AFTER_VERIFICATION → **verified PD** | Now **ingested as synonym enrichment** (2026-05-31): `redistribute_text: true`; 27,083 entries gained `s[]` synonyms. Contributes no visible *headwords* and no prose, so absent from the `src` bitmask (same as Moby). |
| Lewis & Short (`lewis_short`) | MAYBE_CORE_AFTER_VERIFICATION | PD-scan-only; OCR pipeline not built; jurisdiction_caution |
| OED early fascicles (`oed1_fascicles`) | research-only | PD scans need per-fascicle verification |
| GCIDE full (`gcide_full`) | NOT_CORE_COMPATIBLE | GPL — would taint core if copied |
| FreeDict (`freedict`) | NOT_CORE_COMPATIBLE | blocked |
| Reta Vortaro (`reta_vortaro`) | NOT_CORE_COMPATIBLE | blocked |
| OneLook (`onelook`) | NOT_CORE_COMPATIBLE | aggregator, no data ingest |
| EDD (`edd`) | hidden/research | scan license verification pending |
| Wiktionary (`wiktionary`) | build-time signal | CC-BY-SA, hidden only |
| ConceptNet (`conceptnet`) | build-time signal | CC-BY-SA, not redistributed |
| CMUdict (`cmudict`) | build-time signal | **Ingested as coverage signal** (2026-05-31): 126,052 headwords feed the cross-source `q` term; pronunciations not shipped. |
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
