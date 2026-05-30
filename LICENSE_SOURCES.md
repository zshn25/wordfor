# WordFor source license map

WordFor is built to be **statically deployable and license-clean in its visible
core**. The shipped bundle (`data/words.json` + embeddings) contains only text we
may legally redistribute. Everything else is used as **build-time signals** (it
shapes quality scores and hidden embeddings but is never shipped verbatim) or is
**blocked** from core entirely.

Three lanes:

| Lane | What it means | Ships visible text? |
|------|---------------|---------------------|
| **Clean core** | Public-domain / CC0 / CC BY definition text | Yes (with attribution) |
| **Build-time signals** | Share-alike / copyleft / proprietary used only to compute scores, hidden embeddings, candidate terms | No |
| **Optional share-alike / GPL packs** | Could be shipped *only* as a separate, correctly-licensed artifact | Not in core |

Authoritative machine-readable policy: [`build/sources.yaml`](build/sources.yaml)
(classic / PD registry) and [`build/modern_sources.yaml`](build/modern_sources.yaml)
(modern candidate sources). The build **fails** (`audit_sources.py`,
`audit_modern_sources.py`) if any non-redistributable source leaks visible text.

## Core-visible sources (`CORE_VISIBLE_OK`)

| Source | Year | License | Use | Visible | Attribution |
|--------|------|---------|-----|---------|-------------|
| Open English WordNet | 2025 | CC BY 4.0 | Primary definitions + relations | Yes | Required |
| Webster's Unabridged | 1913 | Public domain (US) | Definitions | Yes | Required |
| GCIDE Webster PD subset | 1913 | Public domain (Webster-tagged only) | Definitions | Yes | Required |
| Century Dictionary | 1889–1911 | Public domain (US) | Definitions | Yes | Required |
| Chambers's Twentieth Century Dictionary | 1908 | Public domain (US / PG) | Definitions | Yes | Required |
| Moby Thesaurus II | 1996 | Public domain (author grant) | Synonyms (`s[]` lists, not defs) | Lists only | Required |
| WordFor LLM-augmented | 2025 | CC0 | Original factual definitions | Yes | Optional |
| WordFor generated (modern) | 2026 | CC0 | Original generated definitions | Yes | Optional |
| Roget's Thesaurus | 1911 | Public domain (US) — **jurisdiction caution** | Synonym/semantic signals; visible **only if audit passes** | Hidden by default | Required |
| 1811 Dictionary of the Vulgar Tongue | 1811 | Public domain (US / PG) | Historical slang; visible **only after profanity/quality filter** | Hidden by default | Required |

> **Jurisdiction caution** (Roget/MICRA, Lewis & Short): public-domain status can
> differ by country. Marked `jurisdiction_caution: true` in the registry.

## Maybe-core, pending verification (`MAYBE_CORE_AFTER_VERIFICATION`)

| Source | License | Status | Notes |
|--------|---------|--------|-------|
| Allen's Synonyms and Antonyms | PD (verify edition) | Research-only | Use only a verified pre-1924 US PD edition; never later copyrighted/revised editions. Promote to core only after edition + license proof recorded. |
| Lewis & Short (1879) | Original PD; common digital copies CC BY-SA / CC BY-NC-SA | Research-only | **Never** ingest Perseus/plaintext CC-SA derivatives into core. Verified PD scan/OCR pipeline only. |
| OED1 / NED fascicles | Per-fascicle verification | Backlog | Only individually verified pre-1924 US PD scans, manually OCR-cleaned with provenance. Never modern OED/OUP text. |

## Build-time signals only — not redistributed

| Source | License | Role |
|--------|---------|------|
| Wiktionary (via kaikki.org) | CC BY-SA 3.0 | Hidden embeddings, candidate terms, quality cross-validation |
| ConceptNet 5.7 | CC BY-SA 4.0 | Knowledge-graph centrality/diversity quality signals |
| GCIDE (full) | GPL-3.0-or-later | Hidden embeddings / training only |
| Open Roget's | CC BY-SA 4.0 | Quality / training signals only |
| Moby Project extras (POS, Words II) | PD | Form/quality lists only, never definitions |
| CMU Pronouncing Dictionary | BSD-2-like | Pronunciation/variant signals, OCR-garbage filter |
| English-Valid-Words | Unlicense | Frequency / valid-word numeric signals |

> Wiktionary/wiktextract output is CC BY-SA/GFDL: we may *consume* it for signals,
> but our clean core cannot absorb its text. If ever shipped, it must go into a
> **separate CC-BY-SA attributed pack**, not `words.json`.

## Not core-compatible — blocked from core (`NOT_CORE_COMPATIBLE`)

| Source | License | Decision |
|--------|---------|----------|
| FreeDict | Per-dictionary GPL / GFDL / CC BY-SA | Not core. Build-time signal only where a specific dictionary's terms allow, else a separate share-alike/GPL pack. |
| Reta Vortaro (ReVo) | GPL-2.0 | Not core. Research/build-time only; no visible text. |
| OneLook | Meta-search index | Not a data source. Manual discovery/reference only; never scraped/imported. |
| Merriam-Webster API / Wordnik API | Proprietary / API terms | API lookup only under their terms; never bulk-collected or shipped. |
| Urban Dictionary / Oxford mirrors / "words-definitions" dumps | Unknown / mislabeled | Coverage/discovery diff only; definitions never copied anywhere. |

## Lemma / word-family canonicalization provenance

The visible lemma artifacts (`data/forms_to_lemma.json`,
`data/lemma_families.json`, `data/lemma_family_provenance.json`) are derived only
from **core-compatible evidence**:

| Evidence source | License class | Confidence |
|-----------------|---------------|------------|
| `wordfor_irregular_table` (curated irregulars) | CC0 (owned) | High |
| `wordfor_morph_rules` (regular morphology) | CC0 (owned) | Medium |
| `core_vocab_membership` (OEWN/Webster/Chambers/Century headwords) | Core-compatible | Medium |

A form is **collapsed** (treated as the same lemma at search time) only with
≥1 high-confidence or ≥2 independent medium-confidence sources, and at least one
visible-compatible source. Negative/derivational prefixes (`un-`, `in-`, `im-`,
`il-`, `ir-`, `non-`, `dis-`, …) are **never** collapsed; `audit_sources.py` fails
the build if they are.
