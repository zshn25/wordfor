# AGENTS.md

Instructions for coding agents working in this repository. Read this first, then
[`MAINTENANCE.md`](MAINTENANCE.md) for run/update/release procedures and
[`README.md`](README.md) for the product overview. Deep history and eval notes are in
[`docs/archive/`](docs/archive/).

## What this is

WordFor is a fully client-side reverse dictionary (GitHub Pages + Cloudflare, zero
server compute). The browser loads a quantized dictionary + embeddings and ranks words
by meaning. There is also a Rust/Tauri desktop port (`core/`) and an MCP server
(`wordfor-mcp/`).

## Golden rules

- **Never commit `core/.cargo/config.toml`.** It is gitignored and machine-specific; its
  `[env] CC`/`AR` leak into CI and break the macOS/Windows build matrix.
- **`build/` is gitignored in full.** Parsers, audits, training, and eval scripts are NOT
  in the repo — only their outputs (`data/*.json`, `data/*.bin`). Don't assume a build
  script is committable.
- **Keep `data/*` files < 100 MiB** (GitHub hard limit). `data/words.json` is the largest.
- **License lanes are enforced.** Only public-domain / CC0 / CC BY definition *text* may
  ship in `data/words.json`. Share-alike/proprietary sources (Wiktionary, ConceptNet, …)
  are build-time signals / hidden embeddings only — never shipped verbatim. The audits
  (`audit_sources.py`, `audit_modern_sources.py`, `verify_sources.py`) fail the build on
  any leak. See [`LICENSE_SOURCES.md`](LICENSE_SOURCES.md).
- **One asset loader.** `app.js` fetches monolithic `data/*` via `dataUrl(...)`. There is
  no sharding/manifest layer (it was removed — it produced a 404). Don't reintroduce a
  speculative loader.
- **Commit authorship:** every commit must be
  `git commit --author="Zeeshan Khan Suri <zshn25@gmail.com>"`. Do not add any other
  author/co-author/attribution trailers.

## Architecture quick map

- `app.js` — single-file app: mode selection (full/lite), model load, two-stage scoring
  (binary ITQ Hamming first pass → int3 rerank), search UI, status/progress, service
  worker registration.
- `data/` — `words.json` (576k entries: `{w,d,p,s,q,src}`), quantized embedding `.bin`
  tiers (binary, int3, potion int4 for lite), `*_ranges.bin`, `forms_to_lemma.json`,
  `source_manifest.json`, `provenance.json`.
- `core/` — Rust workspace: `wordfor-core` (shared ranking), `wordfor-tauri` (desktop),
  `wordfor-ios`/`wordfor-android`, `wordfor-uniffi`.
- `wordfor-mcp/` — TypeScript MCP server (stdio + optional HTTP); lexical engine, no model.
- `tests/asset-smoke.mjs` — fails on any missing locally-referenced asset (the 404 guard).

## words.json schema

JSON array of `{"w":[forms], "d":"definition", "p":"pos", "s":[synonyms], "q":quality,
"src":bitmask}`. `src` bits index into `data/source_manifest.json` (alphabetical order).

## Conventions / gotchas

- Windows PowerShell, cp1252: emit ASCII only in scripts; chain with `;` not `&&`; prefer
  `Set-Location` over `cd`; set `$env:PYTHONIOENCODING='utf-8'` for Python.
- Python venv: `build/.venv/Scripts/python.exe`. `cargo` is at `%USERPROFILE%\.cargo\bin`
  (not on PATH). Node ≥ 18.
- Ranking is pure cosine + quality prior; no keyword blending. Negative prefixes are never
  collapsed in lemma grouping (unhappy ≠ happy).
- After any change, run the relevant validation (see MAINTENANCE.md) before committing.

## Before you finish

Run the validation block in [`MAINTENANCE.md`](MAINTENANCE.md): asset 404 smoke test,
MCP build + smoke, source audits. Don't claim CI/eval results you didn't observe.
