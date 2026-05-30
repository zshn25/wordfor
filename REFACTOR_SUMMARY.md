# Refactor Summary

Goal: reduce complexity, not add features. This pass removed a dead asset-loading layer,
deleted dead code, fixed the production 404, consolidated documentation, and added a
deploy-time 404 guard.

## What broke and was fixed

| Problem | Root cause | Fix |
|---------|-----------|-----|
| `https://wordfor.xyz/assets/model-manifest.json` 404 in production | `shard-loader.js` fetched a manifest that is gitignored and was never deployed; sharding was never shipped | Removed the sharding layer entirely; `app.js` fetches monolithic `data/*` files directly (its existing fallback path) |
| macOS arm64 + Windows x64 desktop CI failures | `core/.cargo/config.toml` `[env] CC`/`AR` hardcoded the maintainer's local MinGW paths and applied to *all* targets, so CI runners tried to invoke `C:\Users\...\gcc.exe` | Untracked + gitignored the file; CI now uses stock per-runner toolchains |

## Deleted

- `assets/shard-loader.js` — the only consumer of the missing manifest (sharding layer).
- `assets/model-manifest.json` — gitignored, never deployed, source of the 404.
- `core/.cargo/config.toml` — removed from git (kept on disk locally, now gitignored).
- Dead int4 full-rerank code in `app.js`: `fullEmbInt4` declaration, the
  `embeddings_int4.bin` fetch branch in `loadFullRerank`, and the `else if (fullEmbInt4)`
  rerank branch. (`scoreInt4` + `potionEmbInt4` for the *live lite* path were kept — they
  use a different file, `embeddings_potion_int4.bin`.)

## Merged / simplified

- **One asset loader.** Removed `window.ShardLoader` indirection from `app.js` and the
  `<script src="assets/shard-loader.js">` from `index.html`. All `data/*` loads now go
  through a single `fetch(dataUrl(...))` path with the pre-existing graceful fallbacks.
- `assets/` directory is now empty and gone.

## Documentation consolidation

Root `.md` files reduced to the canonical set: **README.md, AGENTS.md (new),
MAINTENANCE.md (new), LICENSE_SOURCES.md** (+ `docs/mcp.md`).

- `AGENTS.md` — concise coding-agent conventions (golden rules, architecture map, schema,
  gotchas, license lanes).
- `MAINTENANCE.md` — run/update/evaluate/release procedures, repo layout, validation block.
- Moved to `docs/archive/` (history preserved via `git mv`):
  `MEMORY.md`, `asset_hosting_strategy.md`, `mcp_publish_checklist.md`, `perf_report.md`,
  `platform_verification.md`, `remaining_eval_work.md`, `remaining_source_work.md`,
  `rust_tauri_eval.md`, `seo_content_plan.md`.
- Fixed the now-relative doc link in `wordfor-mcp/README.md`.

## Added

- `tests/asset-smoke.mjs` — scans every HTML page and `app.js` for local asset references
  (`src`/`href`, `dataUrl(...)`, `fetch("…")`) and fails on any missing file. Run before
  every deploy. `embeddings_int8.bin` and `embeddings_int4.bin` are treated as optional
  (locally-generated, not deployed).
- `.gitignore`: `data/shards/` (leftover local sharding output) and
  `core/.cargo/config.toml` (CI-breaking, machine-specific).
- `sw.js` `CACHE_NAME` bumped `v16` → `v17`.

## HTML shell — intentionally left as-is

Audited `index.html`, `about.html`, `bench.html`, `blog/*.html`. Pages already share
`style.css`/`blog.css`; the remaining inline `<script>` blocks are page-specific
(per-page JSON-LD, the `?q=` noindex guard, the standalone benchmark harness), not
duplication. Introducing a templating/partials build would *add* a build step and deploy
complexity to a currently zero-build static site — contrary to the goal — so the canonical
shell was kept and no build step was added.

## Validation (all passing)

| Check | Command | Result |
|-------|---------|--------|
| Asset 404 guard | `node tests/asset-smoke.mjs` | PASS (28 refs, 9 pages) |
| MCP build + smoke | `cd wordfor-mcp && npm run build && npm run smoke` | PASS (entries 576,394 / 23 sources) |
| Source license audit | `python audit_sources.py` | PASS (no policy violations) |
| Modern source audit | `python audit_modern_sources.py` | PASS (no restricted text visible) |
| Source drift | `python verify_sources.py --check` | PASS (no drift) |
| Asset size guard | tracked files < 100 MiB | PASS (largest `words.json` 81.4 MiB) |

## New simplified file structure (top level)

```
README.md  AGENTS.md  MAINTENANCE.md  LICENSE_SOURCES.md
index.html about.html bench.html blog/        # static site (zero build)
app.js style.css sw.js manifest.json          # one app shell, one asset loader
tests/asset-smoke.mjs                          # deploy 404 guard
data/                                          # shipped dictionary + quantized embeddings
wordfor-mcp/                                   # MCP server
core/                                          # Rust/Tauri workspace
docs/mcp.md  docs/archive/                     # live MCP doc + archived reports
build/                                         # gitignored: parsers, audits, training, eval
```

## Remaining TODOs (tracked, not blocking)

- **Add approved data sources** (Allen's synonyms/antonyms, CMU dict, opengloss, MongoDB
  english-words-definitions; hidden-embedding lane for non-distributable text) and
  regenerate embeddings + eval. Requires network downloads + GPU; see
  [`docs/archive/remaining_source_work.md`](docs/archive/remaining_source_work.md) and
  [`docs/archive/remaining_eval_work.md`](docs/archive/remaining_eval_work.md).
  The working-tree `data/*` already reflects a rebuild (23 declared sources); it is left
  uncommitted to land together with the new-source ingestion.
