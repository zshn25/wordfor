# Performance report & staged-loading plan

Goal: render a usable page **immediately**, enable **fast search** as soon as the
binary model is ready, then **upgrade ranking** in the background without blocking the UI.

## What is already true (verified by reading app.js)

The loader is already phased, not monolithic:

| Phase | Code | What loads | Result |
|-------|------|-----------|--------|
| 0 | static `index.html` + `style.css` | header, search box, showcase | page paints with no model |
| 1 | `loadWordList()` | `words.json` + `forms_to_lemma.json` | vocabulary + lemma grouping |
| 2 | `loadFullModel()` | mdbr-leaf-mt (q8/q4f16) + `embeddings_binary` + ITQ | **fast search (binary Hamming)** |
| 3 | `loadFullRerank()` (lazy, after app shown) | `embeddings_int3` (background) | **best ranking (binary + int3 rerank)** |

Mobile / `?scoring=binary` skips Phase 3 entirely (binary-only).

## Changes made this session

1. **Cache-first sharded loading** via `assets/shard-loader.js` + `assets/model-manifest.json`:
   - `words.json`, `forms_to_lemma.json`, `embeddings_int3` now load through `ShardLoader`
     when a manifest is deployed, with monolithic fallback otherwise.
   - Shards are cached in the Cache API keyed by immutable sha256 URLs -> repeat visits
     are near-instant.
2. **Perf instrumentation** (`perf` object in `app.js`, exposed as `window.wordforPerf()`):
   - `words_loaded` (ms to load+parse the word list),
   - `search_ready_fast_ms` (time from app start to fast search enabled / app shown),
   - `search_ready_best_ms` (time until int3 rerank is loaded).

### How to read the instrumentation
Open the deployed site, then in DevTools console:
```js
window.wordforPerf()
// -> { sinceStart_ms, measures: { words_loaded, search_ready_fast_ms, search_ready_best_ms } }
```
Cache hit/miss is visible in DevTools -> Network (size column shows "(disk cache)" /
"(ServiceWorker)" / Cache Storage) and in `ShardLoader` `onProgress(pct, fromCache)`.

## Measurements — STATUS

**NOT RUN as automated Lighthouse/field numbers this session.** No headless browser run
was performed, so no fabricated TTI/LCP figures are recorded here. The instrumentation
above is in place to capture them. Reproducible capture steps:

```powershell
# Serve the site locally
cd d:\projects\word_for\wordfor
python -m http.server 8080
# In another shell, run Lighthouse (Node required):
npx lighthouse http://localhost:8080/ --only-categories=performance --output=json --output-path=lh-home.json --chrome-flags="--headless"
# Blog page (should be near-static, no model load):
npx lighthouse http://localhost:8080/blog/reverse-dictionary-how-to-find-a-word.html --output=json --output-path=lh-blog.json --chrome-flags="--headless"
```
Record into the table below from the real runs:

| Metric | Home (app) | Blog page | Target |
|--------|-----------:|----------:|--------|
| First Contentful Paint | _TBD_ | _TBD_ | < 1.5 s |
| Largest Contentful Paint | _TBD_ | _TBD_ | < 2.5 s |
| Time to Interactive (UI) | _TBD_ | _TBD_ | < 2 s (no model) |
| `search_ready_fast_ms` | _TBD_ | n/a | < 6 s warm cache |
| `search_ready_best_ms` | _TBD_ | n/a | background |
| First query latency | _TBD_ | n/a | < 200 ms |
| Rerank latency | _TBD_ | n/a | < 150 ms |
| Cache hit (2nd visit) | _TBD_ | _TBD_ | > 90% bytes |

## Remaining staged-loading work (planned, not all done)

1. **Explicit status states in the UI.** Add a small badge driven by readiness:
   - "Ready: fast mode" when `search_ready_fast_ms` fires,
   - "Improving ranking…" while Phase 3 loads,
   - "Ready: best ranking" when `search_ready_best_ms` fires.
   `showModeBadge()` already exists; wire it to these three states. (Hook points are in
   place; copy/CSS change pending.)
2. **Web Worker for search.** Move `loadWordList` + scoring into a worker so the main
   thread only renders and handles input; worker returns partial (binary) results fast,
   then a rerank worker upgrades them. This is a larger refactor and is **not done**;
   the `ShardLoader` + `perf` split is the prerequisite groundwork.
3. **Blog/docs pages must not load models.** New static pages under `/blog/` and
   `/examples/` include no `app.js` / model fetch (verified: they are plain HTML).
4. **requestIdleCallback** for Phase 3 kickoff on capable browsers (currently fired after
   first render; move behind `requestIdleCallback` where available).

## Honesty note
Phases 0-3 and the cache-first shard loader are implemented and lint-clean. Lighthouse /
field perf numbers and the Web Worker refactor are **not run / not done**; steps to
produce them are above.
