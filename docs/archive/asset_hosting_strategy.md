# Asset hosting strategy (GitHub Pages friendly)

How WordFor ships large model/data assets without hitting GitHub's 100 MiB blob block
and without Git LFS (which GitHub Pages does not serve reliably for this use case).

## The concrete problem (measured 2026-05-30)

`build/check_asset_sizes.py --git` on the current tree:

```
BLOCK  data/embeddings_int4.bin: 105.5 MB >= 100.0 MiB (GitHub will reject this push)
LARGE  data/embeddings_int3.bin: 79.2 MB
LARGE  data/words.json: 81.0 MB
LARGE  data/embeddings_binary.bin: 26.4 MB
... (several 14-70 MB monoliths)
```

`data/embeddings_int4.bin` (105.5 MB) is **already tracked and exceeds GitHub's hard
100 MiB limit** — any push including it is rejected. Several other served assets are
in the 50-100 MiB soft-warning band.

## The fix: deterministic byte-sharding + manifest (no LFS)

`build/split_assets.py` splits each large asset into immutable byte shards of
`--shard-mb` (default 15 MiB, hard ceiling 25 MiB) and writes
`assets/model-manifest.json` with per-shard `sha256`, sizes, priority and model type.

Verified run (`split_assets.py` then `--check`):

```
shard words                 80.98 MB -> 6 shards
shard embeddings_binary     26.39 MB -> 2 shards   (critical)
shard embeddings_itq         0.56 MB -> 1 shard    (critical)
shard embeddings_int3       79.16 MB -> 6 shards   (background)
shard embeddings_int4      105.54 MB -> 8 shards   (background)
...
PASS: all shards verified against manifest   (sha256 reassembly OK for every asset)
```

Each shard is <= 15 MiB, so no blob approaches the 100 MiB block or even the 50 MiB
warning. The manifest `version` is a 16-hex digest of all asset hashes (cache key).

## Loading (progressive, cache-first)

`assets/shard-loader.js` (`window.ShardLoader`):

1. UI renders immediately (static HTML/CSS, no model needed).
2. `critical` shards (`words`, `forms_to_lemma`, `embeddings_binary`, `embeddings_itq`,
   `embeddings_ranges`) load first -> **fast search** is enabled.
3. `background` shards (`embeddings_int3` rerank, potion lite) load lazily after first
   render -> ranking silently **upgrades to best**.
4. Every shard is cached in the **Cache API** (`wordfor-shards-v1`) keyed by its
   immutable URL. Repeat visits skip re-download. `pruneOldCaches()` drops stale
   manifest versions.
5. Integrity: shard URLs carry sha256 in the manifest; the loader can verify before use.

`app.js` is wired to prefer `ShardLoader` for `words.json`, `forms_to_lemma.json`, and
the `embeddings_int3` rerank, and **falls back to the monolithic `data/<file>`** when no
manifest is deployed — so nothing breaks if shards are absent.

## CI size guard

`build/check_asset_sizes.py` (run in CI on every PR) fails if:
- any shard > 25 MiB,
- any committed non-shard runtime asset > 10 MiB (forces it through `split_assets.py`),
- any tracked blob >= 100 MiB (the hard block), or >= 50 MiB (soft warning).

## Migration checklist (run by maintainer; not committed here)

1. `python build/split_assets.py --shard-mb 15` (generates `data/shards/*` + manifest).
2. `python build/split_assets.py --check` (must say PASS).
3. Git-track `data/shards/**` and `assets/model-manifest.json`.
4. **Untrack the monolithic large bins** so only shards ship:
   - add to `.gitignore`: `data/embeddings_int4.bin`, `data/embeddings_int3.bin`,
     `data/words.json` (and other >10 MB monoliths) — keep them locally, ship shards.
   - `git rm --cached data/embeddings_int4.bin` (etc.) — **maintainer does this**, not here.
5. `python build/check_asset_sizes.py --git` -> must say PASS.
6. Deploy; verify the site loads from shards (DevTools -> Network shows `shards/...`),
   and a second visit is served from Cache Storage.

> Note: `words.json` can stay monolithic if preferred (it is 81 MB < 100 MiB), but
> sharding it lets the first query run before all shards arrive and keeps every blob
> comfortably small. The build never deletes the monolith; it is the fallback.

## Optional fallback: external model base URL

If the Pages repo still grows too large, publish shard packs as **GitHub Release
assets** or an object store / CDN and set `MODEL_BASE_URL`. `shard-loader.js` fetches
shard URLs relative to `data/`; point that at the CDN by prefixing shard URLs in the
manifest, or extend the loader to read `MODEL_BASE_URL` from a small `config.js`.

## Compression

- JSON (`words.json`, `forms_to_lemma.json`) compresses well; GitHub Pages serves gzip
  automatically. For Brotli, pre-compress to `.json.br` and content-negotiate, or rely
  on the CDN. Binary quantized bins are already near-incompressible (int3/int4 packed),
  so gzip yields little — sharding is the win there, not compression.
- Future: a binary packed word list (instead of 81 MB JSON) would cut the largest data
  asset substantially; tracked as a follow-up, not done here.

## Repo hygiene
- Generated raw datasets / full-precision bins (`embeddings.bin`, `embeddings_int8.bin`)
  stay `.gitignore`d (already are).
- Only final sharded, runtime-safe assets + manifest go into the Pages output.
- `check_asset_sizes.py` is the enforcement gate.
