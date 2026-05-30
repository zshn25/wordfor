# Maintenance

How to run, update, evaluate, and release WordFor. Operational companion to
[`README.md`](README.md) (overview) and [`AGENTS.md`](AGENTS.md) (codebase conventions).

> Detailed historical reports live in [`docs/archive/`](docs/archive/) and are referenced
> below where useful. They are point-in-time snapshots, not living docs.

## Repository layout

| Path | What | Tracked? |
|------|------|----------|
| `index.html`, `about.html`, `bench.html`, `blog/` | Static site (GitHub Pages serves repo root verbatim) | yes |
| `app.js`, `style.css`, `sw.js`, `manifest.json` | App shell, service worker, PWA manifest | yes |
| `data/*.json`, `data/*.bin` | Dictionary + quantized embeddings shipped to the browser | yes |
| `vendor/transformers.min.js` | Pinned Transformers.js | yes |
| `models/` | ONNX query encoder | yes |
| `tests/asset-smoke.mjs` | Static 404 guard (run before every deploy) | yes |
| `wordfor-mcp/` | MCP server (TypeScript) | yes |
| `core/` | Rust workspace (Tauri desktop, mobile, shared core) | yes |
| `build/` | **gitignored** — all parsers, audits, training, eval scripts | NO |
| `.github/workflows/deploy.yml` | Pages deploy on push to `main` | yes |
| `.github/workflows/build-desktop.yml` | Cross-platform Tauri CI matrix | yes |

**Important:** the entire `build/` directory is gitignored (`build/*`). Build scripts live
only on the maintainer's machine; only their *outputs* (`data/*.json`, `data/*.bin`) are
committed. `git ls-files build` returns nothing by design.

## Local dev

```sh
python -m http.server 1234     # serve the static site at http://localhost:1234
```

The app degrades gracefully: it fetches monolithic `data/*` files directly (no sharding,
no manifest). There is one asset loader path — direct `fetch(dataUrl(...))`.

## Updating the data (add/refresh a source)

All commands run from `build/` with the project venv
(`build/.venv/Scripts/python.exe` on Windows):

```sh
python build_dictionary.py                                   # 1. rebuild data/words.json
python finetune_potion.py --export --base-model ./potion-distilled-mxbai   # 2. embeddings (~50 min GPU)
python export_embeddings.py                                  # 3. full + quantized .bin files
python compare_eval.py --verbose > eval_<source>_after.txt   # 4. evaluate vs baseline
python audit_sources.py                                      # 5. audits (MUST pass before keep)
python audit_modern_sources.py
python verify_sources.py --check
```

A new source is only "done" when (a) it is declared in `build/sources.yaml`, (b) its parser
is wired into `build_dictionary.py`, (c) it appears in `data/source_manifest.json`, and
(d) the three audits pass. Declaring a source in `sources.yaml` alone does **not** ingest it.

License lanes (enforced by the audits):
- **Clean core** (public-domain / CC0 / CC BY): verbatim text may ship in `words.json`.
- **Build-time signals** (share-alike / proprietary, e.g. Wiktionary, ConceptNet): shape
  quality scores and *hidden* embeddings only — never shipped verbatim. The word may be
  shown without its restricted definition.
- See [`LICENSE_SOURCES.md`](LICENSE_SOURCES.md) for the authoritative per-source policy.

Outstanding source/eval work: [`docs/archive/remaining_source_work.md`](docs/archive/remaining_source_work.md),
[`docs/archive/remaining_eval_work.md`](docs/archive/remaining_eval_work.md).

## Updating the site

- Bump `sw.js` `CACHE_NAME` (e.g. `wordfor-v17` → `v18`) whenever the app shell changes.
- Bump `app.js` `DATA_VERSION` whenever `data/*` changes (busts the CDN/browser cache).
- Update `sitemap.xml` `lastmod`.
- Refresh `vendor/transformers.min.js` from the pinned jsDelivr URL when upgrading.
- Clear the Cloudflare cache after deploy.

## Validation (run before every commit/release)

```sh
node tests/asset-smoke.mjs                 # static 404 guard
cd wordfor-mcp && npm run build && npm run smoke   # MCP build + stdio smoke
cd build && python audit_sources.py && python audit_modern_sources.py && python verify_sources.py --check
```

Asset size: GitHub hard-rejects any file ≥ 100 MiB. Keep every `data/*` file under that
(largest today: `data/words.json` ~81 MiB). The desktop CI workflow re-checks this.

## Releasing

1. Run the full validation block above.
2. Commit in logical groups. **Author every commit as:**
   ```sh
   git commit --author="Zeeshan Khan Suri <zshn25@gmail.com>" -m "<message>"
   ```
3. `git push gh main` — `deploy.yml` publishes to wordfor.xyz; clear Cloudflare cache.
4. Tag releases `vX.Y.Z` and push the tag to trigger the desktop CI matrix.

Remotes: `gh` = GitHub (production, Pages + CI); `origin` = Hugging Face Space mirror.

## Desktop builds (Tauri)

- `core/.cargo/config.toml` is **gitignored** and machine-specific. It hardcodes the
  maintainer's local MinGW `gcc`/`ar` paths for the local `x86_64-pc-windows-gnu` default
  toolchain. It must never be committed: its `[env] CC`/`AR` apply to *all* targets and
  break the macOS/Windows CI runners (this caused the first CI matrix failures).
- CI uses the stock per-runner toolchains: `x86_64-pc-windows-msvc`, `aarch64/x86_64-apple-darwin`,
  `x86_64-unknown-linux-gnu`. Linux builds cleanly; macOS/Windows need their native
  C/C++ toolchains (provided by the runners) — do not inject cross paths via `.cargo/config.toml`.
- Local Windows (gnu default) build prerequisites and the historical toolchain analysis:
  [`docs/archive/rust_tauri_eval.md`](docs/archive/rust_tauri_eval.md).
- Per-platform artifact/verification status:
  [`docs/archive/platform_verification.md`](docs/archive/platform_verification.md).

## MCP server

Build, run, and publish status: [`docs/mcp.md`](docs/mcp.md) and
[`docs/archive/mcp_publish_checklist.md`](docs/archive/mcp_publish_checklist.md).
The server ships no model and does a lexical reverse lookup over `data/`.
