# Rust / Tauri desktop evaluation

Goal: audit the Tauri desktop build, attempt a local build, and compare the desktop app
against the web app. Nothing below is fabricated; commands that did not complete are marked
**NOT RUN** with the exact failure and the steps to reproduce a green build.

## Toolchain (verified on this machine)

| Tool | Version |
|------|---------|
| cargo | 1.95.0 (f2d3ce0bd 2026-03-21) |
| rustc | 1.95.0 (59807616e 2026-04-14) |
| default host | `x86_64-pc-windows-gnu` |
| installed toolchains | `stable-x86_64-pc-windows-gnu` (active), `stable-x86_64-pc-windows-msvc` |
| installed targets | `wasm32-unknown-unknown`, `x86_64-pc-windows-gnu`, `x86_64-pc-windows-msvc` |

## Crate audit

`core/wordfor-tauri/Cargo.toml`:
- Tauri **v2** (`tauri = "2"`, `tauri-build = "2"`, `tauri-plugin-shell = "2"`).
- Depends on `wordfor-core` (the shared reverse-dictionary engine) with `default-features = false`.

`core/wordfor-tauri/tauri.conf.json`:
- `productName: WordFor`, `version: 0.1.0`, `identifier: xyz.wordfor.desktop`.
- `frontendDist: ./web-dist` (a static copy of the web app — see **drift note** below).
- `bundle.targets: "all"`, icons from the repo PNGs.
- **Bundled resources:** `words.json`, `embeddings_binary.bin`, `embeddings_int3.bin`,
  `embeddings_int3_ranges.bin`, `embeddings_itq.bin`, `embeddings_ranges.bin`, `meta.json`.
  This is the *binary + int3-rerank* asset set — consistent with the web app's phased loader.
- CSP allows `connect-src` to `gc.zgo.at`, `*.huggingface.co`, `cdn.jsdelivr.net`.

## Build attempts (this machine)

### `wordfor-core` — PASS (verified)
```
cargo build -p wordfor-core
   Compiling tokenizers v0.21.4
   Compiling wordfor-core v0.1.0
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 51.86s
```
The shared engine compiles cleanly. This is the code that actually does reverse lookup, so
the core logic is confirmed building locally.

### `wordfor-tauri` — NOT RUN (environment blocker, not a code defect)
Two attempts, both blocked by a **mixed GNU/MinGW + MSVC toolchain** on this Windows box:

1. Default GNU toolchain:
   ```
   error: error calling dlltool 'dlltool.exe': program not found
   error: could not compile `parking_lot_core`
   ```
   The active `x86_64-pc-windows-gnu` toolchain has no MinGW `dlltool` installed.

2. MSVC toolchain (`cargo +stable-x86_64-pc-windows-msvc build`):
   ```
   error occurred in cc-rs: command did not execute successfully ...
   ...WinLibs...\mingw64\bin\ar.exe ... -out:...libvswhom.a -nologo ...
   ```
   A MinGW `ar.exe` (WinLibs) is on `PATH` and shadows the MSVC archiver, so `cc-rs` invokes
   the wrong tool for an MSVC target.

**This is purely a local environment problem.** The crate metadata is correct and
`wordfor-core` builds. To get a green desktop build, use a clean MSVC environment:

```powershell
# Use the MSVC toolchain and the Visual Studio "Developer PowerShell" (clean PATH,
# no MinGW ar.exe/dlltool shadowing the MSVC tools).
rustup default stable-x86_64-pc-windows-msvc
# In a VS Developer PowerShell:
cargo install tauri-cli --version "^2" --locked
cd core/wordfor-tauri
cargo tauri build
```
Requires: Visual Studio Build Tools (MSVC + Windows SDK) and WebView2 runtime (preinstalled
on Windows 11). CI (see below) does this on a clean `windows-latest` runner.

## Web vs desktop comparison

| Dimension | Web app | Tauri desktop |
|-----------|---------|---------------|
| Engine | JS/WASM in browser, phased loader (binary -> int3 rerank) | `wordfor-core` (native Rust) |
| Assets | fetched/sharded from `data/` (see asset_hosting_strategy.md) | bundled as Tauri `resources` (binary + int3 set) |
| First search ready | Phase 1 word list, then binary model | bundled assets read from disk (no network) |
| Network | model/data fetch + analytics | none required for search (assets local) |
| Distribution | static site | signed installers per OS |
| Startup / memory | **NOT MEASURED** (needs a desktop run) | **NOT MEASURED** (build did not complete locally) |

Runtime numbers (startup, search-ready, RSS) are **NOT MEASURED** because the desktop bundle
did not build on this machine. Reproduce with the clean-MSVC steps above, then:
- startup: wall-clock from launch to window-interactive,
- search-ready: time to first ranked result for a fixed query,
- memory: peak RSS from Task Manager / `Get-Process WordFor`.

## Drift note (action item, not done here)
`core/wordfor-tauri/web-dist/app.js` is a **separate older copy** of the web `app.js`. The
ShardLoader + perf instrumentation changes made to the root `app.js` are **not** mirrored
there. Before shipping the desktop app, sync `web-dist/` from the repo root (or point
`frontendDist` at a shared build output) so desktop and web stay in lockstep. Desktop reads
assets from bundled `resources`, so the ShardLoader's network path is not required there, but
the perf/status UI changes still belong in the desktop copy.

## Summary
- Tauri v2 config and resource bundling audited — **correct and consistent** with the web loader.
- `wordfor-core` builds locally — **PASS**.
- Desktop bundle — **NOT RUN locally** due to a mixed MinGW/MSVC PATH; reproducible green-build
  steps provided; CI matrix builds it on clean runners.
- Runtime comparison metrics — **NOT MEASURED** (pending a successful desktop build).
