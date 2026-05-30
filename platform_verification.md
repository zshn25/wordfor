# Platform verification

Tracks which platform artifacts have actually been produced and verified, and which are
**CI-only / NOT RUN locally**. Nothing here claims a build that did not happen.

## Build matrix (`.github/workflows/build-desktop.yml`)

| Platform | Runner | Rust target | Local status | CI status |
|----------|--------|-------------|--------------|-----------|
| macOS x64 | `macos-13` | `x86_64-apple-darwin` | NOT RUN (no Mac) | configured, not yet executed |
| macOS ARM64 | `macos-14` | `aarch64-apple-darwin` | NOT RUN (no Mac) | configured, not yet executed |
| Windows x64 | `windows-latest` | `x86_64-pc-windows-msvc` | NOT RUN (local toolchain blocker, see rust_tauri_eval.md) | configured, not yet executed |
| Linux x64 | `ubuntu-22.04` | `x86_64-unknown-linux-gnu` | NOT RUN (no Linux) | configured, not yet executed |

The workflow has not been pushed/run yet (the user controls commits and push). "Configured"
means the YAML exists and is ready; **no artifacts have been produced or downloaded**, so no
runtime verification has occurred. Do not report these as verified until a CI run is green and
the artifacts are downloaded and launched.

The workflow triggers on `workflow_dispatch`, `push` of a `v*` tag, and `pull_request`
touching `core/**`. It **cannot run until it is on the remote default branch**; pushing the
planned `v*` tag (see commit plan) triggers the full matrix.

## Per-target actual status

### macOS x64 (`macos-13`, `x86_64-apple-darwin`)
- CI build PASS/FAIL: **PENDING** (workflow not yet on remote)
- Artifact name / size: **UNKNOWN** (expected `WordFor_x64.dmg` / `.app`)
- Runtime launch manually verified: **NOT RUN** (no Mac hardware, no artifact)
- Search / model loading verified: **NOT RUN**

### macOS Apple Silicon (`macos-14`, `aarch64-apple-darwin`)
- CI build PASS/FAIL: **PENDING**
- Artifact name / size: **UNKNOWN** (expected `WordFor_aarch64.dmg` / `.app`)
- Runtime launch manually verified: **NOT RUN** (no Mac hardware)
- Search / model loading verified: **NOT RUN**

### Windows x64 (`windows-latest`, `x86_64-pc-windows-msvc`)
- CI build PASS/FAIL: **PENDING**
- Local build: **NOT RUN** — mixed MinGW/MSVC PATH blocks the bundle locally; `wordfor-core`
  itself builds (51.86s). See `rust_tauri_eval.md`.
- Artifact name / size: **UNKNOWN** (expected `WordFor_x64-setup.exe` / `.msi`)
- Runtime launch manually verified: **NOT RUN**
- Search / model loading verified: **NOT RUN**

### Linux x64 (`ubuntu-22.04`, `x86_64-unknown-linux-gnu`)
- CI build PASS/FAIL: **PENDING**
- Artifact name / size: **UNKNOWN** (expected `.AppImage` / `.deb`)
- Runtime launch manually verified: **NOT RUN** (no Linux host)
- Search / model loading verified: **NOT RUN**

## What IS verified locally
- `wordfor-core` compiles on Windows (`cargo build -p wordfor-core` -> Finished in 51.86s).
- Asset size guard runs and correctly flags oversize tracked blobs
  (`python build/check_asset_sizes.py --git`).

## Artifact verification checklist (per platform, after a green CI run)
- [ ] Download the `wordfor-<platform>` artifact bundle.
- [ ] Installer opens / app launches without a security block (beyond expected unsigned warnings).
- [ ] App window renders the WordFor UI.
- [ ] A reverse-lookup query returns ranked results (assets load from bundled resources).
- [ ] No crash on quit; no orphaned process.
- [ ] Record bundle size and cold-start time.

## Windows verification from a Mac (no Windows hardware)
If you only have a Mac and need to validate the Windows build:
1. Build Windows artifacts in CI (the matrix above) — do **not** cross-compile Windows from macOS for Tauri.
2. Verify the `.msi` / `-setup.exe` in a Windows VM:
   - Parallels / UTM (ARM Windows 11) for Apple Silicon, or a cloud Windows VM (e.g. a CI
     "windows-latest" interactive session, Azure/AWS Windows instance).
3. Run the artifact checklist above inside the VM.
4. For signature/SmartScreen checks, the VM must have internet (SmartScreen reputation) — an
   unsigned build will show a SmartScreen warning; that is expected until signing is set up.

## Signing & notarization checklist (NOT configured — secrets required)

### macOS
- [ ] Apple Developer ID Application certificate in the CI keychain (base64 secret).
- [ ] `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY` secrets.
- [ ] `codesign --deep --force --options runtime` (Tauri does this when signing identity is set).
- [ ] Notarize: `APPLE_ID`, `APPLE_PASSWORD` (app-specific), `APPLE_TEAM_ID`; `xcrun notarytool submit --wait`.
- [ ] Staple: `xcrun stapler staple WordFor.app`.

### Windows
- [ ] Code-signing certificate (OV/EV). EV gives instant SmartScreen reputation.
- [ ] `signtool sign /fd sha256 /tr <timestamp-url> /td sha256 ...` (or Tauri's `windows.certificateThumbprint`).
- [ ] Verify: `signtool verify /pa WordFor_x64-setup.exe`.

### Linux
- [ ] No OS-level signing required; optionally GPG-sign the `.deb`/`.rpm` and publish a key.

None of the above secrets are present in this repo, so **signing/notarization is NOT done**.
The CI workflow builds **unsigned** artifacts; add a separate signed-release workflow with the
secrets above before public distribution.

## Summary
- CI matrix is defined for all four targets + an asset-size guard gate.
- Local: only `wordfor-core` build and the asset guard are verified.
- All desktop artifacts and their runtime/signing verification are **NOT RUN** — reproducible
  steps and checklists are provided above.
