# MCP publish checklist (`wordfor-mcp`)

Status legend: [ ] = not done, [x] = done in repo, [N/A] = requires maintainer account/network.

## Current status (direct answers)

| Question | Answer |
|----------|--------|
| npm package prepared? | **Yes** — `package.json`/`tsconfig.json`/`src` complete; builds; stdio smoke test passes. Not yet `npm publish`ed. |
| package name available? | **Yes** — `npm view wordfor-mcp` → 404 (name is free, not taken). |
| GitHub release prepared? | **No** — no tag/release created; release notes drafted in this checklist only. |
| Claude connector metadata prepared? | **Yes (docs)** — Claude Desktop config snippet + remote custom-connector steps in `docs/mcp.md` §1–§2. No hosted endpoint deployed. |
| OpenAI Apps SDK demo runnable? | **No** — server speaks MCP over Streamable HTTP (works as a custom connector), but no Apps SDK UI widget and no store demo is built. Documented honestly in `docs/mcp.md` §3. |
| remote HTTP deployment tested? | **No** — HTTP transport implemented and code-complete; only the **stdio** transport was smoke-tested. No public/remote deployment has been stood up or tested. |

**Not published.** Nothing in this repo has been published to npm, the MCP registry, or any store.

## Build & verify
- [x] `npm install` succeeds (95 packages, 0 vulnerabilities).
- [x] `npm run build` compiles with no TypeScript errors.
- [x] stdio smoke test passes: `initialize` -> `tools/list` -> `reverse_lookup` returns ranked words.
  - Verified: `reverse_lookup("lasting for a very short time", best)` -> ephemeral, momentary, barely, trice.
  - Verified: `health` -> 576,405 entries, 18 sources.
- [ ] HTTP smoke test (`--http`) against a real MCP client (Claude custom connector / MCP Inspector).
- [ ] `npx @modelcontextprotocol/inspector node dist/index.js` manual pass (tools/resources/prompts visible).

## Package metadata (package.json)
- [x] `name`, `version`, `bin`, `type: module`, `exports`/`main` set.
- [x] Package name `wordfor-mcp` confirmed available on npm (404).
- [ ] Add `description`, `keywords` (mcp, reverse-dictionary, wordfor), `license`, `repository`, `homepage`, `author`.
- [ ] Add `files` allowlist (`dist`, `README.md`) so `data/` is NOT bundled into the npm tarball.
- [ ] Decide data strategy for npm consumers: require `WORDFOR_DATA_DIR`, or ship a fetch step. (Do NOT publish 80 MB words.json to npm.)
- [ ] `npm pack --dry-run` and confirm tarball contents + size.

## npm publish [N/A — needs npm account]
- [ ] `npm login`.
- [ ] `npm publish --access public`.
- [ ] Verify install: `npx wordfor-mcp` in a clean dir.

## GitHub release [N/A — needs push access]
- [ ] Tag `wordfor-mcp-v0.1.0`.
- [ ] Release notes: tools, transports, security model.

## MCP Registry / directories [N/A — needs submission]
- [ ] Submit to the official MCP server registry (server name, description, install command, transport).
- [ ] Provide `mcp.json` / server manifest if required by the registry at submission time.
- [ ] Ensure README has: what it does, install, config, security, example calls. (README present.)

## Client distribution
- [ ] Claude Desktop config snippet documented (docs/mcp.md §1). [x] documented
- [ ] Claude remote connector steps documented (docs/mcp.md §2). [x] documented
- [ ] ChatGPT / OpenAI Apps SDK status documented honestly (no store listing yet). [x] documented

## Security review
- [x] No arbitrary file access (reads only `WORDFOR_DATA_DIR`).
- [x] No outbound network calls.
- [x] HTTP rate limiting present.
- [x] Query logging off by default.
- [x] Only license-safe source labels emitted; non-redistributable sources never surfaced.
- [ ] Add a SECURITY.md / disclosure contact before public listing.

## Not done / explicitly deferred
- OpenAI Apps SDK custom UI widget (server returns plain JSON content for now).
- Semantic ranking mode (loads embeddings) — lexical engine only by default.
- Public ChatGPT/OpenAI store submission (requires review process + account).
