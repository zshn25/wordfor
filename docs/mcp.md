# Connecting WordFor to AI agents (MCP)

`wordfor-mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server that
exposes WordFor's reverse-dictionary engine to AI agents. It returns ranked candidate
words from a meaning, with explanations, lemma families, and license-safe source labels.

- **Transport (local):** stdio
- **Transport (remote):** Streamable HTTP at `/mcp`
- **Data:** public-domain / openly-licensed only; no restricted text; no outbound network.

## Build once

```bash
cd wordfor-mcp
npm install
npm run build      # -> dist/index.js
```

The server reads `../data/{words.json,source_manifest.json,forms_to_lemma.json}`.
Override the location with `WORDFOR_DATA_DIR=/abs/path/to/data`.

---

## 1. Claude Desktop (local stdio)

Edit `claude_desktop_config.json`:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "wordfor": {
      "command": "node",
      "args": ["C:/path/to/wordfor/wordfor-mcp/dist/index.js"],
      "env": { "WORDFOR_DATA_DIR": "C:/path/to/wordfor/data" }
    }
  }
}
```

Restart Claude Desktop. The `wordfor` tools (and the four prompts) appear in the tools menu.

> If you `npm link` (or globally install) the package, use `"command": "wordfor-mcp"` with no path.

---

## 2. Claude remote connector (Streamable HTTP)

Run the server in HTTP mode behind HTTPS (a reverse proxy such as Caddy/nginx, or a
tunnel for testing):

```bash
node dist/index.js --http --port 8787 --host 127.0.0.1
# MCP endpoint: http://127.0.0.1:8787/mcp
```

In Claude (web/desktop) **Settings -> Connectors -> Add custom connector**, point it at
your public `https://your-host/mcp` URL. The server is stateless per request except for
the MCP session id header (`mcp-session-id`), which the transport manages automatically.

Hardening for public exposure:
- Terminate TLS at your proxy; do not expose plain HTTP publicly.
- Keep the built-in rate limit (`WORDFOR_RATE_MAX`, `WORDFOR_RATE_WINDOW_MS`) or add one at the proxy.
- Optionally require an auth header at the proxy; the server itself ships no auth.

---

## 3. ChatGPT / OpenAI Apps SDK

OpenAI's Apps SDK and ChatGPT "developer mode" connectors speak MCP over Streamable HTTP,
the same endpoint as above (`https://your-host/mcp`). Add it as a custom MCP connector /
app in the relevant developer settings.

> **Status (honest):** there is no one-click public "store" listing for WordFor in ChatGPT.
> Distribution there currently requires either developer-mode custom connectors or going
> through OpenAI's app submission/review process. This repo provides the working MCP
> endpoint; it does not include a submitted/approved store listing. The optional Apps SDK
> *widget* component (a custom UI card) is not implemented yet — the server returns plain
> JSON text content that any MCP client can render.

---

## 4. Generic MCP client

Any MCP-compatible client can spawn the stdio binary:

```bash
node /path/to/wordfor-mcp/dist/index.js
```

or connect to the HTTP endpoint. Capabilities advertised: `tools`, `resources`, `prompts`.

Example (raw stdio JSON-RPC): `initialize` -> `notifications/initialized` ->
`tools/call { name: "reverse_lookup", arguments: { query: "the smell of rain on dry earth" } }`.

---

## Tools

| Tool | Args | Returns |
|------|------|---------|
| `reverse_lookup` | `query`, `limit=10`, `mode="fast"\|"best"` | ranked words + explanation + lemma_family + sources |
| `search_word` | `query`, `limit=10` | headword matches (exact/prefix/substring) |
| `explain_ranking` | `query`, `word` | scoring breakdown |
| `get_word_family` | `word` | inflectional family (no prefix/derivation collapse) |
| `get_sources` | `word` | license-safe source labels |
| `health` | – | status + entry count |

## Resources
`wordfor://about` · `wordfor://license-sources` · `wordfor://ranking-method` · `wordfor://examples`

## Prompts
`find-the-word` · `explain-why-it-fits` · `more-precise-alternatives` · `find-by-emotional-nuance`

---

## Ranking: what this server does vs the website

The website ranks semantically using on-device embeddings (binary first pass + int3
rerank). This server ships a **lexical** engine (token overlap + synonyms + quality prior
+ lemma dedupe); `mode: "best"` adds phrase-contiguity weighting. It needs no model
download and is fast and deterministic. A future semantic mode could load
`data/embeddings_*.bin` for parity; it is intentionally out of scope for the default
zero-dependency install.

## Security / privacy

- **No arbitrary file access.** Only the configured `WORDFOR_DATA_DIR` is read.
- **No hidden network.** The server makes no outbound requests.
- **Rate limiting** on HTTP mode (per remote address).
- **No query logging** unless explicitly opted in (`WORDFOR_LOG_QUERIES=1`).
- **No restricted text.** Source decoding emits only license-safe labels; sources marked
  non-redistributable are never surfaced.
