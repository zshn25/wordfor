# wordfor-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server that gives AI agents
WordFor's **reverse dictionary** / word-finder capabilities. Describe a meaning, get the
word. Public-domain sources only; no restricted text; no hidden network calls; reads only
the WordFor data directory.

## Tools

| Tool | Purpose |
|------|---------|
| `reverse_lookup(query, limit=10, mode="fast"\|"best")` | Find words from a meaning (ranked). |
| `search_word(query, limit=10)` | Find headwords by exact/prefix/substring spelling. |
| `explain_ranking(query, word)` | Why a word fits: lexical overlap, POS, quality, lemma, sources. |
| `get_word_family(word)` | Inflectional family (run/running/ran); prefixes/derivations NOT collapsed. |
| `get_sources(word)` | License-safe source labels for a word. |
| `health()` | Status + dataset size. |

Each result includes: `word`, `score`, `explanation`, `lemma_family`, and `sources`
(name + license class only — never restricted copied text).

## Resources
`wordfor://about` · `wordfor://license-sources` · `wordfor://ranking-method` · `wordfor://examples`

## Prompts
"Find the word I'm thinking of" · "Explain why this word fits" · "Suggest more precise
alternatives" · "Find words by emotional nuance".

## Install & build

```bash
cd wordfor-mcp
npm install
npm run build
```

The server reads `../data/words.json`, `../data/source_manifest.json`, and
`../data/forms_to_lemma.json` by default. Override with `WORDFOR_DATA_DIR`.

## Run

```bash
# Local stdio (Claude Desktop, generic MCP clients)
node dist/index.js
# or after npm link / global install:
wordfor-mcp

# Remote Streamable HTTP (Claude custom connectors, etc.)
node dist/index.js --http --port 8787 --host 127.0.0.1
# endpoint: http://127.0.0.1:8787/mcp
```

## Notes on ranking
The website's *semantic* ranking uses on-device embeddings (mdbr-leaf-mt + quantized
vectors). This server ships a strong **lexical** engine (token overlap + synonym + quality
+ lemma dedupe) that needs no model download. `mode: "best"` adds phrase weighting.
Optional semantic mode (loading the embeddings) is documented in `docs/mcp.md`.

## Security / privacy
- No arbitrary file access — only the configured data dir is read.
- No outbound network calls.
- HTTP mode is rate-limited (`WORDFOR_RATE_MAX` / `WORDFOR_RATE_WINDOW_MS`).
- Queries are not logged unless you opt in (`WORDFOR_LOG_QUERIES=1`, not enabled by default).

See `docs/mcp.md` for client setup (Claude Desktop, Claude remote connector, ChatGPT /
OpenAI Apps SDK, generic clients) and `mcp_publish_checklist.md` for publishing.
