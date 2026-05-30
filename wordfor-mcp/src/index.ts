#!/usr/bin/env node
/**
 * wordfor-mcp -- Model Context Protocol server for WordFor.
 *
 * Exposes WordFor's reverse-dictionary / word-finder capabilities to AI agents as MCP
 * tools, resources, and prompts. Public-domain sources only; no restricted text; no
 * hidden network calls; no arbitrary file access (reads only the WordFor data dir).
 *
 * Transports:
 *   - stdio (default)            : for local clients (Claude Desktop, etc.)
 *   - Streamable HTTP (--http)   : for remote clients (Claude custom connectors, etc.)
 *
 * Usage:
 *   wordfor-mcp                  # stdio
 *   wordfor-mcp --http [--port 8787] [--host 127.0.0.1]
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import { createServer } from "node:http";
import { randomUUID } from "node:crypto";

import { getData } from "./data.js";
import {
  reverseLookup,
  searchWord,
  explainRanking,
  getWordFamily,
  getSources,
} from "./search.js";

const VERSION = "0.1.0";

function buildServer(): McpServer {
  const server = new McpServer(
    { name: "wordfor", version: VERSION },
    {
      instructions:
        "WordFor finds words from meanings (a reverse dictionary). Use reverse_lookup to " +
        "turn a description into ranked candidate words; search_word for spelling/prefix " +
        "lookups; explain_ranking, get_word_family, and get_sources for provenance. All " +
        "data is public-domain / openly licensed; no restricted text is returned.",
    }
  );

  const limit = z.number().int().min(1).max(50).default(10);

  server.registerTool(
    "reverse_lookup",
    {
      title: "Reverse dictionary lookup",
      description:
        "Find the word from a meaning. Give a description/definition and get ranked " +
        "candidate words with explanations, lemma family, and license-safe source labels.",
      inputSchema: {
        query: z.string().min(1).describe("A description or definition of the word you want."),
        limit,
        mode: z.enum(["fast", "best"]).default("fast").describe("fast = lexical; best = lexical + phrase weighting."),
      },
    },
    async ({ query, limit, mode }) => {
      const results = reverseLookup(query, limit, mode);
      return { content: [{ type: "text", text: JSON.stringify({ query, mode, results }, null, 2) }] };
    }
  );

  server.registerTool(
    "search_word",
    {
      title: "Word finder (by spelling)",
      description: "Find headwords by exact match, prefix, or substring (a spelling-based word finder).",
      inputSchema: { query: z.string().min(1), limit },
    },
    async ({ query, limit }) => {
      const results = searchWord(query, limit);
      return { content: [{ type: "text", text: JSON.stringify({ query, results }, null, 2) }] };
    }
  );

  server.registerTool(
    "explain_ranking",
    {
      title: "Explain ranking",
      description: "Explain why a given word fits a query: lexical overlap, POS, quality prior, lemma, and sources.",
      inputSchema: { query: z.string().min(1), word: z.string().min(1) },
    },
    async ({ query, word }) => {
      return { content: [{ type: "text", text: JSON.stringify(explainRanking(query, word), null, 2) }] };
    }
  );

  server.registerTool(
    "get_word_family",
    {
      title: "Get word family (lemma)",
      description: "Return the inflectional family for a word (run/running/ran). Prefixes and derivations are NOT collapsed.",
      inputSchema: { word: z.string().min(1) },
    },
    async ({ word }) => {
      return { content: [{ type: "text", text: JSON.stringify(getWordFamily(word), null, 2) }] };
    }
  );

  server.registerTool(
    "get_sources",
    {
      title: "Get sources for a word",
      description: "Return the license-safe source labels (public-domain / open) that define a word.",
      inputSchema: { word: z.string().min(1) },
    },
    async ({ word }) => {
      return { content: [{ type: "text", text: JSON.stringify(getSources(word), null, 2) }] };
    }
  );

  server.registerTool(
    "health",
    {
      title: "Health check",
      description: "Report server status and dataset size.",
      inputSchema: {},
    },
    async () => {
      const data = getData();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              { status: "ok", version: VERSION, entries: data.entries.length, sources: Object.keys(data.bitIndex).length },
              null,
              2
            ),
          },
        ],
      };
    }
  );

  // ---- Resources --------------------------------------------------------
  server.registerResource(
    "about",
    "wordfor://about",
    { title: "About WordFor", description: "What WordFor is and how it works.", mimeType: "text/markdown" },
    async (uri) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: "text/markdown",
          text:
            "# WordFor\nA privacy-first reverse dictionary: describe a concept, get the word. " +
            "Search runs on-device in the web app; this MCP server provides a lexical engine over " +
            "the same public-domain word data. https://wordfor.xyz",
        },
      ],
    })
  );

  server.registerResource(
    "license-sources",
    "wordfor://license-sources",
    { title: "License & sources", description: "Source list with license classes.", mimeType: "application/json" },
    async (uri) => {
      const data = getData();
      const sources = Object.entries(data.sourceMeta).map(([id, m]) => ({
        id,
        name: m.name,
        license: m.license,
        redistributable: m.redistribute_text !== false,
        visible: m.default_visible === true,
      }));
      return { contents: [{ uri: uri.href, mimeType: "application/json", text: JSON.stringify(sources, null, 2) }] };
    }
  );

  server.registerResource(
    "ranking-method",
    "wordfor://ranking-method",
    { title: "Ranking method", description: "How WordFor ranks candidates.", mimeType: "text/markdown" },
    async (uri) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: "text/markdown",
          text:
            "# Ranking\n1. Lexical match\n2. Semantic similarity (on-device embeddings in the web app)\n" +
            "3. Source confidence\n4. Lemma family grouping (inflections only)\n5. Reranking (binary first pass, int3 rerank).\n" +
            "This MCP server implements 1, 3, 4 directly; 2 and 5 are described for parity with the web app.",
        },
      ],
    })
  );

  server.registerResource(
    "examples",
    "wordfor://examples",
    { title: "Example queries", description: "Sample reverse-dictionary queries.", mimeType: "application/json" },
    async (uri) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: "application/json",
          text: JSON.stringify(
            [
              "the smell of rain on dry earth",
              "lasting for a very short time",
              "nostalgia for a time you never knew",
              "using more words than needed",
              "happy at another's misfortune",
            ],
            null,
            2
          ),
        },
      ],
    })
  );

  // ---- Prompts ----------------------------------------------------------
  server.registerPrompt(
    "find-the-word",
    {
      title: "Find the word I'm thinking of",
      description: "Turn a description into the best candidate words.",
      argsSchema: { description: z.string().describe("Describe the word you're trying to remember.") },
    },
    ({ description }) => ({
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `I'm trying to recall a word. It means: "${description}". Use the reverse_lookup tool, then suggest the 3 best candidates with short reasons.`,
          },
        },
      ],
    })
  );

  server.registerPrompt(
    "explain-why-it-fits",
    {
      title: "Explain why this word fits",
      description: "Explain how well a word matches a meaning.",
      argsSchema: { query: z.string(), word: z.string() },
    },
    ({ query, word }) => ({
      messages: [
        {
          role: "user",
          content: { type: "text", text: `Use explain_ranking for query "${query}" and word "${word}", then summarise why it does or doesn't fit.` },
        },
      ],
    })
  );

  server.registerPrompt(
    "more-precise-alternatives",
    {
      title: "Suggest more precise alternatives",
      description: "Find sharper words than a given one.",
      argsSchema: { word: z.string(), context: z.string().optional() },
    },
    ({ word, context }) => ({
      messages: [
        {
          role: "user",
          content: { type: "text", text: `I'm using the word "${word}"${context ? ` in this context: ${context}` : ""}. Use reverse_lookup to find more precise alternatives and rank them.` },
        },
      ],
    })
  );

  server.registerPrompt(
    "find-by-emotional-nuance",
    {
      title: "Find words by emotional nuance",
      description: "Find words for a subtle feeling.",
      argsSchema: { feeling: z.string().describe("Describe the feeling/nuance.") },
    },
    ({ feeling }) => ({
      messages: [
        {
          role: "user",
          content: { type: "text", text: `Use reverse_lookup (mode "best") to find words for this feeling: "${feeling}". Return 5 nuanced options with one-line distinctions.` },
        },
      ],
    })
  );

  return server;
}

// --------------------------------------------------------------------------
// Transports
// --------------------------------------------------------------------------

async function runStdio() {
  getData(); // warm load before connecting (fail fast on missing data)
  const server = buildServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  // stdio: do not write to stdout (reserved for protocol). Logs go to stderr.
  process.stderr.write(`wordfor-mcp ${VERSION} ready (stdio)\n`);
}

async function runHttp(port: number, host: string) {
  getData();

  // Minimal in-memory rate limiter (per remote address): RATE_MAX requests / RATE_WINDOW ms.
  const RATE_MAX = Number(process.env.WORDFOR_RATE_MAX ?? 120);
  const RATE_WINDOW = Number(process.env.WORDFOR_RATE_WINDOW_MS ?? 60_000);
  const hits = new Map<string, { n: number; reset: number }>();
  function rateLimited(ip: string): boolean {
    const now = Date.now();
    const cur = hits.get(ip);
    if (!cur || now > cur.reset) {
      hits.set(ip, { n: 1, reset: now + RATE_WINDOW });
      return false;
    }
    cur.n++;
    return cur.n > RATE_MAX;
  }

  const transports = new Map<string, StreamableHTTPServerTransport>();

  const httpServer = createServer(async (req, res) => {
    const ip = req.socket.remoteAddress || "unknown";
    if (rateLimited(ip)) {
      res.writeHead(429, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "rate limited" }));
      return;
    }
    if (req.url !== "/mcp") {
      res.writeHead(404).end();
      return;
    }

    const sessionId = req.headers["mcp-session-id"] as string | undefined;
    let transport = sessionId ? transports.get(sessionId) : undefined;

    if (!transport) {
      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: (sid) => {
          transports.set(sid, transport!);
        },
      });
      transport.onclose = () => {
        if (transport!.sessionId) transports.delete(transport!.sessionId);
      };
      const server = buildServer();
      await server.connect(transport);
    }

    // Collect body for POST
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", async () => {
      let parsed: unknown = undefined;
      if (body) {
        try {
          parsed = JSON.parse(body);
        } catch {
          res.writeHead(400).end();
          return;
        }
      }
      await transport!.handleRequest(req, res, parsed);
    });
  });

  httpServer.listen(port, host, () => {
    process.stderr.write(`wordfor-mcp ${VERSION} ready (http) at http://${host}:${port}/mcp\n`);
  });
}

function main() {
  const args = process.argv.slice(2);
  if (args.includes("--http")) {
    const portIdx = args.indexOf("--port");
    const hostIdx = args.indexOf("--host");
    const port = portIdx >= 0 ? Number(args[portIdx + 1]) : Number(process.env.PORT ?? 8787);
    const host = hostIdx >= 0 ? args[hostIdx + 1] : process.env.HOST ?? "127.0.0.1";
    runHttp(port, host).catch((e) => {
      process.stderr.write(`fatal: ${e}\n`);
      process.exit(1);
    });
  } else {
    runStdio().catch((e) => {
      process.stderr.write(`fatal: ${e}\n`);
      process.exit(1);
    });
  }
}

main();
