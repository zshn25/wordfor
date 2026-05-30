// Smoke test for wordfor-mcp: spawn the stdio server, initialize, list tools,
// call reverse_lookup + health. Exits non-zero on failure. Run: node test/smoke.mjs
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const child = spawn(process.execPath, [join(root, "dist", "index.js")], {
  stdio: ["pipe", "pipe", "inherit"],
});
let buf = "";
const send = (o) => child.stdin.write(JSON.stringify(o) + "\n");
const fail = (m) => { console.error("SMOKE FAIL:", m); child.kill(); process.exit(1); };

child.stdout.on("data", (d) => {
  buf += d.toString();
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i).trim();
    buf = buf.slice(i + 1);
    if (!line) continue;
    const msg = JSON.parse(line);
    if (msg.id === 1) {
      send({ jsonrpc: "2.0", method: "notifications/initialized" });
      send({ jsonrpc: "2.0", id: 2, method: "tools/list" });
    } else if (msg.id === 2) {
      const names = msg.result.tools.map((t) => t.name);
      console.log("TOOLS:", names.join(", "));
      for (const t of ["reverse_lookup", "search_word", "explain_ranking", "get_word_family", "get_sources", "health"])
        if (!names.includes(t)) return fail(`missing tool ${t}`);
      send({ jsonrpc: "2.0", id: 3, method: "tools/call", params: { name: "reverse_lookup", arguments: { query: "lasting for a very short time", limit: 5, mode: "best" } } });
    } else if (msg.id === 3) {
      const parsed = JSON.parse(msg.result.content[0].text);
      if (!parsed.results || parsed.results.length === 0) return fail("reverse_lookup returned no results");
      console.log("RESULTS:", parsed.results.map((r) => `${r.word} (${r.score})`).join(", "));
      send({ jsonrpc: "2.0", id: 4, method: "tools/call", params: { name: "health", arguments: {} } });
    } else if (msg.id === 4) {
      const h = JSON.parse(msg.result.content[0].text);
      if (h.status !== "ok") return fail("health not ok");
      console.log("HEALTH:", JSON.stringify(h));
      console.log("SMOKE PASS");
      child.kill();
      process.exit(0);
    }
  }
});

send({ jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "smoke", version: "0" } } });
setTimeout(() => fail("timeout"), 60000);
