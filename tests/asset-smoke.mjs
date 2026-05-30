#!/usr/bin/env node
/**
 * asset-smoke.mjs -- static 404 guard for the deployed site.
 *
 * Scans every HTML page and app.js for locally-referenced runtime assets
 * (scripts, styles, icons, manifests, data/model files) and fails if any
 * REQUIRED one is missing from the repo. GitHub Pages serves the repo root
 * verbatim, so "missing on disk" == "404 in production".
 *
 * This is what would have caught the assets/model-manifest.json 404.
 *
 * Usage:  node tests/asset-smoke.mjs
 * Exit:   0 = all referenced assets present, 1 = missing asset(s).
 */
import { readFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// HTML pages that ship to Pages (the canonical app shell + content pages).
const HTML_PAGES = [
  "index.html",
  "about.html",
  "bench.html",
  "blog/index.html",
  "blog/how-wordfor-ranks-candidate-words.html",
  "blog/public-domain-dictionaries-without-copying-restricted-data.html",
  "blog/reverse-dictionary-how-to-find-a-word-from-a-meaning.html",
  "blog/tip-of-my-tongue-word-finder.html",
  "blog/word-for-a-feeling-you-cant-name.html",
];

// JS files whose fetch()/dataUrl() literals reference runtime assets.
const JS_FILES = ["app.js"];

// Assets that are legitimately optional (graceful fallbacks that may be absent
// in a given deployment). Missing these must NOT fail the smoke test.
const OPTIONAL = new Set([
  "data/embeddings_int8.bin", // int8 rerank fallback (int3 is the shipped tier)
  "data/embeddings_int4.bin", // removed: unused 105 MB full-rerank tier
]);

function isExternal(u) {
  return (
    !u ||
    u.startsWith("http://") ||
    u.startsWith("https://") ||
    u.startsWith("//") ||
    u.startsWith("data:") ||
    u.startsWith("mailto:") ||
    u.startsWith("#") ||
    u.startsWith("{") // template placeholder
  );
}

/** Resolve a URL referenced from `fromFile` to a repo-relative path, or null. */
function toRepoPath(url, fromFile) {
  if (isExternal(url)) return null;
  let clean = url.split("?")[0].split("#")[0].trim();
  if (!clean) return null;
  let abs;
  if (clean.startsWith("/")) {
    abs = join(ROOT, clean.replace(/^\/+/, ""));
  } else {
    abs = resolve(ROOT, dirname(fromFile), clean);
  }
  // Directory-style URL ("/") maps to index.html.
  if (clean.endsWith("/")) abs = join(abs, "index.html");
  return abs;
}

const refs = new Map(); // repoPath -> Set(referencedFrom)

function record(url, fromFile) {
  const abs = toRepoPath(url, fromFile);
  if (!abs) return;
  const rel = abs.slice(ROOT.length + 1).replace(/\\/g, "/");
  if (!refs.has(rel)) refs.set(rel, new Set());
  refs.get(rel).add(fromFile);
}

// --- scan HTML: src="...", href="..." (skip rel=canonical/alternate URLs which are absolute) ---
for (const page of HTML_PAGES) {
  const abs = join(ROOT, page);
  if (!existsSync(abs)) {
    console.error(`MISSING PAGE: ${page}`);
    process.exitCode = 1;
    continue;
  }
  const html = readFileSync(abs, "utf8");
  const attrRe = /\b(?:src|href)\s*=\s*"([^"]+)"/g;
  let m;
  while ((m = attrRe.exec(html)) !== null) record(m[1], page);
}

// --- scan JS: dataUrl("X") -> data/X, and fetch("Y") relative literals ---
for (const jsFile of JS_FILES) {
  const abs = join(ROOT, jsFile);
  if (!existsSync(abs)) continue;
  const js = readFileSync(abs, "utf8");
  let m;
  const dataUrlRe = /dataUrl\(\s*"([^"]+)"/g;
  while ((m = dataUrlRe.exec(js)) !== null) record(`data/${m[1]}`, jsFile);
  const fetchRe = /fetch\(\s*"([^"]+)"/g;
  while ((m = fetchRe.exec(js)) !== null) record(m[1], jsFile);
}

// --- verify existence ---
const missing = [];
for (const [rel, from] of [...refs].sort()) {
  if (OPTIONAL.has(rel)) continue;
  if (!existsSync(join(ROOT, rel))) {
    missing.push({ rel, from: [...from].join(", ") });
  }
}

console.log(`Checked ${refs.size} unique local asset references across ${HTML_PAGES.length} pages + ${JS_FILES.length} script(s).`);
if (missing.length) {
  console.error(`\nASSET 404 GUARD: FAIL -- ${missing.length} missing asset(s):`);
  for (const { rel, from } of missing) console.error(`  MISSING ${rel}   (referenced by: ${from})`);
  process.exit(1);
}
console.log("ASSET 404 GUARD: PASS (no missing local assets).");
