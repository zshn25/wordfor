/**
 * WordFor — Reverse Dictionary
 * © 2025 Zeeshan Khan Suri (zshn25). Licensed under CC-BY-NC-ND-4.0.
 *
 * Two modes (auto-detected):
 *   Full (desktop):  mdbr-leaf-mt (query) + mxbai-embed-large (defs) via Transformers.js
 *   Lite (mobile):   potion-base-8M via pure JS static embeddings (sub-1ms)
 *
 * Override with ?mode=full or ?mode=lite in the URL.
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DATA_ROOT      = "data";
const TOP_K          = 30;
const DEBOUNCE       = 400;
const RATE_LIMIT_MAX = 15;
const RATE_LIMIT_MS  = 10_000;

const W_COSINE       = 0.70;
const W_KEYWORD      = 0.30;

const FULL_MODEL_ID  = "onnx-community/mdbr-leaf-mt-ONNX";
const FULL_DIMS      = 384;
const LITE_DIMS      = 256;

let MODE = null;
let DIMS = null;
let fullReady = false;

// Stop words for keyword matching
const STOP_WORDS = new Set([
  "a","an","the","is","are","was","were","be","been","being","of","in","to",
  "for","on","with","at","by","from","that","this","it","as","or","and","not",
  "who","which","what","where","when","how","if","but","than","so","very",
  "can","do","does","did","has","have","had","will","would","shall","should",
  "may","might","could","about","into","through","during","before","after",
  "above","below","between","under","again","further","then","once","all",
  "each","every","both","few","more","most","other","some","such","no","nor",
  "only","own","same","too","just","also","now","much","many","like","one",
  "two","person","thing","something","someone","people",
]);

// ---------------------------------------------------------------------------
// Float-16 → Float-32 lookup table  (65 536 entries ≈ 256 KB)
// ---------------------------------------------------------------------------

const f16LUT = new Float32Array(65536);
(function buildLUT() {
  for (let i = 0; i < 65536; i++) {
    const sign = (i >> 15) & 1;
    const exp  = (i >> 10) & 0x1f;
    const frac = i & 0x3ff;
    if (exp === 0) {
      f16LUT[i] = (sign ? -1 : 1) * 2 ** -14 * (frac / 1024);
    } else if (exp === 31) {
      f16LUT[i] = frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
    } else {
      f16LUT[i] = (sign ? -1 : 1) * 2 ** (exp - 15) * (1 + frac / 1024);
    }
  }
})();

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

const $loader        = document.getElementById("loader");
const $progressStack = document.getElementById("progress-stack");
const $loaderNote    = document.getElementById("loader-note");
const $app           = document.getElementById("app");
const $input         = document.getElementById("search-input");
const $btn           = document.getElementById("search-btn");
const $results       = document.getElementById("results");
const $status        = document.getElementById("results-status");

// ---------------------------------------------------------------------------
// Progress helpers
// ---------------------------------------------------------------------------

function addProgressRow(id, label) {
  const row = document.createElement("div");
  row.className = "progress-item";
  row.id = `prog-${id}`;
  row.innerHTML = `
    <div class="progress-label">
      <span>${label}</span><span class="progress-pct">0 %</span>
    </div>
    <div class="progress-bar"><div class="progress-fill"></div></div>`;
  $progressStack.appendChild(row);
}

function setProgress(id, pct) {
  const row = document.getElementById(`prog-${id}`);
  if (!row) return;
  row.querySelector(".progress-fill").style.width = `${pct}%`;
  row.querySelector(".progress-pct").textContent = `${Math.round(pct)} %`;
}

// ---------------------------------------------------------------------------
// Keyword scoring
// ---------------------------------------------------------------------------

function contentWords(text) {
  return text.toLowerCase().match(/[a-z]{2,}/g)?.filter(w => !STOP_WORDS.has(w)) ?? [];
}

let defTokens;

function buildDefTokenIndex() {
  defTokens = new Array(wordEntries.length);
  for (let i = 0; i < wordEntries.length; i++) {
    const entry = wordEntries[i];
    defTokens[i] = new Set(contentWords(entry.d + " " + entry.w.join(" ")));
  }
}

function keywordScore(queryTokens, defTokenSet) {
  if (queryTokens.length === 0) return 0;
  let hits = 0;
  for (const t of queryTokens) if (defTokenSet.has(t)) hits++;
  return hits / queryTokens.length;
}

// ---------------------------------------------------------------------------
// Device detection
// ---------------------------------------------------------------------------

function shouldUseLiteMode() {
  const params = new URLSearchParams(location.search);
  if (params.get("mode") === "lite") return true;
  return false;
}

// ---------------------------------------------------------------------------
// PotionModel — Pure JS Model2Vec inference
// ---------------------------------------------------------------------------

class PotionModel {
  constructor(vocabMap, matrixRaw, dims) {
    this.vocab  = vocabMap;
    this.matrix = matrixRaw;
    this.dims   = dims;
    this.unkId  = vocabMap.get("[UNK]");
  }

  _preTokenize(text) {
    text = text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
    const tokens = [];
    let i = 0;
    while (i < text.length) {
      if (/\s/.test(text[i])) { i++; continue; }
      if (this._isPunct(text[i])) { tokens.push(text[i]); i++; continue; }
      let word = "";
      while (i < text.length && !this._isPunct(text[i]) && !/\s/.test(text[i])) {
        word += text[i]; i++;
      }
      if (word) tokens.push(word);
    }
    return tokens;
  }

  _isPunct(ch) {
    const cp = ch.codePointAt(0);
    return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
           (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126);
  }

  _wordPiece(word) {
    if (this.vocab.has(word)) return [this.vocab.get(word)];
    const ids = [];
    let start = 0;
    while (start < word.length) {
      let end = word.length;
      let found = false;
      while (start < end) {
        const sub = (start === 0 ? "" : "##") + word.slice(start, end);
        if (this.vocab.has(sub)) { ids.push(this.vocab.get(sub)); found = true; break; }
        end--;
      }
      if (!found) { ids.push(this.unkId); start++; } else { start = end; }
    }
    return ids;
  }

  tokenize(text) {
    const words = this._preTokenize(text);
    const ids = [];
    for (const w of words) ids.push(...this._wordPiece(w));
    return ids.filter(id => id !== this.unkId);
  }

  encode(text) {
    const ids = this.tokenize(text);
    if (ids.length === 0) return new Float32Array(this.dims);
    const vec = new Float32Array(this.dims);
    for (const id of ids) {
      const off = id * this.dims;
      if (off + this.dims > this.matrix.length) continue;
      for (let d = 0; d < this.dims; d++) vec[d] += f16LUT[this.matrix[off + d]];
    }
    const n = ids.length;
    for (let d = 0; d < this.dims; d++) vec[d] /= n;
    let norm = 0;
    for (let d = 0; d < this.dims; d++) norm += vec[d] * vec[d];
    norm = Math.sqrt(norm) || 1e-32;
    for (let d = 0; d < this.dims; d++) vec[d] /= norm;
    return vec;
  }
}

// ---------------------------------------------------------------------------
// Model + data state
// ---------------------------------------------------------------------------

let fullTokenizer;
let fullModel;
let potionModel;
let wordEntries;

// Potion int8 embeddings (lite mode scoring)
let potionEmbInt8;      // Uint8Array  — int8 quantized potion embeddings
let potionRangeMin;     // Float32Array(256) — per-dim min
let potionRangeScale;   // Float32Array(256) — per-dim range

// Full-mode int8 embeddings
let fullEmbInt8;        // Uint8Array  — int8 quantized full embeddings
let fullRangeMin;       // Float32Array(384) — per-dim min
let fullRangeScale;     // Float32Array(384) — per-dim range

// ---------------------------------------------------------------------------
// Transformers.js loader
// ---------------------------------------------------------------------------

async function loadTransformers() {
  const { AutoModel, AutoTokenizer, env } = await import(
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4"
  );
  env.allowLocalModels  = true;
  env.allowRemoteModels = false;

  let device = "wasm";
  if (typeof navigator !== "undefined" && navigator.gpu) {
    try { if (await navigator.gpu.requestAdapter()) device = "webgpu"; } catch {}
  }
  return { AutoModel, AutoTokenizer, device };
}

// ---------------------------------------------------------------------------
// Data loading (shared by both modes)
// ---------------------------------------------------------------------------

async function fetchWithProgress(url, progressId) {
  const res = await fetch(url);
  const total = +res.headers.get("Content-Length") || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (total) setProgress(progressId, (loaded / total) * 100);
  }
  setProgress(progressId, 100);
  const buf = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) { buf.set(c, offset); offset += c.length; }
  return buf.buffer;
}

async function loadPotionData() {
  addProgressRow("matrix", "Embedding model (~15 MB)");
  addProgressRow("emb",    "Dictionary vectors (~28 MB)");
  addProgressRow("words",  "Word list + vocab (~12 MB)");
  $loaderNote.textContent = $loaderNote.textContent || "First visit downloads ~55 MB (cached for future visits)";

  const vocabPromise = fetch(`${DATA_ROOT}/vocab.txt`).then(async r => {
    const lines = (await r.text()).split(/\r?\n/);
    const map = new Map();
    for (let i = 0; i < lines.length; i++) if (lines[i] !== "") map.set(lines[i], i);
    setProgress("words", 50);
    return map;
  });

  const matrixPromise = fetchWithProgress(`${DATA_ROOT}/potion_matrix.bin`, "matrix")
    .then(buf => new Uint16Array(buf));

  const embPromise = fetchWithProgress(`${DATA_ROOT}/embeddings_potion_int8.bin`, "emb")
    .then(buf => { potionEmbInt8 = new Uint8Array(buf); });

  const rangesPromise = fetch(`${DATA_ROOT}/embeddings_potion_ranges.bin`)
    .then(r => r.arrayBuffer()).then(buf => {
      const data = new Float32Array(buf);
      potionRangeMin   = data.subarray(0, LITE_DIMS);
      potionRangeScale = data.subarray(LITE_DIMS, LITE_DIMS * 2);
    });

  const wordsPromise = fetch(`${DATA_ROOT}/words.json`).then(async r => {
    wordEntries = await r.json();
    setProgress("words", 100);
  });

  const [vocabMap, matrixRaw] = await Promise.all([
    vocabPromise, matrixPromise, embPromise, rangesPromise, wordsPromise,
  ]);
  potionModel = new PotionModel(vocabMap, matrixRaw, LITE_DIMS);
}

// ---------------------------------------------------------------------------
// Full-mode loader (during init, with progress bars)
// ---------------------------------------------------------------------------

async function loadFullModel() {
  addProgressRow("tf",    "AI model (~22 MB)");
  addProgressRow("femb",  "Full vectors (~42 MB)");

  const tfPromise = loadTransformers();

  const int8Promise = fetchWithProgress(`${DATA_ROOT}/embeddings_int8.bin`, "femb")
    .then(buf => new Uint8Array(buf));
  const rangesPromise = fetch(`${DATA_ROOT}/embeddings_ranges.bin`)
    .then(r => r.arrayBuffer()).then(buf => new Float32Array(buf));

  const { AutoModel, AutoTokenizer, device } = await tfPromise;
  setProgress("tf", 40);

  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(FULL_MODEL_ID),
    AutoModel.from_pretrained(FULL_MODEL_ID, { dtype: "q8", device }),
  ]);
  fullTokenizer = tokenizer;
  fullModel = model;
  setProgress("tf", 80);

  // Warm up: run a dummy inference so first real query is fast
  const warmInput = await fullTokenizer("warm up", { padding: true, truncation: true });
  await fullModel(warmInput);
  setProgress("tf", 100);

  const [int8Data, rangesData] = await Promise.all([int8Promise, rangesPromise]);
  fullEmbInt8    = int8Data;
  fullRangeMin   = rangesData.subarray(0, FULL_DIMS);
  fullRangeScale = rangesData.subarray(FULL_DIMS, FULL_DIMS * 2);

  fullReady = true;
  DIMS = FULL_DIMS;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  MODE = shouldUseLiteMode() ? "lite" : "full";
  DIMS = MODE === "full" ? FULL_DIMS : LITE_DIMS;

  if (MODE === "full") {
    // Full mode: load everything during the loading screen
    // Potion data + full model in parallel
    $loaderNote.textContent = "First visit downloads ~120 MB (cached for future visits)";
    const potionPromise = loadPotionData();
    const fullPromise = loadFullModel().catch(err => {
      console.warn("Full model load failed, falling back to lite:", err.message);
      MODE = "lite";
      DIMS = LITE_DIMS;
    });
    await Promise.all([potionPromise, fullPromise]);
  } else {
    await loadPotionData();
  }

  buildDefTokenIndex();

  // Show app
  $loader.classList.add("done");
  $app.classList.remove("hidden");
  $input.focus();
  startShowcase();
  showModeBadge();

  // Deep link
  const q = new URLSearchParams(location.search).get("q");
  if (q) { $input.value = q; search(q); }
}

// ---------------------------------------------------------------------------
// Mode badge
// ---------------------------------------------------------------------------

function showModeBadge() {
  const badge = document.createElement("span");
  badge.className = "mode-badge";
  badge.id = "mode-badge";
  const isLite = MODE === "lite";
  badge.textContent = isLite ? "Lite" : "Full";
  badge.title = isLite
    ? "Lite mode (potion-base-8M). Add ?mode=full for higher quality."
    : "Full mode (mdbr-leaf-mt). Add ?mode=lite for lower memory.";
  document.querySelector(".brand")?.appendChild(badge);
}

function updateModeBadge() {
  const badge = document.getElementById("mode-badge");
  if (!badge) return;
  badge.textContent = "Full";
  badge.title = "Full mode (mdbr-leaf-mt). Add ?mode=lite for lower memory.";
}

// ---------------------------------------------------------------------------
// Query embedding
// ---------------------------------------------------------------------------

async function embedQuery(query) {
  if (MODE === "lite" || !fullReady) {
    return potionModel.encode(query);
  }
  const inputs = await fullTokenizer(
    "Represent this sentence for searching relevant passages: " + query,
    { padding: true, truncation: true },
  );
  const { sentence_embedding } = await fullModel(inputs);
  const emb1024 = sentence_embedding.data;

  // MRL truncation to 384d + re-normalize
  const vec = new Float32Array(FULL_DIMS);
  for (let i = 0; i < FULL_DIMS; i++) vec[i] = emb1024[i];
  let norm = 0;
  for (let i = 0; i < FULL_DIMS; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1e-32;
  for (let i = 0; i < FULL_DIMS; i++) vec[i] /= norm;
  return vec;
}

// ---------------------------------------------------------------------------
// Rate limiter
// ---------------------------------------------------------------------------

const _ts = [];

function isRateLimited() {
  const now = Date.now();
  while (_ts.length && _ts[0] <= now - RATE_LIMIT_MS) _ts.shift();
  if (_ts.length >= RATE_LIMIT_MAX) return true;
  _ts.push(now);
  return false;
}

// ---------------------------------------------------------------------------
// Int8 dot product scoring (shared by lite + full modes)
// ---------------------------------------------------------------------------

function scoreInt8(qvec, embData, rangeMin, rangeScale, dims, count, out) {
  const qScaled = new Float32Array(dims);
  let qOffset = 0;
  for (let d = 0; d < dims; d++) {
    qScaled[d] = qvec[d] * rangeScale[d] / 255;
    qOffset += qvec[d] * rangeMin[d];
  }
  for (let i = 0; i < count; i++) {
    let dot = qOffset;
    const base = i * dims;
    for (let d = 0; d < dims; d++) dot += qScaled[d] * embData[base + d];
    out[i] = dot;
  }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async function search(query) {
  query = query.trim();
  if (!query) { $results.innerHTML = ""; $status.textContent = ""; return; }
  if (isRateLimited()) { $status.textContent = "Too many searches — please wait a moment."; return; }

  const qTokens = contentWords(query);
  const count   = wordEntries.length;

  // Instant potion preview (full mode: show while model runs)
  if (fullReady && potionModel && potionEmbInt8) {
    $status.textContent = "Searching\u2026";
    const potionVec = potionModel.encode(query);
    const preview = new Float32Array(count);
    scoreInt8(potionVec, potionEmbInt8, potionRangeMin, potionRangeScale, LITE_DIMS, count, preview);
    for (let i = 0; i < count; i++) {
      preview[i] = W_COSINE * preview[i] + W_KEYWORD * keywordScore(qTokens, defTokens[i]);
    }
    render(topK(preview, count), query);
    $status.textContent = "Refining\u2026";
  } else {
    $status.textContent = "Searching\u2026";
  }

  const qvec   = await embedQuery(query);
  const scored = new Float32Array(count);

  if (fullReady) {
    scoreInt8(qvec, fullEmbInt8, fullRangeMin, fullRangeScale, FULL_DIMS, count, scored);
  } else {
    // Lite: int8 cosine + keyword
    scoreInt8(qvec, potionEmbInt8, potionRangeMin, potionRangeScale, LITE_DIMS, count, scored);
    for (let i = 0; i < count; i++) {
      scored[i] = W_COSINE * scored[i] + W_KEYWORD * keywordScore(qTokens, defTokens[i]);
    }
  }

  render(topK(scored, count), query);
}

function topK(scored, count) {
  const indices = Array.from({ length: count }, (_, i) => i);
  indices.sort((a, b) => scored[b] - scored[a]);
  const seen = new Set();
  const deduped = [];
  for (const idx of indices) {
    if (deduped.length >= TOP_K) break;
    const entry = wordEntries[idx];
    const primary = entry.w[0].toLowerCase();
    if (seen.has(primary)) continue;
    seen.add(primary);
    deduped.push({ ...entry, score: scored[idx] });
  }
  return deduped;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function render(items, query) {
  if (items.length === 0) {
    $results.innerHTML = "";
    $status.textContent = "No matches — try rephrasing your description.";
    return;
  }
  $status.textContent = `Top ${items.length} matches`;
  const url = new URL(location.href);
  url.searchParams.set("q", query);
  history.replaceState(null, "", url);

  const maxScore = items[0].score;
  $results.innerHTML = items.map((it, i) => {
    const primary = it.w[0];
    const alt = it.w.length > 1 ? it.w.slice(1).join(", ") : "";
    const pct = Math.round((it.score / maxScore) * 100);
    return `
      <article class="result-card" style="animation-delay:${i * 30}ms">
        <div class="card-head">
          <span class="card-word">${esc(primary)}</span>
          <span class="card-pos" data-pos="${esc(it.p)}">${esc(it.p)}</span>
          ${alt ? `<span class="card-words-alt">${esc(alt)}</span>` : ""}
        </div>
        <p class="card-def">${esc(it.d)}</p>
        <div class="card-score">
          <div class="score-bar"><div class="score-fill" style="width:${pct}%"></div></div>
          <span class="score-pct">${pct}%</span>
        </div>
      </article>`;
  }).join("");
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// ---------------------------------------------------------------------------
// Rolling showcase
// ---------------------------------------------------------------------------

const SHOWCASE = [
  { q: "a feeling of longing for the past",       w: "nostalgia" },
  { q: "fear of being forgotten",                  w: "athazagoraphobia" },
  { q: "the art of beautiful handwriting",         w: "calligraphy" },
  { q: "a word that sounds like what it means",    w: "onomatopoeia" },
  { q: "pleasure from someone else's misfortune",  w: "schadenfreude" },
  { q: "unable to be put into words",              w: "ineffable" },
  { q: "lasting only a very short time",           w: "ephemeral" },
  { q: "a love of books",                          w: "bibliophilia" },
  { q: "wanderlust for the sea",                   w: "thalassophilia" },
  { q: "the smell of rain on dry earth",           w: "petrichor" },
];

let showcaseIdx = 0;
let showcaseInterval;

function cycleShowcase() {
  const $showcase = document.getElementById("showcase");
  if (!$showcase) return;
  const item = SHOWCASE[showcaseIdx % SHOWCASE.length];
  showcaseIdx++;
  const $row = $showcase.querySelector(".showcase-row");
  $row.style.animation = "none";
  void $row.offsetWidth;
  $row.querySelector(".showcase-query").textContent = `"${item.q}"`;
  $row.querySelector(".showcase-word").textContent = item.w;
  $row.style.animation = "showcaseFade .6s ease both";
}

function startShowcase() {
  cycleShowcase();
  showcaseInterval = setInterval(cycleShowcase, 3500);
}

function stopShowcase() {
  if (showcaseInterval) { clearInterval(showcaseInterval); showcaseInterval = null; }
  const $showcase = document.getElementById("showcase");
  if ($showcase) $showcase.style.display = "none";
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

const $hamburger = document.getElementById("nav-hamburger");
const $navLinks  = document.getElementById("nav-links");
if ($hamburger && $navLinks) {
  $hamburger.addEventListener("click", () => {
    const open = $navLinks.classList.toggle("open");
    $hamburger.setAttribute("aria-expanded", open);
  });
  $navLinks.addEventListener("click", (e) => {
    if (e.target.closest("a")) {
      $navLinks.classList.remove("open");
      $hamburger.setAttribute("aria-expanded", "false");
    }
  });
}

let timer;
$input.addEventListener("input", () => {
  clearTimeout(timer);
  if ($input.value.trim()) stopShowcase();
  timer = setTimeout(() => search($input.value), DEBOUNCE);
});
$input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { clearTimeout(timer); search($input.value); }
  if (e.key === "Escape") { $input.value = ""; $results.innerHTML = ""; $status.textContent = ""; }
});
$btn.addEventListener("click", () => { clearTimeout(timer); search($input.value); });

document.querySelectorAll(".example-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    stopShowcase();
    $input.value = chip.dataset.query;
    search(chip.dataset.query);
    $input.focus();
  });
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}

init().catch(err => {
  console.error(err);
  $loaderNote.textContent = `Error: ${err.message}. Please refresh the page.`;
  $loaderNote.style.color = "#DC2626";
});
