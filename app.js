/**
 * WordFor: Reverse Dictionary
 * © 2025 Zeeshan Khan Suri (zshn25). Licensed under CC-BY-NC-ND-4.0.
 *
 * Default:       mdbr-leaf-mt (query) + mxbai-embed-large (defs) via Transformers.js
 *                Binary (ITQ) first-pass + int3 reranking for best quality at binary speed.
 * Mobile:        Same model, but pure binary ITQ scoring (no rerank download).
 * Lite fallback: potion-base-8M via pure JS static embeddings (sub-1ms)
 *
 * Lite mode activates automatically if the full model fails to load,
 * or manually via ?mode=lite in the URL.
 * Binary-only scoring activates on mobile, or via ?scoring=binary.
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DATA_ROOT = "data";
const DATA_VERSION = "v3";  // Bump when data files change to bust browser/CDN cache
const TOP_K = 30;
const SHOW_K = 9;
const DEBOUNCE = 400;
const RATE_LIMIT_MAX = 15;
const RATE_LIMIT_MS = 10_000;

// Words that should never surface as top results in a reverse dictionary.
// These stay in the embedding database (they're useful as variants and context)
// but are skipped when creating new result groups in topK().
// Only the primary word (w[0]) is checked — morphological variants are unaffected.
const RESULT_EXCLUDE = new Set([
  // Articles and determiners
  "a", "an", "the",
  // Coordinating conjunctions
  "and", "but", "or", "nor", "for", "yet", "so",
  // Common prepositions
  "at", "by", "from", "in", "into", "of", "on", "to", "up", "with",
  // Core auxiliary verbs
  "be", "been", "being", "is", "are", "was", "were",
  "have", "has", "had", "do", "does", "did",
  "will", "would", "shall", "should", "may", "might", "must", "can", "could",
  // Pronouns
  "he", "she", "it", "they", "we", "i", "you",
  "his", "her", "its", "their", "our", "my", "your",
  "this", "that", "these", "those",
  // Single-letter non-word entries
  "b", "c", "d", "e", "f", "g", "h", "j", "k", "l", "m",
  "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
  // Ultra-generic nouns never useful as a "word you were thinking of"
  "thing", "stuff", "sort", "kind", "type", "part", "bit", "lot",
]);

const FULL_MODEL_ID = "onnx-community/mdbr-leaf-mt-ONNX";
const FULL_DIMS = 384;
const LITE_DIMS = 256;
let MODE = null;
let DIMS = null;
let fullReady = false;

// ---------------------------------------------------------------------------
// Lightweight performance instrumentation (TTI / search-ready / fetch / latency)
// Marks are kept in-memory and can be dumped via window.wordforPerf() for the
// perf_report.md measurements. No data leaves the browser.
// ---------------------------------------------------------------------------

const perf = {
  t0: (typeof performance !== "undefined" ? performance.now() : Date.now()),
  marks: {},
  measures: {},
  now() { return (typeof performance !== "undefined" ? performance.now() : Date.now()); },
  mark(name) { this.marks[name] = this.now(); },
  measure(name, fromMark) {
    const end = this.now();
    const start = this.marks[fromMark] != null ? this.marks[fromMark] : this.t0;
    this.measures[name] = +(end - start).toFixed(1);
    return this.measures[name];
  },
  sinceStart(name) { this.measures[name] = +(this.now() - this.t0).toFixed(1); return this.measures[name]; },
  dump() { return { sinceStart_ms: +(this.now() - this.t0).toFixed(1), measures: this.measures }; },
};
if (typeof window !== "undefined") window.wordforPerf = () => perf.dump();

// ---------------------------------------------------------------------------
// Float-16 → Float-32 lookup table  (65 536 entries ≈ 256 KB)
// ---------------------------------------------------------------------------

const f16LUT = new Float32Array(65536);
(function buildLUT() {
  for (let i = 0; i < 65536; i++) {
    const sign = (i >> 15) & 1;
    const exp = (i >> 10) & 0x1f;
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

const $loader = document.getElementById("loader");
const $progressStack = document.getElementById("progress-stack");
const $loaderNote = document.getElementById("loader-note");
const $app = document.getElementById("app");
const $input = document.getElementById("search-input");
const $btn = document.getElementById("search-btn");
const $results = document.getElementById("results");
const $status = document.getElementById("results-status");

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
  pct = Math.min(100, Math.max(0, pct));
  row.querySelector(".progress-fill").style.width = `${pct}%`;
  row.querySelector(".progress-pct").textContent = `${Math.round(pct)} %`;
}

// ---------------------------------------------------------------------------
// Device detection
// ---------------------------------------------------------------------------

function shouldUseLiteMode() {
  const params = new URLSearchParams(location.search);
  if (params.get("mode") === "lite") return true;
  if (params.get("mode") === "full") return false;
  // iOS Safari can't load ONNX models (WASM OOM for both q8 and q4f16)
  const ua = navigator.userAgent;
  if (/iPhone|iPad|iPod/.test(ua)) return true;
  return false;
}

/**
 * Detect whether to use lightweight binary-only scoring (skip int8 download).
 * Mobile devices get pure binary ITQ by default (saves ~65 MB).
 * Override with ?scoring=rerank (force int8 reranking) or ?scoring=binary.
 */
function shouldUseBinaryOnly() {
  const params = new URLSearchParams(location.search);
  const scoring = params.get("scoring");
  if (scoring === "binary") return true;
  if (scoring === "rerank") return false;
  // Auto-detect: mobile/tablet -> binary only
  const ua = navigator.userAgent;
  return /Android|iPhone|iPad|iPod|Mobile|Tablet/i.test(ua);
}

let BINARY_ONLY = false;  // set during init

// ---------------------------------------------------------------------------
// WasmPotionModel: model2vec-rs WASM inference (fast, uses real tokenizer)
// ---------------------------------------------------------------------------

class WasmPotionModel {
  constructor(wasmModel) {
    this._model = wasmModel;
    this.dims = wasmModel.dims();
  }

  static async load(progressId) {
    const root = `${DATA_ROOT}/wasm`;
    const wasmModule = await import(`./${root}/model2vec_wasm.js`);
    // Init WASM runtime (auto-fetches .wasm file relative to module URL)
    await wasmModule.default();
    // Fetch model files in parallel
    const [tokBytes, modelBytes, cfgBytes] = await Promise.all([
      fetch(`${root}/tokenizer.json`).then(r => r.arrayBuffer()).then(b => new Uint8Array(b)),
      fetchWithProgress(`${root}/model.safetensors`, progressId)
        .then(buf => new Uint8Array(buf)),
      fetch(`${root}/config.json`).then(r => r.arrayBuffer()).then(b => new Uint8Array(b)),
    ]);
    const model = new wasmModule.Model(tokBytes, modelBytes, cfgBytes);
    return new WasmPotionModel(model);
  }

  encode(text) {
    return this._model.encode_single(text);
  }
}

// ---------------------------------------------------------------------------
// PotionModel: Pure JS Model2Vec inference (fallback)
// ---------------------------------------------------------------------------

class PotionModel {
  constructor(vocabMap, matrixRaw, dims) {
    this.vocab = vocabMap;
    this.matrix = matrixRaw;
    this.dims = dims;
    this.unkId = vocabMap.get("[UNK]");
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
let potionEmbInt4;      // Uint8Array : int4 packed potion embeddings (2 dims/byte)
let potionRangeMin;     // Float32Array(256): per-dim min
let potionRangeScale;   // Float32Array(256): per-dim range

// Full-mode embeddings (for reranking stage 2)
let fullEmbInt8;        // Uint8Array : int8 quantized full embeddings
let fullEmbInt4;        // Uint8Array : int4 packed full embeddings (2 nibbles/byte)
let fullEmbInt3;        // Uint8Array : int3 packed full embeddings (8 dims/3 bytes)
let fullRangeMin;       // Float32Array(384): per-dim min
let fullRangeScale;     // Float32Array(384): per-dim range

// Full-mode binary (1-bit) embeddings with ITQ rotation (primary scoring)
const FULL_BINARY_BYTES = FULL_DIMS / 8;  // 384 / 8 = 48 bytes per entry
let fullEmbBinary;      // Uint8Array : packed binary embeddings (ITQ-rotated)
let fullBinaryReady = false;
let itqMean;            // Float32Array(384): ITQ centering vector
let itqR;               // Float32Array(384*384): ITQ rotation matrix (flattened, row-major)
let itqReady = false;   // true when ITQ calibration is loaded
const RERANK_K = 500;   // number of binary candidates to rerank with int8


// ---------------------------------------------------------------------------
// Transformers.js loader
// ---------------------------------------------------------------------------

async function loadTransformers() {
  // Try self-hosted vendor first so the full model works even with CDN blockers.
  // To enable: download the library file (e.g. with your browser) and save to:
  //   wordfor/vendor/transformers.min.js
  // URL: https://cdn.jsdelivr.net/npm/@huggingface/transformers@4/dist/transformers.min.js
  const SOURCES = [
    "/vendor/transformers.min.js",
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4",
    "https://unpkg.com/@huggingface/transformers@4",
  ];
  let imported;
  for (const src of SOURCES) {
    try { imported = await import(src); break; } catch { /* try next */ }
  }
  if (!imported) throw new Error("AI library blocked by browser extension (all sources failed)");

  const { AutoModel, AutoTokenizer, env } = imported;
  env.allowLocalModels = true;
  env.allowRemoteModels = false;  // Model files are self-hosted; suppress HuggingFace Hub fetches

  // Use ?device=webgpu to opt-in; default is always WASM (reliable everywhere)
  const params = new URLSearchParams(location.search);
  let device = params.get("device") === "webgpu" ? "webgpu" : "wasm";
  if (device === "webgpu") {
    try {
      if (!navigator.gpu || !(await navigator.gpu.requestAdapter())) device = "wasm";
    } catch { device = "wasm"; }
  }
  return { AutoModel, AutoTokenizer, device };
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

/** Append cache-busting version query param to data URLs. */
function dataUrl(path) {
  return `${DATA_ROOT}/${path}?v=${DATA_VERSION}`;
}

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

async function loadWordList() {
  addProgressRow("words", "Word list (~18 MB)");
  perf.mark("words_start");
  // Prefer cache-first sharded loader (assets/model-manifest.json); fall back to
  // the monolithic data/words.json when no manifest is deployed.
  let loadedViaShards = false;
  try {
    if (window.ShardLoader && (await window.ShardLoader.isSharded("words"))) {
      wordEntries = await window.ShardLoader.loadJSON("words", {
        onProgress: (pct) => setProgress("words", pct),
      });
      loadedViaShards = true;
    }
  } catch (e) {
    console.warn("shard load of words.json failed, falling back to monolith:", e.message);
  }
  if (!loadedViaShards) {
    const res = await fetch(dataUrl("words.json"));
    setProgress("words", 50);
    wordEntries = await res.json();
  }
  setProgress("words", 100);
  perf.measure("words_loaded", "words_start");
  // Lemma family map (built at compile time; replaces runtime stemming).
  // Non-fatal: search still works without it, just without inflectional grouping.
  try {
    let obj = null;
    if (window.ShardLoader && (await window.ShardLoader.isSharded("forms_to_lemma"))) {
      obj = await window.ShardLoader.loadJSON("forms_to_lemma");
    } else {
      const lemRes = await fetch(dataUrl("forms_to_lemma.json"));
      if (lemRes.ok) obj = await lemRes.json();
    }
    if (obj) formsToLemma = new Map(Object.entries(obj));
  } catch (e) {
    console.warn("forms_to_lemma.json not loaded:", e.message);
  }
}

async function loadPotionData() {
  addProgressRow("matrix", "Embedding model (~15 MB)");
  addProgressRow("emb", "Dictionary vectors (~22 MB)");
  $loaderNote.textContent = $loaderNote.textContent || "First visit downloads ~55 MB (cached for future visits)";

  // Shared data needed regardless of WASM or JS model
  const embPromise = fetchWithProgress(dataUrl("embeddings_potion_int4.bin"), "emb")
    .then(buf => { potionEmbInt4 = new Uint8Array(buf); });

  const rangesPromise = fetch(dataUrl("embeddings_potion_ranges.bin"))
    .then(r => r.arrayBuffer()).then(buf => {
      const data = new Float32Array(buf);
      potionRangeMin = data.subarray(0, LITE_DIMS);
      potionRangeScale = data.subarray(LITE_DIMS, LITE_DIMS * 2);
    });

  // Try WASM model first (faster inference, real tokenizer)
  let wasmOk = false;
  const wasmPromise = WasmPotionModel.load("matrix")
    .then(m => { potionModel = m; wasmOk = true; })
    .catch(err => { console.warn("WASM model failed, falling back to pure JS:", err); });

  await Promise.all([wasmPromise, embPromise, rangesPromise]);

  if (!wasmOk) {
    // Fallback: load pure JS PotionModel (vocab + f16 matrix)
    const vocabPromise = fetch(dataUrl("vocab.txt")).then(async r => {
      const lines = (await r.text()).split(/\r?\n/);
      const map = new Map();
      for (let i = 0; i < lines.length; i++) if (lines[i] !== "") map.set(lines[i], i);
      return map;
    });
    const matrixPromise = fetchWithProgress(dataUrl("potion_matrix.bin"), "matrix")
      .then(buf => new Uint16Array(buf));
    const [vocabMap, matrixRaw] = await Promise.all([vocabPromise, matrixPromise]);
    potionModel = new PotionModel(vocabMap, matrixRaw, LITE_DIMS);
  }
}

// ---------------------------------------------------------------------------
// Full-mode loader (during init, with progress bars)
// ---------------------------------------------------------------------------

function timeout(ms, promise) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), ms)),
  ]);
}

async function loadFullModel() {
  addProgressRow("tf", "AI model (~22 MB)");
  addProgressRow("femb", "Dictionary vectors (~9 MB)");

  const tfPromise = loadTransformers();

  // Binary embeddings (~8 MB): primary scoring (fast Hamming)
  const binaryPromise = fetchWithProgress(dataUrl("embeddings_binary.bin"), "femb")
    .then(buf => new Uint8Array(buf))
    .catch(() => null);

  // ITQ calibration (~577 KB): rotation matrix for binary scoring
  const itqPromise = fetch(dataUrl("embeddings_itq.bin"))
    .then(r => r.ok ? r.arrayBuffer() : null)
    .catch(() => null);

  // Int8 embeddings loaded lazily after app is shown (loadFullInt8)

  const { AutoModel, AutoTokenizer, device } = await tfPromise;
  setProgress("tf", 20);

  // Try q8 first; on iOS Safari WASM OOM, fall back to q4f16 with reduced memory
  let tokenizer, model;
  tokenizer = await AutoTokenizer.from_pretrained(FULL_MODEL_ID);
  const modelProgress = (p) => {
    // p.progress goes 0-100 for each file download; map to 20%-80% range
    if (p.status === "progress" && p.progress != null) {
      setProgress("tf", 20 + p.progress * 0.6);
    }
  };
  try {
    model = await timeout(BINARY_ONLY ? 30_000 : 90_000, AutoModel.from_pretrained(FULL_MODEL_ID, {
      dtype: "q8", device,
      session_options: { enableCpuMemArena: false },
      progress_callback: modelProgress,
    }));
  } catch (e) {
    console.warn("q8 model failed, trying q4f16 with reduced memory:", e.message);
    setProgress("tf", 20);
    model = await timeout(BINARY_ONLY ? 45_000 : 90_000, AutoModel.from_pretrained(FULL_MODEL_ID, {
      dtype: "q4f16", device,
      session_options: { enableCpuMemArena: false },
      progress_callback: modelProgress,
    }));
  }
  fullTokenizer = tokenizer;
  fullModel = model;
  setProgress("tf", 85);

  // Warm up: run a dummy inference so first real query is fast
  const warmInput = await fullTokenizer("warm up", { padding: true, truncation: true });
  await timeout(30_000, fullModel(warmInput));
  setProgress("tf", 100);

  // Load binary + ITQ calibration
  const binaryData = await binaryPromise;
  if (binaryData) {
    fullEmbBinary = binaryData;
    fullBinaryReady = true;
  }

  const itqData = await itqPromise;
  if (itqData) {
    const itqFloat = new Float32Array(itqData);
    itqMean = itqFloat.subarray(0, FULL_DIMS);
    itqR = itqFloat.subarray(FULL_DIMS, FULL_DIMS + FULL_DIMS * FULL_DIMS);
    itqReady = true;
  }

  fullReady = true;
  DIMS = FULL_DIMS;
}

/**
 * Lazy-load reranking embeddings (desktop only).
 * Tries int3 (~75 MB, best MRR) first, then int4 (~100 MB), then int8 (~200 MB).
 * The app starts with binary-only scoring and upgrades silently.
 */
async function loadFullRerank() {
  if (BINARY_ONLY) return;

  // Load ranges (shared by all quant formats)
  const rangesBuf = await fetch(dataUrl("embeddings_ranges.bin")).then(r => r.arrayBuffer());
  const rd = new Float32Array(rangesBuf);
  fullRangeMin = rd.subarray(0, FULL_DIMS);
  fullRangeScale = rd.subarray(FULL_DIMS, FULL_DIMS * 2);

  // Try int3 first (best MRR)
  try {
    let int3Buf, int3RangesBuf;
    if (window.ShardLoader && (await window.ShardLoader.isSharded("embeddings_int3"))) {
      // Cache-first sharded fetch (background priority) -> upgrades ranking silently
      [int3Buf, int3RangesBuf] = await Promise.all([
        window.ShardLoader.loadAsset("embeddings_int3"),
        window.ShardLoader.loadAsset("embeddings_int3_ranges"),
      ]);
    } else {
      [int3Buf, int3RangesBuf] = await Promise.all([
        fetch(dataUrl("embeddings_int3.bin")).then(r => {
          if (!r.ok) throw new Error("int3 not found");
          return r.arrayBuffer();
        }),
        fetch(dataUrl("embeddings_int3_ranges.bin")).then(r => {
          if (!r.ok) throw new Error("int3 ranges not found");
          return r.arrayBuffer();
        }),
      ]);
    }
    fullEmbInt3 = new Uint8Array(int3Buf);
    const rd3 = new Float32Array(int3RangesBuf);
    fullRangeMin = rd3.subarray(0, FULL_DIMS);
    fullRangeScale = rd3.subarray(FULL_DIMS, FULL_DIMS * 2);
    perf.sinceStart("search_ready_best_ms");
    console.log("Loaded int3 reranking embeddings");
    return;
  } catch (e) {
    console.log("Int3 not available, trying int4:", e.message);
  }

  // Try int4 (half the size of int8)
  try {
    const int4Buf = await fetch(dataUrl("embeddings_int4.bin")).then(r => {
      if (!r.ok) throw new Error("int4 not found");
      return r.arrayBuffer();
    });
    fullEmbInt4 = new Uint8Array(int4Buf);
    console.log("Loaded int4 reranking embeddings");
    return;
  } catch (e) {
    console.log("Int4 not available, trying int8:", e.message);
  }

  // Fallback to int8
  try {
    const int8Buf = await fetch(dataUrl("embeddings_int8.bin")).then(r => r.arrayBuffer());
    fullEmbInt8 = new Uint8Array(int8Buf);
    console.log("Loaded int8 reranking embeddings");
  } catch (e) {
    console.warn("Reranking embeddings failed, using binary-only scoring:", e.message);
  }
}


// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  MODE = shouldUseLiteMode() ? "lite" : "full";
  DIMS = MODE === "full" ? FULL_DIMS : LITE_DIMS;
  BINARY_ONLY = MODE === "full" && shouldUseBinaryOnly();
  let liteFallback = false;

  if (MODE === "full") {
    $loaderNote.textContent = BINARY_ONLY
      ? "First visit downloads ~30 MB (cached for future visits)"
      : "First visit downloads ~48 MB (cached for future visits)";
    const wordsPromise = loadWordList();
    const fullPromise = loadFullModel().catch(err => {
      console.warn("Full model load failed, falling back to lite:", err.message);
      MODE = "lite";
      DIMS = LITE_DIMS;
      liteFallback = true;
    });
    await Promise.all([wordsPromise, fullPromise]);
    // If full model failed, load potion as fallback
    if (liteFallback) await loadPotionData();
  } else {
    const wordsPromise = loadWordList();
    const potionPromise = loadPotionData();
    await Promise.all([wordsPromise, potionPromise]);
  }

  // Show app
  $loader.classList.add("done");
  $app.classList.remove("hidden");
  perf.sinceStart("search_ready_fast_ms");
  $input.focus();
  startShowcase();
  showModeBadge();

  if (liteFallback) showLiteFallbackBanner();

  // Deep link
  const q = new URLSearchParams(location.search).get("q");
  if (q) { $input.value = q; search(q); }


  // Lazy-load reranking embeddings (non-blocking, upgrades from binary-only)
  loadFullRerank().catch(e => console.warn("Rerank load:", e.message));
}

// ---------------------------------------------------------------------------
// Mode badge
// ---------------------------------------------------------------------------

function showModeBadge() {
  const badge = document.createElement("span");
  badge.className = "mode-badge";
  badge.id = "mode-badge";
  const isLite = MODE === "lite";
  badge.textContent = isLite ? "Lite" : BINARY_ONLY ? "Full (binary)" : "Full";
  badge.title = isLite
    ? "Lite mode (potion-base-8M). Add ?mode=full for higher quality."
    : BINARY_ONLY
      ? "Full mode (mdbr-leaf-mt). Binary-only scoring (mobile). Add ?scoring=rerank for higher quality."
      : "Full mode (mdbr-leaf-mt). Binary+rerank scoring. Add ?mode=lite for lower memory.";
  document.querySelector(".brand")?.appendChild(badge);
}

function updateModeBadge() {
  const badge = document.getElementById("mode-badge");
  if (!badge) return;
  badge.textContent = "Full";
  badge.title = "Full mode (mdbr-leaf-mt). Add ?mode=lite for lower memory.";
}

function showLiteFallbackBanner() {
  const banner = document.createElement("div");
  banner.className = "lite-fallback-banner";
  banner.innerHTML = `
    <span>Running in <strong>Lite mode</strong> \u2014 results may be less accurate. The full AI library couldn\u2019t load, likely blocked by an ad-blocker (e.g. uBlock Origin). To enable Full mode, allow <strong>cdn.jsdelivr.net</strong> or <strong>unpkg.com</strong> in your browser extension settings.</span>
    <button class="banner-close" aria-label="Dismiss">&times;</button>`;
  banner.querySelector(".banner-close").addEventListener("click", () => banner.remove());
  document.getElementById("app").prepend(banner);
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
// Int4 dot product scoring (lite mode)
// ---------------------------------------------------------------------------

function scoreInt4(qvec, int4Data, rangeMin, rangeScale, dims, count, out) {
  const qScaled = new Float32Array(dims);
  let qOffset = 0;
  for (let d = 0; d < dims; d++) {
    qScaled[d] = qvec[d] * rangeScale[d] / 15;
    qOffset += qvec[d] * rangeMin[d];
  }
  const halfDims = dims >> 1;
  for (let i = 0; i < count; i++) {
    let dot = qOffset;
    const base = i * halfDims;
    for (let d = 0; d < dims; d += 2) {
      const packed = int4Data[base + (d >> 1)];
      dot += qScaled[d] * (packed >> 4);
      dot += qScaled[d + 1] * (packed & 0x0F);
    }
    out[i] = dot;
  }
}

// ---------------------------------------------------------------------------
// Int3 dot product scoring (8 dims packed per 3 bytes)
// ---------------------------------------------------------------------------

function scoreInt3(qvec, int3Data, rangeMin, rangeScale, dims, count, out) {
  const qScaled = new Float32Array(dims);
  let qOffset = 0;
  for (let d = 0; d < dims; d++) {
    qScaled[d] = qvec[d] * rangeScale[d] / 7;
    qOffset += qvec[d] * rangeMin[d];
  }
  const bytesPerEntry = (dims * 3) >> 3;
  const nGroups = dims >> 3;
  for (let i = 0; i < count; i++) {
    let dot = qOffset;
    const base = i * bytesPerEntry;
    for (let g = 0; g < nGroups; g++) {
      const b = base + g * 3;
      const w = (int3Data[b] << 16) | (int3Data[b + 1] << 8) | int3Data[b + 2];
      const d8 = g << 3;
      dot += qScaled[d8]     * ((w >> 21) & 7);
      dot += qScaled[d8 + 1] * ((w >> 18) & 7);
      dot += qScaled[d8 + 2] * ((w >> 15) & 7);
      dot += qScaled[d8 + 3] * ((w >> 12) & 7);
      dot += qScaled[d8 + 4] * ((w >>  9) & 7);
      dot += qScaled[d8 + 5] * ((w >>  6) & 7);
      dot += qScaled[d8 + 6] * ((w >>  3) & 7);
      dot += qScaled[d8 + 7] * ( w        & 7);
    }
    out[i] = dot;
  }
}

// ---------------------------------------------------------------------------
// Binary (1-bit) Hamming distance scoring
// ---------------------------------------------------------------------------

// Popcount lookup for 8-bit values
const POPCNT8 = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  let n = i;
  n = n - ((n >> 1) & 0x55);
  n = (n & 0x33) + ((n >> 2) & 0x33);
  POPCNT8[i] = (n + (n >> 4)) & 0x0f;
}

/**
 * Score entries by Hamming similarity between query sign bits and packed binary embeddings.
 * Hamming similarity = (dims - hamming_distance) / dims, mapped to [-1, 1] range.
 * @param {Float32Array} qvec - query vector (float32, FULL_DIMS)
 * @param {Uint8Array} binData - packed binary embeddings (N * bytesPerEntry)
 * @param {number} bytesPerEntry - FULL_DIMS / 8
 * @param {number} count - number of entries
 * @param {Float32Array} out - output scores
 */
function scoreHamming(qvec, binData, bytesPerEntry, count, out) {
  // Apply ITQ rotation if available, then pack query sign bits
  let rotated = qvec;
  if (itqReady) {
    rotated = new Float32Array(FULL_DIMS);
    for (let d = 0; d < FULL_DIMS; d++) {
      let sum = 0;
      for (let k = 0; k < FULL_DIMS; k++) sum += (qvec[k] - itqMean[k]) * itqR[k * FULL_DIMS + d];
      rotated[d] = sum;
    }
  }
  const qBin = new Uint8Array(bytesPerEntry);
  for (let b = 0; b < bytesPerEntry; b++) {
    let byte = 0;
    for (let bit = 0; bit < 8; bit++) {
      if (rotated[b * 8 + bit] > 0) byte |= (128 >> bit);
    }
    qBin[b] = byte;
  }
  const dims = bytesPerEntry * 8;
  for (let i = 0; i < count; i++) {
    const base = i * bytesPerEntry;
    let dist = 0;
    for (let b = 0; b < bytesPerEntry; b++) {
      dist += POPCNT8[qBin[b] ^ binData[base + b]];
    }
    // Map to [-1, 1]: agreement = (dims - 2*dist) / dims
    out[i] = (dims - 2 * dist) / dims;
  }
}

/**
 * Two-stage scoring: binary Hamming first-pass, then int3/int4/int8 dot product reranking.
 * Returns float32 scores for all entries (non-candidates get -Infinity).
 */
function scoreBinaryRerank(qvec, count, out) {
  // Stage 1: Binary Hamming over all entries
  const hamming = new Float32Array(count);
  scoreHamming(qvec, fullEmbBinary, FULL_BINARY_BYTES, count, hamming);

  // Stage 2: Find top RERANK_K candidates by Hamming score
  const topIdx = new Int32Array(count);
  for (let i = 0; i < count; i++) topIdx[i] = i;
  const k = Math.min(RERANK_K, count);
  nthElement(topIdx, hamming, 0, count - 1, k);

  // Fill all with -Infinity, then overwrite reranked candidates
  for (let i = 0; i < count; i++) out[i] = -Infinity;

  if (fullEmbInt3) {
    // Rerank with int3 (best MRR, 8 dims / 3 bytes)
    const qScaled = new Float32Array(FULL_DIMS);
    let qOffset = 0;
    for (let d = 0; d < FULL_DIMS; d++) {
      qScaled[d] = qvec[d] * fullRangeScale[d] / 7;
      qOffset += qvec[d] * fullRangeMin[d];
    }
    const bytesPerEntry = (FULL_DIMS * 3) >> 3;
    const nGroups = FULL_DIMS >> 3;
    for (let j = 0; j < k; j++) {
      const idx = topIdx[j];
      let dot = qOffset;
      const base = idx * bytesPerEntry;
      for (let g = 0; g < nGroups; g++) {
        const b = base + g * 3;
        const w = (fullEmbInt3[b] << 16) | (fullEmbInt3[b + 1] << 8) | fullEmbInt3[b + 2];
        const d8 = g << 3;
        dot += qScaled[d8]     * ((w >> 21) & 7);
        dot += qScaled[d8 + 1] * ((w >> 18) & 7);
        dot += qScaled[d8 + 2] * ((w >> 15) & 7);
        dot += qScaled[d8 + 3] * ((w >> 12) & 7);
        dot += qScaled[d8 + 4] * ((w >>  9) & 7);
        dot += qScaled[d8 + 5] * ((w >>  6) & 7);
        dot += qScaled[d8 + 6] * ((w >>  3) & 7);
        dot += qScaled[d8 + 7] * ( w        & 7);
      }
      out[idx] = dot;
    }
  } else if (fullEmbInt4) {
    // Rerank with int4 (packed nibbles)
    const qScaled = new Float32Array(FULL_DIMS);
    let qOffset = 0;
    for (let d = 0; d < FULL_DIMS; d++) {
      qScaled[d] = qvec[d] * fullRangeScale[d] / 15;
      qOffset += qvec[d] * fullRangeMin[d];
    }
    const halfDims = FULL_DIMS >> 1;
    for (let j = 0; j < k; j++) {
      const idx = topIdx[j];
      let dot = qOffset;
      const base = idx * halfDims;
      for (let d = 0; d < FULL_DIMS; d += 2) {
        const packed = fullEmbInt4[base + (d >> 1)];
        dot += qScaled[d] * (packed >> 4);
        dot += qScaled[d + 1] * (packed & 0x0F);
      }
      out[idx] = dot;
    }
  } else {
    // Rerank with int8
    const qScaled = new Float32Array(FULL_DIMS);
    let qOffset = 0;
    for (let d = 0; d < FULL_DIMS; d++) {
      qScaled[d] = qvec[d] * fullRangeScale[d] / 255;
      qOffset += qvec[d] * fullRangeMin[d];
    }
    for (let j = 0; j < k; j++) {
      const idx = topIdx[j];
      let dot = qOffset;
      const base = idx * FULL_DIMS;
      for (let d = 0; d < FULL_DIMS; d++) dot += qScaled[d] * fullEmbInt8[base + d];
      out[idx] = dot;
    }
  }
}

/**
 * In-place partial sort: rearranges arr[lo..hi] so that the top-k elements
 * (by descending scores[arr[i]]) are in arr[lo..lo+k-1].
 * Quickselect (Hoare partition).
 */
function nthElement(arr, scores, lo, hi, k) {
  while (lo < hi) {
    const pivotScore = scores[arr[lo + ((hi - lo) >> 1)]];
    let i = lo, j = hi;
    while (i <= j) {
      while (scores[arr[i]] > pivotScore) i++;
      while (scores[arr[j]] < pivotScore) j--;
      if (i <= j) { const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp; i++; j--; }
    }
    if (j - lo + 1 >= k) { hi = j; }
    else if (i - lo <= k) { k -= (i - lo); lo = i; }
    else break;
  }
}


// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async function search(query) {
  query = query.trim();
  if (!query) { $results.innerHTML = ""; $status.textContent = ""; return; }
  if (isRateLimited()) { $status.textContent = "Too many searches: please wait a moment."; return; }

  const count = wordEntries.length;

  // Instant preview removed: potion no longer loaded in full mode
  $status.textContent = "Searching\u2026";

  const qvec = await embedQuery(query);
  const scored = new Float32Array(count);

  const rerankReady = fullEmbInt3 || fullEmbInt4 || fullEmbInt8;

  if (fullReady) {
    if (fullBinaryReady && rerankReady) {
      // Best: binary first-pass + int4/int8 reranking
      scoreBinaryRerank(qvec, count, scored);
    } else if (fullBinaryReady) {
      // Binary-only fallback (reranking data not yet loaded)
      scoreHamming(qvec, fullEmbBinary, FULL_BINARY_BYTES, count, scored);
    } else if (fullEmbInt8) {
      // No binary data: pure int8 fallback
      scoreInt8(qvec, fullEmbInt8, fullRangeMin, fullRangeScale, FULL_DIMS, count, scored);
    }
  } else {
    scoreInt4(qvec, potionEmbInt4, potionRangeMin, potionRangeScale, LITE_DIMS, count, scored);
  }
  applyQualityWeights(scored, count);

  render(topK(scored, count), query);
}

/** Multiply scores by per-entry quality weights (if available). */
function applyQualityWeights(scored, count) {
  for (let i = 0; i < count; i++) {
    const q = wordEntries[i].q;
    if (q !== undefined) scored[i] *= q;
  }
}

/**
 * Canonical lemma lookup, sourced from the build-time forms_to_lemma.json map.
 * Replaces the old runtime suffix-stripping stemmer with evidence-based,
 * license-audited inflectional collapse (e.g. ran/running/runs -> run,
 * mice -> mouse). Negative/derivational prefixes are NEVER collapsed here
 * (unhappy stays unhappy); that policy lives in build_lemma_families.py.
 * Returns the canonical lemma for a word, or the word itself if none.
 */
let formsToLemma = new Map();

function canonicalLemma(word) {
  const w = word.toLowerCase().replace(/[^a-z'\-]/g, "");
  if (w.length < 2) return w;
  return formsToLemma.get(w) || w;
}

function topK(scored, count) {
  // Get top candidates from main pool
  const CANDIDATE_LIMIT = TOP_K * 4; // enough candidates for dedup
  const mainIndices = Array.from({ length: count }, (_, i) => i);
  mainIndices.sort((a, b) => scored[b] - scored[a]);

  // Build candidate list from top-scoring entries
  const combined = [];
  const mainLimit = Math.min(mainIndices.length, CANDIDATE_LIMIT);
  for (let i = 0; i < mainLimit; i++) {
    const idx = mainIndices[i];
    combined.push({ entry: wordEntries[idx], score: scored[idx] });
  }

  combined.sort((a, b) => b.score - a.score);

  const groups = new Map();   // primary word -> { ...entry, defs: [{d, p, score}], score }
  const order = [];           // insertion-order keys
  const wordToGroup = new Map();  // any word -> group key (for cross-variant merging)
  const lemmaToGroup = new Map(); // canonical lemma -> group key (inflectional merging)
  for (const item of combined) {
    const entry = item.entry;
    const itemScore = item.score;
    if (order.length >= TOP_K && !groups.has(entry.w[0].toLowerCase())) {
      // Also check if any variant word maps to an existing group
      let found = false;
      for (const w of entry.w) {
        if (wordToGroup.has(w.toLowerCase())) { found = true; break; }
      }
      if (!found) {
        // Check canonical-lemma grouping too
        for (const w of entry.w) {
          const s = canonicalLemma(w);
          if (s && lemmaToGroup.has(s)) { found = true; break; }
        }
      }
      if (!found) break;
    }
    // Find existing group via any shared word
    let groupKey = null;
    for (const w of entry.w) {
      if (wordToGroup.has(w.toLowerCase())) {
        groupKey = wordToGroup.get(w.toLowerCase());
        break;
      }
    }
    // If no exact match, try canonical-lemma grouping (running -> run, mice -> mouse)
    if (!groupKey) {
      for (const w of entry.w) {
        const s = canonicalLemma(w);
        if (s && lemmaToGroup.has(s)) {
          groupKey = lemmaToGroup.get(s);
          break;
        }
      }
    }
    if (groupKey && groups.has(groupKey)) {
      const g = groups.get(groupKey);
      if (!entry.h && g.defs.length < 3) {
        g.defs.push({ d: entry.d, p: entry.p, score: itemScore });
      }
      for (const w of entry.w) {
        if (!g.w.includes(w)) g.w.push(w);
        wordToGroup.set(w.toLowerCase(), groupKey);
        const s = canonicalLemma(w);
        if (s) lemmaToGroup.set(s, groupKey);
      }
      if (entry.s) for (const syn of entry.s) {
        if (!g.s.includes(syn)) g.s.push(syn);
      }
    } else {
      if (order.length >= TOP_K) continue;
      const primary = entry.w[0].toLowerCase();
      if (RESULT_EXCLUDE.has(primary)) continue;  // skip ultra-generic headwords as results
      const defs = entry.h ? [] : [{ d: entry.d, p: entry.p, score: itemScore }];
      const g = { w: [...entry.w], s: entry.s ? [...entry.s] : [], defs, score: itemScore };
      groups.set(primary, g);
      order.push(primary);
      for (const w of entry.w) {
        wordToGroup.set(w.toLowerCase(), primary);
        const s = canonicalLemma(w);
        if (s) lemmaToGroup.set(s, primary);
      }
    }
  }
  return order.map(k => groups.get(k)).filter(g => g.defs.length > 0);
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function escAttr(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function render(items, query) {
  if (items.length === 0) {
    $results.innerHTML = "";
    $status.textContent = "No matches: try rephrasing your description.";
    return;
  }
  $status.textContent = `Top ${items.length} matches`;
  const url = new URL(location.href);
  url.searchParams.set("q", query);
  history.replaceState(null, "", url);

  const maxScore = items[0].score;
  const shownWords = new Set(items.map(it => it.w[0].toLowerCase()));
  const MAX_ALT = 8;
  const renderCard = (it, i) => {
    const primary = it.w[0];
    // Morphological variants (w[] minus primary) - always show on separate line
    const altWords = it.w.slice(1).filter(w => !shownWords.has(w.toLowerCase()));
    const altHtml = altWords.length > 0
      ? `<div class="card-variants">${esc(altWords.slice(0, MAX_ALT).join(", "))}${altWords.length > MAX_ALT ? `, +${altWords.length - MAX_ALT}` : ""}</div>`
      : "";
    const pct = Math.round((it.score / maxScore) * 100);
    const defsHtml = it.defs.map((def, di) => {
      const posTag = def.p ? `<span class="card-pos" data-pos="${esc(def.p)}">${esc(def.p)}</span> ` : "";
      return `<p class="card-def">${di > 0 ? `<span class="def-num">${di + 1}.</span> ` : ""}${posTag}${esc(def.d)}</p>`;
    }).join("");
    // Synonyms from Moby Thesaurus
    const syns = (it.s || []).filter(w => !shownWords.has(w.toLowerCase()));
    const SHOW_SYNS = 5;
    const synHtml = syns.length > 0
      ? `<div class="card-synonyms"><span class="syn-label">Synonyms: </span><span class="syn-visible">${esc(syns.slice(0, SHOW_SYNS).join(", "))}</span>${syns.length > SHOW_SYNS ? `<span class="syn-hidden">, ${esc(syns.slice(SHOW_SYNS).join(", "))}</span><span class="syn-toggle">+${syns.length - SHOW_SYNS}</span>` : ""}</div>`
      : "";
    return `
      <article class="result-card" style="animation-delay:${i * 30}ms">
        <div class="card-head">
          <span class="card-word">${esc(primary)}</span>
          ${it.defs.length === 1 ? `<span class="card-pos" data-pos="${esc(it.defs[0].p)}">${esc(it.defs[0].p)}</span>` : ""}
          <button
            class="card-copy"
            type="button"
            data-copy-word="${escAttr(primary)}"
            aria-label="Copy ${escAttr(primary)}"
          >
            Copy
          </button>
        </div>
        ${altHtml}
        <div class="card-defs-wrap">${it.defs.length === 1 ? `<p class="card-def">${esc(it.defs[0].d)}</p>` : defsHtml}</div>
        ${synHtml}
        <div class="card-score">
          <div class="score-bar"><div class="score-fill" style="width:${pct}%"></div></div>
          <span class="score-pct">${pct}%</span>
        </div>
      </article>`;
  };

  const visible = items.slice(0, SHOW_K);
  const hidden = items.slice(SHOW_K);
  let html = visible.map(renderCard).join("");
  if (hidden.length > 0) {
    html += `<div class="more-results collapsed" id="moreResults">
      ${hidden.map((it, i) => renderCard(it, SHOW_K + i)).join("")}
    </div>
    <button class="show-more-btn" id="showMoreBtn" onclick="
      const more = document.getElementById('moreResults');
      more.classList.toggle('collapsed');
      this.textContent = this.textContent.includes('Show') ? 'Show fewer' : 'Show ${hidden.length} more matches';
      if (!more.classList.contains('collapsed')) more.querySelectorAll('.card-defs-wrap:not(.truncated)').forEach(el => { if (el.scrollHeight > el.clientHeight + 2) el.classList.add('truncated'); });
    ">Show ${hidden.length} more matches</button>`;
  }
  $results.innerHTML = html;
  // Mark overflowing def blocks as truncated (show fade + cursor)
  $results.querySelectorAll(".card-defs-wrap").forEach(el => {
    if (el.scrollHeight > el.clientHeight + 2) el.classList.add("truncated");
  });
}

document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy-word]");
  if (!button) return;

  const word = button.dataset.copyWord;
  if (!word) return;

  try {
    await navigator.clipboard.writeText(word);

    const oldText = button.textContent;
    button.textContent = "Copied";
    button.classList.add("copied");

    setTimeout(() => {
      button.textContent = oldText;
      button.classList.remove("copied");
    }, 1200);
  } catch {
    const oldText = button.textContent;
    button.textContent = "Failed";

    setTimeout(() => {
      button.textContent = oldText;
    }, 1200);
  }
});

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// ---------------------------------------------------------------------------
// Copy-event feedback (DISABLED: privacy concern: query field is free-text)
// Enable only after setting up FEEDBACK_ENDPOINT and adding input sanitization.
// ---------------------------------------------------------------------------

const FEEDBACK_ENDPOINT = "";   // set to a URL to enable; empty = disabled
const FEEDBACK_KEY = "wf_fb";
const FEEDBACK_MAX = 200;  // max buffered events before oldest are dropped

/*  --- disabled for now ---
$results.addEventListener("copy", () => {
  if (!FEEDBACK_ENDPOINT) return;   // no-op unless endpoint is configured
  const sel = window.getSelection();
  if (!sel || !sel.rangeCount) return;
  const card = sel.anchorNode?.parentElement?.closest?.(".result-card");
  if (!card) return;
  const word = card.querySelector(".card-word")?.textContent?.trim();
  const query = $input.value.trim();
  if (!word || !query) return;
  const day = new Date().toISOString().slice(0, 10);
  bufferFeedback({ q: query, w: word, d: day });
});
*/

// Click-to-expand definitions and synonyms
$results.addEventListener("click", (e) => {
  const defsWrap = e.target.closest(".card-defs-wrap");
  if (defsWrap) defsWrap.classList.toggle("expanded");
  const synToggle = e.target.closest(".syn-toggle");
  if (synToggle) {
    const synDiv = synToggle.closest(".card-synonyms");
    if (synDiv) synDiv.classList.toggle("expanded");
  }
});

function bufferFeedback(event) {
  if (!FEEDBACK_ENDPOINT) return;
  try {
    const buf = JSON.parse(localStorage.getItem(FEEDBACK_KEY) || "[]");
    buf.push(event);
    while (buf.length > FEEDBACK_MAX) buf.shift();
    localStorage.setItem(FEEDBACK_KEY, JSON.stringify(buf));
    flushFeedback();
  } catch { /* localStorage unavailable or full: silently skip */ }
}

function flushFeedback() {
  if (!FEEDBACK_ENDPOINT) return;
  try {
    const buf = JSON.parse(localStorage.getItem(FEEDBACK_KEY) || "[]");
    if (buf.length === 0) return;
    const ok = navigator.sendBeacon(
      FEEDBACK_ENDPOINT,
      new Blob([JSON.stringify(buf)], { type: "application/json" })
    );
    if (ok) localStorage.removeItem(FEEDBACK_KEY);
  } catch { /* silently skip */ }
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") flushFeedback();
});

// ---------------------------------------------------------------------------
// Rolling showcase
// ---------------------------------------------------------------------------

const SHOWCASE = [
  { q: "a feeling of longing for the past", w: "nostalgia" },
  { q: "fear of being forgotten", w: "athazagoraphobia" },
  { q: "the art of beautiful handwriting", w: "calligraphy" },
  { q: "a word that sounds like what it means", w: "onomatopoeia" },
  { q: "pleasure from someone else's misfortune", w: "schadenfreude" },
  { q: "unable to be put into words", w: "ineffable" },
  { q: "lasting only a very short time", w: "ephemeral" },
  { q: "a love of books", w: "bibliophilia" },
  { q: "wanderlust for the sea", w: "thalassophilia" },
  { q: "the smell of rain on dry earth", w: "petrichor" },
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
const $navLinks = document.getElementById("nav-links");
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
  });
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => { });
}

init().catch(err => {
  console.error(err);
  $loaderNote.textContent = `Error: ${err.message}. Please refresh the page.`;
  $loaderNote.style.color = "#DC2626";
});

const privacyBanner = document.getElementById("privacy-banner");
const privacyAccept = document.getElementById("privacy-banner-accept");
const privacyClose = document.getElementById("privacy-banner-close");

const PRIVACY_KEY = "wordfor_privacy_notice_dismissed";

if (privacyBanner && localStorage.getItem(PRIVACY_KEY) !== "1") {
  privacyBanner.hidden = false;
}

function dismissPrivacyBanner() {
  if (!privacyBanner) return;
  privacyBanner.hidden = true;
  localStorage.setItem(PRIVACY_KEY, "1");
}

privacyAccept?.addEventListener("click", dismissPrivacyBanner);
privacyClose?.addEventListener("click", dismissPrivacyBanner);