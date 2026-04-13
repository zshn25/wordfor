/**
 * WordFor: Reverse Dictionary
 * © 2025 Zeeshan Khan Suri (zshn25). Licensed under CC-BY-NC-ND-4.0.
 *
 * Default:       mdbr-leaf-mt (query) + mxbai-embed-large (defs) via Transformers.js
 *                Binary (ITQ) first-pass + int8 reranking for near-float32 quality at binary speed.
 * Mobile:        Same model, but pure binary ITQ scoring (no int8 download, saves ~65 MB).
 * Lite fallback: potion-base-8M via pure JS static embeddings (sub-1ms)
 *
 * Lite mode activates automatically if the full model fails to load,
 * or manually via ?mode=lite in the URL.
 * Binary-only scoring activates on mobile, or via ?scoring=binary.
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DATA_ROOT      = "data";
const TOP_K          = 30;
const SHOW_K         = 9;
const DEBOUNCE       = 400;
const RATE_LIMIT_MAX = 15;
const RATE_LIMIT_MS  = 10_000;

const FULL_MODEL_ID  = "onnx-community/mdbr-leaf-mt-ONNX";
const FULL_DIMS      = 384;
const LITE_DIMS      = 256;

let MODE = null;
let DIMS = null;
let fullReady = false;

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
// Device detection
// ---------------------------------------------------------------------------

function shouldUseLiteMode() {
  const params = new URLSearchParams(location.search);
  if (params.get("mode") === "lite") return true;
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
let potionEmbInt8;      // Uint8Array : int8 quantized potion embeddings
let potionRangeMin;     // Float32Array(256): per-dim min
let potionRangeScale;   // Float32Array(256): per-dim range

// Full-mode int8 embeddings (for reranking stage 2)
let fullEmbInt8;        // Uint8Array : int8 quantized full embeddings
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

// Wiktionary supplement (CC-BY-SA, loaded lazily)
let wikiEntries;         // Array: words_wiki.json entries
let wikiEmbInt8;         // Uint8Array : int8 wiki embeddings (full-mode)
let wikiRangeMin;        // Float32Array(384): per-dim min
let wikiRangeScale;      // Float32Array(384): per-dim range
let wikiEmbBinary;       // Uint8Array : binary wiki embeddings
let wikiPotionInt8;      // Uint8Array : potion wiki embeddings
let wikiPotionRangeMin;  // Float32Array(256)
let wikiPotionRangeScale;// Float32Array(256)
let wikiReady = false;

// ---------------------------------------------------------------------------
// Transformers.js loader
// ---------------------------------------------------------------------------

async function loadTransformers() {
  const { AutoModel, AutoTokenizer, env } = await import(
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4"
  );
  env.allowLocalModels  = true;
  env.allowRemoteModels = false;

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
  const res = await fetch(`${DATA_ROOT}/words.json`);
  setProgress("words", 50);
  wordEntries = await res.json();
  setProgress("words", 100);
}

async function loadPotionData() {
  addProgressRow("matrix", "Embedding model (~15 MB)");
  addProgressRow("emb",    "Dictionary vectors (~43 MB)");
  $loaderNote.textContent = $loaderNote.textContent || "First visit downloads ~76 MB (cached for future visits)";

  // Shared data needed regardless of WASM or JS model
  const embPromise = fetchWithProgress(`${DATA_ROOT}/embeddings_potion_int8.bin`, "emb")
    .then(buf => { potionEmbInt8 = new Uint8Array(buf); });

  const rangesPromise = fetch(`${DATA_ROOT}/embeddings_potion_ranges.bin`)
    .then(r => r.arrayBuffer()).then(buf => {
      const data = new Float32Array(buf);
      potionRangeMin   = data.subarray(0, LITE_DIMS);
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
    const vocabPromise = fetch(`${DATA_ROOT}/vocab.txt`).then(async r => {
      const lines = (await r.text()).split(/\r?\n/);
      const map = new Map();
      for (let i = 0; i < lines.length; i++) if (lines[i] !== "") map.set(lines[i], i);
      return map;
    });
    const matrixPromise = fetchWithProgress(`${DATA_ROOT}/potion_matrix.bin`, "matrix")
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
  addProgressRow("tf",    "AI model (~22 MB)");
  if (!BINARY_ONLY) {
    addProgressRow("femb",  "Dictionary vectors (~73 MB)");
  } else {
    addProgressRow("femb",  "Dictionary vectors (~9 MB)");
  }

  const tfPromise = loadTransformers();

  // Binary embeddings (~8 MB): primary scoring (fast Hamming)
  const binaryPromise = fetch(`${DATA_ROOT}/embeddings_binary.bin`)
    .then(r => r.ok ? r.arrayBuffer() : null)
    .then(buf => buf ? new Uint8Array(buf) : null)
    .catch(() => null);

  // ITQ calibration (~577 KB): rotation matrix for binary scoring
  const itqPromise = fetch(`${DATA_ROOT}/embeddings_itq.bin`)
    .then(r => r.ok ? r.arrayBuffer() : null)
    .catch(() => null);

  // Int8 embeddings (~65 MB): used for reranking binary candidates
  // Skipped in binary-only mode (mobile) to save bandwidth
  let int8Promise, rangesPromise;
  if (!BINARY_ONLY) {
    int8Promise = fetchWithProgress(`${DATA_ROOT}/embeddings_int8.bin`, "femb")
      .then(buf => new Uint8Array(buf));
    rangesPromise = fetch(`${DATA_ROOT}/embeddings_ranges.bin`)
      .then(r => r.arrayBuffer()).then(buf => new Float32Array(buf));
  } else {
    setProgress("femb", 100);
  }

  const { AutoModel, AutoTokenizer, device } = await tfPromise;
  setProgress("tf", 40);

  // Try q8 first; on iOS Safari WASM OOM, fall back to q4f16 with reduced memory
  let tokenizer, model;
  tokenizer = await AutoTokenizer.from_pretrained(FULL_MODEL_ID);
  try {
    model = await timeout(BINARY_ONLY ? 30_000 : 90_000, AutoModel.from_pretrained(FULL_MODEL_ID, {
      dtype: "q8", device,
      session_options: { enableCpuMemArena: false },
    }));
  } catch (e) {
    console.warn("q8 model failed, trying q4f16 with reduced memory:", e.message);
    model = await timeout(90_000, AutoModel.from_pretrained(FULL_MODEL_ID, {
      dtype: "q4f16", device,
      session_options: { enableCpuMemArena: false },
    }));
  }
  fullTokenizer = tokenizer;
  fullModel = model;
  setProgress("tf", 80);

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

  // Load int8 embeddings (for reranking): skipped in binary-only mode
  if (!BINARY_ONLY) {
    try {
      const [int8Data, rangesData] = await Promise.all([int8Promise, rangesPromise]);
      fullEmbInt8    = int8Data;
      fullRangeMin   = rangesData.subarray(0, FULL_DIMS);
      fullRangeScale = rangesData.subarray(FULL_DIMS, FULL_DIMS * 2);
    } catch (e) {
      console.warn("Int8 embeddings failed, using binary-only scoring:", e.message);
    }
  }

  fullReady = true;
  DIMS = FULL_DIMS;
}

// ---------------------------------------------------------------------------
// Wiktionary supplement loader (lazy, after main app is ready)
// ---------------------------------------------------------------------------

async function loadWikiData() {
  try {
    // Load word entries
    const wordsRes = await fetch(`${DATA_ROOT}/words_wiki.json`);
    if (!wordsRes.ok) return;
    wikiEntries = await wordsRes.json();
    if (!wikiEntries || wikiEntries.length === 0) return;

    const wikiCount = wikiEntries.length;

    // Load embeddings in parallel based on mode
    const promises = [];

    if (MODE === "full" || fullReady) {
      // Full-mode wiki: binary for first pass, int8 for reranking (skip int8 on BINARY_ONLY)
      promises.push(
        fetch(`${DATA_ROOT}/embeddings_wiki_binary.bin`)
          .then(r => r.ok ? r.arrayBuffer() : null)
          .then(buf => { if (buf) wikiEmbBinary = new Uint8Array(buf); })
          .catch(() => {})
      );
      if (!BINARY_ONLY) {
        promises.push(
          fetch(`${DATA_ROOT}/embeddings_wiki_int8.bin`)
            .then(r => r.ok ? r.arrayBuffer() : null)
            .then(buf => { if (buf) wikiEmbInt8 = new Uint8Array(buf); })
            .catch(() => {}),
          fetch(`${DATA_ROOT}/embeddings_wiki_ranges.bin`)
            .then(r => r.ok ? r.arrayBuffer() : null)
            .then(buf => {
              if (buf) {
                const data = new Float32Array(buf);
                wikiRangeMin   = data.subarray(0, FULL_DIMS);
                wikiRangeScale = data.subarray(FULL_DIMS, FULL_DIMS * 2);
              }
            })
            .catch(() => {})
        );
      }
    }

    // Potion wiki embeddings (only for lite mode)
    if (MODE === "lite") {
      promises.push(
        fetch(`${DATA_ROOT}/embeddings_wiki_potion_int8.bin`)
          .then(r => r.ok ? r.arrayBuffer() : null)
          .then(buf => { if (buf) wikiPotionInt8 = new Uint8Array(buf); })
          .catch(() => {}),
        fetch(`${DATA_ROOT}/embeddings_wiki_potion_ranges.bin`)
          .then(r => r.ok ? r.arrayBuffer() : null)
          .then(buf => {
            if (buf) {
              const data = new Float32Array(buf);
              wikiPotionRangeMin   = data.subarray(0, LITE_DIMS);
              wikiPotionRangeScale = data.subarray(LITE_DIMS, LITE_DIMS * 2);
            }
          })
          .catch(() => {})
      );
    }

    await Promise.all(promises);
    wikiReady = true;
    console.log(`Wiki supplement loaded: ${wikiCount} entries`);
  } catch (e) {
    console.warn("Wiki supplement failed to load:", e.message);
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
      : "First visit downloads ~95 MB (cached for future visits)";
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
  $input.focus();
  startShowcase();
  showModeBadge();

  if (liteFallback) showLiteFallbackBanner();

  // Deep link
  const q = new URLSearchParams(location.search).get("q");
  if (q) { $input.value = q; search(q); }

  // Lazy-load Wiktionary supplement (non-blocking, enhances results silently)
  loadWikiData().catch(e => console.warn("Wiki load:", e.message));
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
    <span>The full AI model couldn\u2019t load on this device. Running in <strong>Lite mode</strong> \u2014 results may be less accurate. For best quality, try on a desktop browser.</span>
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
 * Two-stage scoring: binary Hamming first-pass, then int8 dot product reranking.
 * Returns float32 scores for all entries (non-candidates get -Infinity).
 */
function scoreBinaryRerank(qvec, count, out) {
  // Stage 1: Binary Hamming over all entries
  const hamming = new Float32Array(count);
  scoreHamming(qvec, fullEmbBinary, FULL_BINARY_BYTES, count, hamming);

  // Stage 2: Find top RERANK_K candidates by Hamming score
  const topIdx = new Int32Array(count);
  for (let i = 0; i < count; i++) topIdx[i] = i;
  // Partial sort: partition around RERANK_K-th element
  const k = Math.min(RERANK_K, count);
  nthElement(topIdx, hamming, 0, count - 1, k);

  // Rerank top candidates with int8 dot product
  const qScaled = new Float32Array(FULL_DIMS);
  let qOffset = 0;
  for (let d = 0; d < FULL_DIMS; d++) {
    qScaled[d] = qvec[d] * fullRangeScale[d] / 255;
    qOffset += qvec[d] * fullRangeMin[d];
  }

  // Fill all with -Infinity, then overwrite reranked candidates
  for (let i = 0; i < count; i++) out[i] = -Infinity;
  for (let j = 0; j < k; j++) {
    const idx = topIdx[j];
    let dot = qOffset;
    const base = idx * FULL_DIMS;
    for (let d = 0; d < FULL_DIMS; d++) dot += qScaled[d] * fullEmbInt8[base + d];
    out[idx] = dot;
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

/**
 * Two-stage scoring for wiki entries: binary Hamming + int8 reranking.
 */
function scoreBinaryRerankWiki(qvec, count, out) {
  const hamming = new Float32Array(count);
  scoreHamming(qvec, wikiEmbBinary, FULL_BINARY_BYTES, count, hamming);

  const topIdx = new Int32Array(count);
  for (let i = 0; i < count; i++) topIdx[i] = i;
  const k = Math.min(RERANK_K, count);
  nthElement(topIdx, hamming, 0, count - 1, k);

  const qScaled = new Float32Array(FULL_DIMS);
  let qOffset = 0;
  for (let d = 0; d < FULL_DIMS; d++) {
    qScaled[d] = qvec[d] * wikiRangeScale[d] / 255;
    qOffset += qvec[d] * wikiRangeMin[d];
  }

  for (let i = 0; i < count; i++) out[i] = -Infinity;
  for (let j = 0; j < k; j++) {
    const idx = topIdx[j];
    let dot = qOffset;
    const base = idx * FULL_DIMS;
    for (let d = 0; d < FULL_DIMS; d++) dot += qScaled[d] * wikiEmbInt8[base + d];
    out[idx] = dot;
  }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async function search(query) {
  query = query.trim();
  if (!query) { $results.innerHTML = ""; $status.textContent = ""; return; }
  if (isRateLimited()) { $status.textContent = "Too many searches: please wait a moment."; return; }

  const count   = wordEntries.length;

  // Instant preview removed: potion no longer loaded in full mode
  $status.textContent = "Searching\u2026";

  const qvec   = await embedQuery(query);
  const scored = new Float32Array(count);

  if (fullReady) {
    if (fullBinaryReady && fullEmbInt8) {
      // Best: binary first-pass + int8 reranking (near float32 quality, binary speed)
      scoreBinaryRerank(qvec, count, scored);
    } else if (fullBinaryReady) {
      // Binary-only fallback (int8 failed to load)
      scoreHamming(qvec, fullEmbBinary, FULL_BINARY_BYTES, count, scored);
    } else {
      // No binary data: pure int8 fallback
      scoreInt8(qvec, fullEmbInt8, fullRangeMin, fullRangeScale, FULL_DIMS, count, scored);
    }
  } else {
    scoreInt8(qvec, potionEmbInt8, potionRangeMin, potionRangeScale, LITE_DIMS, count, scored);
  }
  applyQualityWeights(scored, count);

  // Score wiki entries if loaded
  let wikiScored = null;
  if (wikiReady && wikiEntries && wikiEntries.length > 0) {
    const wc = wikiEntries.length;
    wikiScored = new Float32Array(wc);
    if (fullReady) {
      if (fullBinaryReady && wikiEmbBinary && wikiEmbInt8 && wikiRangeMin) {
        // Binary rerank for wiki
        scoreBinaryRerankWiki(qvec, wc, wikiScored);
      } else if (wikiEmbBinary && fullBinaryReady) {
        scoreHamming(qvec, wikiEmbBinary, FULL_BINARY_BYTES, wc, wikiScored);
      } else if (wikiEmbInt8 && wikiRangeMin) {
        scoreInt8(qvec, wikiEmbInt8, wikiRangeMin, wikiRangeScale, FULL_DIMS, wc, wikiScored);
      }
    } else if (wikiPotionInt8 && wikiPotionRangeMin && potionModel) {
      const potionVec = potionModel.encode(query);
      scoreInt8(potionVec, wikiPotionInt8, wikiPotionRangeMin, wikiPotionRangeScale, LITE_DIMS, wc, wikiScored);
    }
  }

  render(topK(scored, count, wikiScored), query);
}

/** Multiply scores by per-entry quality weights (if available). */
function applyQualityWeights(scored, count) {
  for (let i = 0; i < count; i++) {
    const q = wordEntries[i].q;
    if (q !== undefined) scored[i] *= q;
  }
}

/**
 * Simple suffix-stripping stemmer for grouping morphological variants.
 * Returns a stem if the word is long enough (5+ chars after stripping),
 * or null if the word is too short to safely stem.
 */
function stemWord(word) {
  const w = word.toLowerCase().replace(/[^a-z]/g, "");
  if (w.length < 7) return null;
  // Strip common suffixes (longest first to avoid partial matches)
  const suffixes = [
    "ically", "ation", "ition", "phobic", "phobia", "mania",
    "ness", "ment", "ible", "able",
    "ical", "ious", "eous",
    "ist", "ism", "ous", "ive", "ful", "ing", "ant", "ent", "ial",
    "ic", "al", "ly", "er", "ed", "ia",
  ];
  for (const suf of suffixes) {
    if (w.endsWith(suf) && w.length - suf.length >= 5) {
      return w.slice(0, w.length - suf.length);
    }
  }
  return null;
}

function topK(scored, count, wikiScored) {
  // Get top candidates from main pool
  const CANDIDATE_LIMIT = TOP_K * 4; // enough candidates for merging
  const mainIndices = Array.from({ length: count }, (_, i) => i);
  mainIndices.sort((a, b) => scored[b] - scored[a]);

  // Build combined candidate list from main + wiki top candidates
  const combined = [];
  const mainLimit = Math.min(mainIndices.length, CANDIDATE_LIMIT);
  for (let i = 0; i < mainLimit; i++) {
    const idx = mainIndices[i];
    combined.push({ entry: wordEntries[idx], score: scored[idx] });
  }

  if (wikiScored && wikiEntries && wikiEntries.length > 0) {
    const wikiIndices = Array.from({ length: wikiEntries.length }, (_, i) => i);
    wikiIndices.sort((a, b) => wikiScored[b] - wikiScored[a]);
    const wikiLimit = Math.min(wikiIndices.length, CANDIDATE_LIMIT);
    for (let i = 0; i < wikiLimit; i++) {
      const idx = wikiIndices[i];
      combined.push({ entry: wikiEntries[idx], score: wikiScored[idx] });
    }
  }

  combined.sort((a, b) => b.score - a.score);

  const groups = new Map();   // primary word -> { ...entry, defs: [{d, p, score}], score }
  const order = [];           // insertion-order keys
  const wordToGroup = new Map();  // any word -> group key (for cross-variant merging)
  const stemToGroup = new Map();  // stem -> group key (for morphological merging)
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
        // Check stem-based grouping too
        for (const w of entry.w) {
          const s = stemWord(w);
          if (s && stemToGroup.has(s)) { found = true; break; }
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
    // If no exact match, try stem-based grouping (bibliophilic -> bibliophilist)
    if (!groupKey) {
      for (const w of entry.w) {
        const s = stemWord(w);
        if (s && stemToGroup.has(s)) {
          groupKey = stemToGroup.get(s);
          break;
        }
      }
    }
    if (groupKey && groups.has(groupKey)) {
      const g = groups.get(groupKey);
      if (g.defs.length < 3) {
        g.defs.push({ d: entry.d, p: entry.p, score: itemScore });
        for (const w of entry.w) {
          if (!g.w.includes(w)) g.w.push(w);
          wordToGroup.set(w.toLowerCase(), groupKey);
          const s = stemWord(w);
          if (s) stemToGroup.set(s, groupKey);
        }
      }
    } else {
      if (order.length >= TOP_K) continue;
      const primary = entry.w[0].toLowerCase();
      const g = { w: [...entry.w], defs: [{ d: entry.d, p: entry.p, score: itemScore }], score: itemScore };
      groups.set(primary, g);
      order.push(primary);
      for (const w of entry.w) {
        wordToGroup.set(w.toLowerCase(), primary);
        const s = stemWord(w);
        if (s) stemToGroup.set(s, primary);
      }
    }
  }
  return order.map(k => groups.get(k));
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

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
    // Filter out synonyms that duplicate other result headwords
    const altWords = it.w.slice(1).filter(w => !shownWords.has(w.toLowerCase()));
    if (altWords.length === 0) {
      var altHtml = "";
    } else {
      const visiblePart = altWords.slice(0, MAX_ALT).join(", ");
      const hiddenPart = altWords.slice(MAX_ALT).join(", ");
      altHtml = `<span class="card-words-alt">`
        + `<span class="alt-visible">${esc(visiblePart)}</span>`
        + (hiddenPart ? `<span class="alt-hidden">, ${esc(hiddenPart)}</span>` : "")
        + `</span>`;
    }
    const pct = Math.round((it.score / maxScore) * 100);
    const defsHtml = it.defs.map((def, di) => {
      const posTag = def.p ? `<span class="card-pos" data-pos="${esc(def.p)}">${esc(def.p)}</span> ` : "";
      return `<p class="card-def">${di > 0 ? `<span class="def-num">${di + 1}.</span> ` : ""}${posTag}${esc(def.d)}</p>`;
    }).join("");
    return `
      <article class="result-card" style="animation-delay:${i * 30}ms">
        <div class="card-head">
          <span class="card-word">${esc(primary)}</span>
          ${it.defs.length === 1 ? `<span class="card-pos" data-pos="${esc(it.defs[0].p)}">${esc(it.defs[0].p)}</span>` : ""}
          ${altHtml}
        </div>
        ${it.defs.length === 1 ? `<p class="card-def">${esc(it.defs[0].d)}</p>` : defsHtml}
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
      document.getElementById('moreResults').classList.toggle('collapsed');
      this.textContent = this.textContent.includes('Show') ? 'Show fewer' : 'Show ${hidden.length} more matches';
    ">Show ${hidden.length} more matches</button>`;
  }
  $results.innerHTML = html;
}

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
const FEEDBACK_KEY      = "wf_fb";
const FEEDBACK_MAX      = 200;  // max buffered events before oldest are dropped

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

// Click-to-expand synonyms
$results.addEventListener("click", (e) => {
  const alt = e.target.closest(".card-words-alt");
  if (alt) alt.classList.toggle("expanded");
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
