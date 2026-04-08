/**
 * WordFor — Reverse Dictionary
 *
 * Runs a sentence-embedding model (bge-small-en-v1.5) in the browser via
 * Transformers.js / ONNX Runtime Web (WASM backend).  Pre-computed WordNet
 * definition embeddings are loaded from a static binary file; cosine
 * similarity search is performed entirely client-side.
 *
 * Scoring: hybrid of cosine similarity + keyword overlap to boost
 * results whose definitions share specific words with the query.
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const MODEL_ID       = "Xenova/bge-small-en-v1.5";
const DATA_ROOT      = "data";
const DIMS           = 384;
const TOP_K          = 30;   // top results before dedup
const DEBOUNCE       = 400;  // ms after last keystroke

// Hybrid scoring weights (tuned for reverse-dictionary relevance)
const W_COSINE       = 0.70; // weight for embedding cosine similarity
const W_KEYWORD      = 0.30; // weight for keyword overlap score

const QUERY_PREFIX   = "Represent this sentence: ";

// Stop words to ignore in keyword matching
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

function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = f16LUT[u16[i]];
  return out;
}

// ---------------------------------------------------------------------------
// DOM references
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
// Keyword scoring helpers
// ---------------------------------------------------------------------------

/** Tokenise a string into lower-case content words, dropping stop words. */
function contentWords(text) {
  return text.toLowerCase().match(/[a-z]{2,}/g)?.filter(w => !STOP_WORDS.has(w)) ?? [];
}

/**
 * Build per-word sets of content tokens from all definitions, once after load.
 * This avoids re-tokenising 110k definitions on every search.
 */
let defTokens; // Array<Set<string>>

function buildDefTokenIndex() {
  defTokens = new Array(wordEntries.length);
  for (let i = 0; i < wordEntries.length; i++) {
    const entry = wordEntries[i];
    // Combine definition text + the words themselves for matching
    const combined = entry.d + " " + entry.w.join(" ");
    defTokens[i] = new Set(contentWords(combined));
  }
}

/**
 * Keyword overlap score: fraction of query content-words found in definition.
 * Returns 0..1.
 */
function keywordScore(queryTokens, defTokenSet) {
  if (queryTokens.length === 0) return 0;
  let hits = 0;
  for (const t of queryTokens) {
    if (defTokenSet.has(t)) hits++;
  }
  return hits / queryTokens.length;
}

// ---------------------------------------------------------------------------
// Load model + data
// ---------------------------------------------------------------------------

async function loadTransformers() {
  const { pipeline, env } = await import(
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3"
  );
  env.backends.onnx.wasm.numThreads = navigator.hardwareConcurrency || 4;
  return pipeline;
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

let extractor;
let wordEntries;
let embMatrix;

async function init() {
  addProgressRow("model", "AI model (~34 MB)");
  addProgressRow("emb",   "Dictionary vectors (~90 MB)");
  addProgressRow("words", "Word list (~8 MB)");

  const pipelineFactory = loadTransformers();

  const embPromise = fetchWithProgress(`${DATA_ROOT}/embeddings.bin`, "emb").then(buf => {
    const u16 = new Uint16Array(buf);
    embMatrix = f16ToF32(u16);
  });

  const wordsPromise = fetch(`${DATA_ROOT}/words.json`).then(async r => {
    wordEntries = await r.json();
    setProgress("words", 100);
  });

  const makePipeline = await pipelineFactory;
  extractor = await makePipeline("feature-extraction", MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => {
      if (p.status === "progress" && p.progress != null) {
        setProgress("model", p.progress);
      }
      if (p.status === "done") {
        setProgress("model", 100);
      }
    },
  });

  await Promise.all([embPromise, wordsPromise]);

  // Build the keyword index for hybrid search
  buildDefTokenIndex();

  const actualCount = embMatrix.length / DIMS;
  if (actualCount !== wordEntries.length) {
    console.warn(`Count mismatch: embeddings=${actualCount}, words=${wordEntries.length}`);
  }

  $loader.classList.add("done");
  $app.classList.remove("hidden");
  $input.focus();
  startShowcase();

  const params = new URLSearchParams(location.search);
  if (params.get("q")) {
    $input.value = params.get("q");
    search($input.value);
  }
}

// ---------------------------------------------------------------------------
// Search  (hybrid: cosine similarity + keyword overlap)
// ---------------------------------------------------------------------------

async function search(query) {
  query = query.trim();
  if (!query) { $results.innerHTML = ""; $status.textContent = ""; return; }

  $status.textContent = "Searching…";

  // Embed the query
  const out = await extractor(QUERY_PREFIX + query, {
    pooling: "cls",
    normalize: true,
  });
  const qvec = out.data;

  // Tokenise query for keyword matching
  const qTokens = contentWords(query);

  // Score every entry: hybrid = W_COSINE * cosine + W_KEYWORD * keyword
  const count = wordEntries.length;
  const scored = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    // Cosine similarity (embeddings are pre-normalised)
    let dot = 0;
    const base = i * DIMS;
    for (let d = 0; d < DIMS; d++) dot += qvec[d] * embMatrix[base + d];

    // Keyword overlap
    const kw = keywordScore(qTokens, defTokens[i]);

    scored[i] = W_COSINE * dot + W_KEYWORD * kw;
  }

  // Top-K indices
  const indices = Array.from({ length: count }, (_, i) => i);
  indices.sort((a, b) => scored[b] - scored[a]);

  // Deduplicate by primary word
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

  render(deduped, query);
}

// ---------------------------------------------------------------------------
// Render results
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
// Rolling showcase  (cycles through example query→word pairs)
// ---------------------------------------------------------------------------

const SHOWCASE_ITEMS = [
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

  const item = SHOWCASE_ITEMS[showcaseIdx % SHOWCASE_ITEMS.length];
  showcaseIdx++;

  const $row = $showcase.querySelector(".showcase-row");
  $row.style.animation = "none";
  // Force reflow to restart animation
  void $row.offsetWidth;

  $row.querySelector(".showcase-query").textContent = `"${item.q}"`;
  $row.querySelector(".showcase-word").textContent = item.w;
  $row.style.animation = "showcaseFade .6s ease both";
}

function startShowcase() {
  cycleShowcase();
  showcaseInterval = setInterval(cycleShowcase, 3500);
}

// Pause showcase once user starts searching
function stopShowcase() {
  if (showcaseInterval) { clearInterval(showcaseInterval); showcaseInterval = null; }
  const $showcase = document.getElementById("showcase");
  if ($showcase) $showcase.style.display = "none";
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

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

init().catch(err => {
  console.error(err);
  $loaderNote.textContent = `Error: ${err.message}. Please refresh the page.`;
  $loaderNote.style.color = "#DC2626";
});
