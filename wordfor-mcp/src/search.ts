/**
 * search.ts -- lexical reverse-dictionary scoring over the public-domain word data.
 *
 * This is the "fast" lexical engine: token overlap between the query and each entry's
 * definition / synonyms / headword, with quality and exact-match boosts. It needs no
 * model download and runs anywhere Node runs.
 *
 * NOTE on "best" mode: the website's semantic ranking uses on-device embeddings
 * (mdbr-leaf-mt + quantized vectors). Reproducing that here would require shipping the
 * model + embeddings. The MCP server therefore exposes a strong lexical engine by
 * default and documents semantic mode as an optional extension (see docs/mcp.md).
 * `mode: "best"` currently applies extra phrase/synonym weighting on top of lexical.
 */
import { getData, WordEntry, SourceLabel } from "./data.js";

export interface RankedWord {
  word: string;
  score: number;            // 0..1 normalized
  pos?: string;
  definition: string;
  explanation: string;
  lemma_family: string[];
  sources: SourceLabel[];
}

const STOP = new Set([
  "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "be",
  "that", "this", "with", "as", "it", "something", "someone", "word", "feeling",
  "thing", "when", "you", "your", "who", "which", "what",
]);

function tokenize(s: string): string[] {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s'-]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP.has(t));
}

function scoreEntry(e: WordEntry, qTokens: string[], qSet: Set<string>, best: boolean): number {
  const def = (e.d || "").toLowerCase();
  const defTokens = tokenize(def);
  const synTokens = (e.s || []).join(" ").toLowerCase();
  const headword = (e.w?.[0] || "").toLowerCase();

  let overlap = 0;
  for (const t of new Set(defTokens)) if (qSet.has(t)) overlap++;
  // synonym overlap (weighted lower)
  let synOverlap = 0;
  for (const t of qSet) if (synTokens.includes(t)) synOverlap++;

  if (overlap === 0 && synOverlap === 0 && !qSet.has(headword)) return 0;

  const denom = Math.max(qTokens.length, 1);
  let score = overlap / denom + 0.4 * (synOverlap / denom);

  // exact headword match (e.g. query literally names the word)
  if (qSet.has(headword)) score += 0.5;

  // quality prior (q ~ around 1); keep its influence modest
  const q = typeof e.q === "number" ? e.q : 1;
  score *= 0.8 + 0.2 * Math.min(2, q);

  if (best) {
    // reward phrase contiguity: consecutive query tokens appearing in the definition
    for (let i = 0; i + 1 < qTokens.length; i++) {
      if (def.includes(qTokens[i] + " " + qTokens[i + 1])) score += 0.15;
    }
  }
  return score;
}

export function reverseLookup(query: string, limit = 10, mode: "fast" | "best" = "fast"): RankedWord[] {
  const data = getData();
  const qTokens = tokenize(query);
  const qSet = new Set(qTokens);
  const best = mode === "best";

  // Score, keeping the best entry per lemma family (dedupe inflections).
  const bestByLemma = new Map<string, { e: WordEntry; raw: number }>();
  let maxRaw = 0;
  for (const e of data.entries) {
    const raw = scoreEntry(e, qTokens, qSet, best);
    if (raw <= 0) continue;
    if (raw > maxRaw) maxRaw = raw;
    const lemma = data.canonicalLemma(e.w?.[0] || "");
    const cur = bestByLemma.get(lemma);
    if (!cur || raw > cur.raw) bestByLemma.set(lemma, { e, raw });
  }

  const ranked = [...bestByLemma.entries()]
    .sort((a, b) => b[1].raw - a[1].raw)
    .slice(0, limit)
    .map(([lemma, { e, raw }]) => toRanked(lemma, e, raw, maxRaw, qSet));
  return ranked;
}

export function searchWord(query: string, limit = 10): RankedWord[] {
  // Headword / prefix search (a "word finder" by spelling rather than meaning).
  const data = getData();
  const q = query.toLowerCase().trim();
  const seen = new Set<string>();
  const hits: { e: WordEntry; raw: number }[] = [];
  for (const e of data.entries) {
    const hw = (e.w?.[0] || "").toLowerCase();
    if (!hw || seen.has(hw)) continue;
    let raw = 0;
    if (hw === q) raw = 1;
    else if (hw.startsWith(q)) raw = 0.7;
    else if (hw.includes(q)) raw = 0.4;
    else if ((e.w || []).some((w) => w.toLowerCase() === q)) raw = 0.6;
    if (raw > 0) {
      seen.add(hw);
      hits.push({ e, raw: raw * (0.8 + 0.2 * Math.min(2, e.q ?? 1)) });
    }
  }
  return hits
    .sort((a, b) => b.raw - a.raw)
    .slice(0, limit)
    .map(({ e, raw }) => toRanked(data.canonicalLemma(e.w?.[0] || ""), e, raw, 1, new Set()));
}

function toRanked(lemma: string, e: WordEntry, raw: number, maxRaw: number, qSet: Set<string>): RankedWord {
  const data = getData();
  const sources = data.decodeSources(e.src || 0);
  const matched = tokenize(e.d || "").filter((t) => qSet.has(t));
  const explanation = matched.length
    ? `Definition matches on: ${[...new Set(matched)].slice(0, 6).join(", ")}.`
    : `Closest lexical match for the description.`;
  // lemma family = all headwords that canonicalize to this lemma (capped)
  const family = new Set<string>([e.w?.[0] || lemma]);
  for (const [form, lem] of data.formsToLemma) {
    if (lem === lemma) family.add(form);
    if (family.size >= 8) break;
  }
  return {
    word: e.w?.[0] || lemma,
    score: maxRaw > 0 ? +(raw / maxRaw).toFixed(4) : +raw.toFixed(4),
    pos: e.p,
    definition: e.d,
    explanation,
    lemma_family: [...family],
    sources,
  };
}

export function explainRanking(query: string, word: string): {
  query: string;
  word: string;
  found: boolean;
  signals: Record<string, unknown>;
  note: string;
} {
  const data = getData();
  const entries = data.entriesForHeadword(word);
  if (entries.length === 0) {
    return {
      query, word, found: false, signals: {},
      note: `'${word}' is not a headword in the public-domain core.`,
    };
  }
  const qSet = new Set(tokenize(query));
  const e = entries.reduce((a, b) => ((b.q ?? 1) > (a.q ?? 1) ? b : a));
  const defTokens = tokenize(e.d || "");
  const overlap = [...new Set(defTokens)].filter((t) => qSet.has(t));
  return {
    query,
    word,
    found: true,
    signals: {
      lexical_overlap_terms: overlap,
      lexical_overlap_count: overlap.length,
      part_of_speech: e.p ?? null,
      quality_prior: e.q ?? null,
      lemma: data.canonicalLemma(word),
      sources: data.decodeSources(e.src || 0).map((s) => `${s.name} (${s.license})`),
    },
    note:
      "Ranking blends lexical overlap (shown), semantic similarity (computed on-device " +
      "in the web app, not in this server), source confidence, lemma grouping, and reranking.",
  };
}

export function getWordFamily(word: string): { lemma: string; forms: string[]; note: string } {
  const data = getData();
  const lemma = data.canonicalLemma(word);
  const forms = new Set<string>([lemma]);
  for (const [form, lem] of data.formsToLemma) if (lem === lemma) forms.add(form);
  return {
    lemma,
    forms: [...forms],
    note:
      "Inflected forms only (e.g. run/running/ran). Prefix-changed words (unhappy) and " +
      "derivations (happiness) are deliberately NOT collapsed into the lemma.",
  };
}

export function getSources(word: string): { word: string; found: boolean; sources: SourceLabel[] } {
  const data = getData();
  const entries = data.entriesForHeadword(word);
  if (entries.length === 0) return { word, found: false, sources: [] };
  let mask = 0;
  for (const e of entries) mask |= e.src || 0;
  return { word, found: true, sources: data.decodeSources(mask) };
}
