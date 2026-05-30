/**
 * data.ts -- load WordFor's public-domain word data and expose license-safe lookups.
 *
 * Reads the same artifacts the website ships:
 *   - data/words.json            (entries: w[], d, p, s[], q, src bitmask)
 *   - data/source_manifest.json  (bit_index + per-source license metadata)
 *   - data/forms_to_lemma.json   (inflected form -> canonical lemma)
 *
 * Only LICENSE-SAFE source *labels* are ever emitted (names + license class). No
 * restricted dictionary text is exposed beyond the public-domain / openly-licensed
 * definitions already published on wordfor.xyz.
 */
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

export interface WordEntry {
  w: string[];      // word + variants (w[0] is the headword)
  d: string;        // definition (public-domain / openly licensed)
  p?: string;       // part of speech
  s?: string[];     // synonyms
  q?: number;       // quality score
  src: number;      // source bitmask
}

export interface SourceLabel {
  id: string;
  name: string;
  license: string;
  redistributable: boolean;
}

const __dirname = dirname(fileURLToPath(import.meta.url));

function resolveDataDir(): string {
  if (process.env.WORDFOR_DATA_DIR) return process.env.WORDFOR_DATA_DIR;
  // dist/ -> package root -> repo root /data
  return join(__dirname, "..", "..", "data");
}

export class WordForData {
  readonly entries: WordEntry[];
  readonly bitIndex: Record<string, number>;
  readonly invBit: Record<number, string>;
  readonly sourceMeta: Record<string, { name: string; license: string; redistribute_text?: boolean; default_visible?: boolean }>;
  readonly formsToLemma: Map<string, string>;
  private readonly headwordIndex: Map<string, number[]>;

  constructor(dataDir = resolveDataDir()) {
    const manifest = JSON.parse(readFileSync(join(dataDir, "source_manifest.json"), "utf8"));
    this.bitIndex = manifest.bit_index;
    this.sourceMeta = manifest.sources || {};
    this.invBit = {};
    for (const [k, v] of Object.entries(this.bitIndex)) this.invBit[v as number] = k;

    const raw = JSON.parse(readFileSync(join(dataDir, "words.json"), "utf8"));
    this.entries = Array.isArray(raw) ? raw : raw.words;

    this.formsToLemma = new Map();
    try {
      const f2l = JSON.parse(readFileSync(join(dataDir, "forms_to_lemma.json"), "utf8"));
      for (const [k, v] of Object.entries(f2l)) this.formsToLemma.set(k, v as string);
    } catch {
      /* optional */
    }

    // headword -> entry indices, for fast exact lookups
    this.headwordIndex = new Map();
    for (let i = 0; i < this.entries.length; i++) {
      const hw = (this.entries[i].w?.[0] || "").toLowerCase();
      if (!hw) continue;
      const arr = this.headwordIndex.get(hw);
      if (arr) arr.push(i);
      else this.headwordIndex.set(hw, [i]);
    }
  }

  /** Decode a src bitmask into license-safe source labels (visible sources only). */
  decodeSources(mask: number): SourceLabel[] {
    const out: SourceLabel[] = [];
    for (let bit = 0; mask; bit++, mask >>= 1) {
      if (!(mask & 1)) continue;
      const id = this.invBit[bit];
      if (!id) continue;
      const meta = this.sourceMeta[id];
      if (!meta) continue;
      // Only emit sources that are redistributable / visible -- never restricted text sources.
      if (meta.redistribute_text === false) continue;
      out.push({
        id,
        name: meta.name || id,
        license: meta.license || "unknown",
        redistributable: true,
      });
    }
    return out;
  }

  canonicalLemma(word: string): string {
    const w = word.toLowerCase();
    return this.formsToLemma.get(w) || w;
  }

  entriesForHeadword(word: string): WordEntry[] {
    const idx = this.headwordIndex.get(word.toLowerCase()) || [];
    return idx.map((i) => this.entries[i]);
  }
}

let singleton: WordForData | null = null;
export function getData(): WordForData {
  if (!singleton) singleton = new WordForData();
  return singleton;
}
