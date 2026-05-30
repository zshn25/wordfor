/**
 * shard-loader.js -- progressive, cache-first loader for sharded runtime assets.
 *
 * GitHub Pages cannot serve files > 100 MiB and won't serve Git LFS reliably, so
 * large model/data assets are split into <=15 MiB shards by build/split_assets.py
 * and described in assets/model-manifest.json. This loader:
 *   - fetches shards in manifest order (critical first, background lazily)
 *   - caches each shard in the Cache API keyed by its sha256 (immutable)
 *   - reassembles shards into one ArrayBuffer
 *   - falls back to the monolithic data/<file> when no manifest/shard exists
 *
 * No build step / bundler required; attaches window.ShardLoader. Safe to load with a
 * plain <script> tag before app.js. All methods are no-throw friendly: callers should
 * still guard, but a missing manifest simply triggers monolithic fallback.
 */
(function () {
  "use strict";

  const CACHE_NAME = "wordfor-shards-v1";
  const MANIFEST_URL = "assets/model-manifest.json";
  const DATA_ROOT = "data";

  let manifestPromise = null;

  async function getManifest() {
    if (!manifestPromise) {
      manifestPromise = fetch(MANIFEST_URL)
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null);
    }
    return manifestPromise;
  }

  async function openCache() {
    try {
      return await caches.open(CACHE_NAME);
    } catch {
      return null; // Cache API unavailable (e.g. file://) -> network only
    }
  }

  /** Fetch one shard, cache-first by its immutable sha256-bearing URL. */
  async function fetchShard(shard, cache, onBytes) {
    const url = `${DATA_ROOT}/${shard.url}`;
    if (cache) {
      const hit = await cache.match(url);
      if (hit) {
        const buf = await hit.arrayBuffer();
        if (onBytes) onBytes(buf.byteLength, true);
        return new Uint8Array(buf);
      }
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`shard fetch failed: ${shard.url}`);
    const buf = await res.arrayBuffer();
    if (cache) {
      try {
        await cache.put(url, new Response(buf.slice(0), { headers: { "Content-Type": "application/octet-stream" } }));
      } catch { /* quota / private mode: ignore */ }
    }
    if (onBytes) onBytes(buf.byteLength, false);
    return new Uint8Array(buf);
  }

  function concat(parts, total) {
    const out = new Uint8Array(total);
    let off = 0;
    for (const p of parts) { out.set(p, off); off += p.length; }
    return out;
  }

  /**
   * Load a named asset as an ArrayBuffer.
   * @param {string} name        manifest asset name (e.g. "embeddings_int3")
   * @param {object} [opts]
   * @param {string} [opts.fallbackFile]  monolithic filename if not sharded
   * @param {(pct:number, fromCache:boolean)=>void} [opts.onProgress]
   * @returns {Promise<ArrayBuffer>}
   */
  async function loadAsset(name, opts = {}) {
    const manifest = await getManifest();
    const asset = manifest && manifest.assets && manifest.assets[name];

    if (!asset) {
      // Fallback: monolithic file (legacy / non-sharded deployment)
      const file = opts.fallbackFile || `${name}.bin`;
      const res = await fetch(`${DATA_ROOT}/${file}`);
      if (!res.ok) throw new Error(`asset not found: ${name}`);
      return res.arrayBuffer();
    }

    const cache = await openCache();
    const parts = [];
    let loaded = 0;
    let anyFromNetwork = false;
    for (const shard of asset.shards) {
      const part = await fetchShard(shard, cache, (n, fromCache) => {
        loaded += n;
        if (!fromCache) anyFromNetwork = true;
        if (opts.onProgress) opts.onProgress((loaded / asset.size) * 100, !anyFromNetwork);
      });
      parts.push(part);
    }
    return concat(parts, asset.size).buffer;
  }

  /** Load a named JSON asset (e.g. "words", "forms_to_lemma"). */
  async function loadJSON(name, opts = {}) {
    const buf = await loadAsset(name, opts);
    return JSON.parse(new TextDecoder().decode(new Uint8Array(buf)));
  }

  /** True if the manifest exists and describes this asset (i.e. sharded mode). */
  async function isSharded(name) {
    const m = await getManifest();
    return !!(m && m.assets && m.assets[name]);
  }

  /** Drop caches for manifest versions other than the current one. */
  async function pruneOldCaches() {
    try {
      const names = await caches.keys();
      await Promise.all(
        names.filter((n) => n.startsWith("wordfor-shards-") && n !== CACHE_NAME).map((n) => caches.delete(n))
      );
    } catch { /* ignore */ }
  }

  window.ShardLoader = { getManifest, loadAsset, loadJSON, isSharded, pruneOldCaches, CACHE_NAME };
})();
