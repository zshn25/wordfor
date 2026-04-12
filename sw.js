/**
 * WordFor — Service Worker
 * Enables offline support and "Add to Home Screen" (PWA).
 *
 * Strategy:
 *   - App shell (HTML, CSS, JS): stale-while-revalidate.
 *   - Data files & model files (.bin, .json, .txt, .onnx): cache-first (large, rarely change).
 *   - Cross-origin (CDN libs): NOT intercepted.
 */

const CACHE_NAME = "wordfor-v12";

const APP_SHELL = [
  "/",
  "/index.html",
  "/about.html",
  "/style.css",
  "/app.js",
  "/manifest.json",
];

// Pre-cache app shell on install
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

// Clean old caches on activate
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k.startsWith("wordfor-") && k !== CACHE_NAME)
          .map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// Fetch strategy — same-origin only
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Only handle same-origin http/https. Let cross-origin (CDN, HF models) pass through.
  if (url.origin !== self.location.origin) return;
  if (url.protocol !== "http:" && url.protocol !== "https:") return;

  // Data files: cache-first (large, content-versioned)
  if (isDataFile(url.pathname)) {
    event.respondWith(cacheFirst(event.request));
    return;
  }

  // App shell: stale-while-revalidate
  event.respondWith(staleWhileRevalidate(event.request));
});

function isDataFile(pathname) {
  if (pathname.startsWith("/data/") &&
    (pathname.endsWith(".bin") || pathname.endsWith(".json") || pathname.endsWith(".txt"))) {
    return true;
  }
  if (pathname.startsWith("/models/") &&
    (pathname.endsWith(".onnx") || pathname.endsWith(".onnx_data") || pathname.endsWith(".json") || pathname.endsWith(".bin"))) {
    return true;
  }
  return false;
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) {
    cache.put(request, response.clone());
  }
  return response;
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  const fetchPromise = fetch(request).then((response) => {
    if (response.ok) cache.put(request, response.clone());
    return response;
  }).catch(() => cached);
  return cached || fetchPromise;
}
