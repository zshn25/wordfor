/**
 * WordFor: Service Worker
 * Enables offline support and "Add to Home Screen" (PWA).
 *
 * Strategy:
 *   - App shell (HTML, CSS, JS, icons): stale-while-revalidate.
 *   - Data files & model files (.bin, .json, .txt, .onnx): cache-first (large, rarely change).
 *   - Cross-origin model files (HuggingFace CDN): cache-first.
 *   - Navigation: network-first with offline fallback to cached shell.
 */

const CACHE_NAME = "wordfor-v16";

const APP_SHELL = [
  "/",
  "/index.html",
  "/about.html",
  "/style.css",
  "/app.js",
  "/manifest.json",
  "/android-chrome-192x192.png",
  "/android-chrome-512x512.png",
];

// Pre-cache app shell on install

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(async cache => {
      await Promise.allSettled(
        APP_SHELL.map(url => cache.add(url))
      );
    })
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

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  if (event.request.method !== "GET") return;

  // Skip analytics
  if (url.hostname.includes("goatcounter")) return;

  // Navigation: network-first with offline fallback
  if (event.request.mode === "navigate") {
    event.respondWith(networkFirstNav(event.request));
    return;
  }

  // Cross-origin model files (HuggingFace, CDN): cache-first
  if (url.origin !== self.location.origin && isModelFile(url)) {
    event.respondWith(cacheFirst(event.request));
    return;
  }

  // Only handle same-origin from here
  if (url.origin !== self.location.origin) return;

  if (isDataFile(url.pathname)) {
    event.respondWith(cacheFirst(event.request));
    return;
  }

  event.respondWith(staleWhileRevalidate(event.request));
});

function isDataFile(pathname) {
  if (pathname.startsWith("/data/") &&
    (pathname.endsWith(".bin") || pathname.endsWith(".json") || pathname.endsWith(".txt") || pathname.endsWith(".wasm") || pathname.endsWith(".safetensors"))) {
    return true;
  }
  if (pathname.startsWith("/vendor/")) return true;
  if (pathname.startsWith("/models/") &&
    (pathname.endsWith(".onnx") || pathname.endsWith(".onnx_data") || pathname.endsWith(".json") || pathname.endsWith(".bin"))) {
    return true;
  }
  return false;
}

function isModelFile(url) {
  const h = url.hostname;
  return h.includes("huggingface.co") || h.includes("cdn.jsdelivr.net") || h.includes("unpkg.com");
}

async function networkFirstNav(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    if (cached) return cached;
    // Fallback to cached index for SPA-like navigation
    const index = await caches.match("/index.html");
    if (index) return index;
    return new Response("Offline", { status: 503, headers: { "Content-Type": "text/plain" } });
  }
}

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return new Response("Service Unavailable", {
      status: 503,
      headers: { "Content-Type": "text/plain" },
    });
  }
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);

  const fetchPromise = fetch(request)
    .then((response) => {
      if (response && response.ok) cache.put(request, response.clone());
      return response;
    })
    .catch(() => null);

  return cached || (await fetchPromise) || new Response("Service Unavailable", {
    status: 503,
    headers: { "Content-Type": "text/plain" },
  });
}
