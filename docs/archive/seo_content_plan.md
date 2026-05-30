# SEO content plan

WordFor is a single-page app on GitHub Pages (`wordfor.xyz`). The homepage already has
strong technical SEO (canonical, OpenGraph/Twitter, `WebSite` + `WebApplication` +
`Person` JSON-LD, sitelinks SearchAction). This plan adds (a) crawlable content pages
with real URLs and (b) a batch of genuinely useful articles targeting natural search
queries. First batch (>=5 posts) is implemented this session.

## Technical SEO — status

| Item | Status |
|------|--------|
| Canonical URLs | DONE on `/`, `/about.html`; added to every new page |
| OpenGraph / Twitter cards | DONE on `/`; added per blog post |
| `WebSite` + `WebApplication` JSON-LD | DONE on `/` |
| `Article` + `BreadcrumbList` JSON-LD | ADDED on each new blog post |
| `FAQPage` JSON-LD | ADDED only where a visible FAQ exists (feeling-you-cant-name post) |
| `sitemap.xml` | ADDED (replaces ambiguous `sitemap.txt`); lists all pages + posts |
| `robots.txt` | UPDATED to point at `sitemap.xml` |
| Content visible without model load | DONE — blog/example pages are static HTML, load **no** `app.js`/models |
| Crawl-trap guard on `?q=` | already present on `/` (noindex,follow when `q` set) |

### Real URLs (no SPA-only state)
The app is one page, but content lives at crawlable URLs:
- `/` — the app (reverse dictionary)
- `/about.html` — how it works
- `/blog/` — article index
- `/blog/<slug>.html` — articles (static, no model load)
- `/examples/<slug>.html` — example-search landing pages (future batch; pattern defined)

> The conceptual routes `/reverse-dictionary`, `/tip-of-my-tongue`, `/word-finder` are
> served as **content articles** (below) that each link into the app with a prefilled
> `?q=` query, rather than as separate SPA states. This keeps every important intent on a
> real, indexable URL without a router.

## Content batch 1 (implemented)

| Slug | Target query | Schema |
|------|--------------|--------|
| `reverse-dictionary-how-to-find-a-word-from-a-meaning.html` | "reverse dictionary how to find a word from a meaning" | Article + Breadcrumb |
| `tip-of-my-tongue-word-finder.html` | "tip of my tongue word finder" | Article + Breadcrumb |
| `word-for-a-feeling-you-cant-name.html` | "what is the word for a feeling you can't name" | Article + Breadcrumb + FAQPage |
| `how-wordfor-ranks-candidate-words.html` | "how WordFor ranks candidate words" | Article + Breadcrumb |
| `public-domain-dictionaries-without-copying-restricted-data.html` | "public domain dictionaries without copying restricted data" | Article + Breadcrumb |

Each post: exact query phrase in `<title>`/H1; 3-5 example searches linking to
`/?q=<query>`; example candidate words; a "How ranking works" section (lexical match ->
semantic similarity -> source confidence -> lemma family grouping -> reranking); related
terms (reverse dictionary, word finder, tip of my tongue, vocabulary, thesaurus, word
meaning, find a word, semantic search); internal links to the app and sibling posts.

## Content batch 2 (planned — not yet written)

- "Word for nostalgia for something you never experienced" (anemoia / saudade)
- "Word for being happy and sad at the same time" (bittersweet / melancholy)
- "Word for fear of missing out" (FOMO)
- "Word for wanting something impossible" (quixotic / wishful)
- "WordFor vs thesaurus vs reverse dictionary"

These follow the same template; `/examples/<slug>.html` landing pages can wrap each with
a prefilled query and the top example results.

## Guidelines applied
- Exact query phrase appears once, naturally, in title + H1 (no stuffing).
- Real, useful English vocabulary in examples (not fabricated app output — examples are
  labeled as "words WordFor surfaces", and the linked query lets the reader verify live).
- Internal linking between posts and into the app.
- No keyword stuffing; readable prose first.

## Measurement (not run)
Submit `sitemap.xml` in Google Search Console; track impressions/clicks per query. No
ranking claims are made here — only the on-page work is done.
