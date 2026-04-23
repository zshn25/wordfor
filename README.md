# WordFor: A reverse dictionary that runs entirely in your browser

![wordfor](wordfor.gif)

**A free, private reverse dictionary using sentence embeddings, 1-bit quantization, and static model inference - zero server-side compute.**

[Blog post](https://zshn25.github.io/wordfor-reverse-dictionary) | [Try it](https://wordfor.xyz)

## Architecture

- **Full mode (desktop)**: Asymmetric retrieval: definitions encoded offline by [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) (1024d, MRL-truncated to 384d), queries encoded at runtime by [mdbr-leaf-mt](https://huggingface.co/MongoDB/mdbr-leaf-mt) (22M params) via [Transformers.js](https://huggingface.co/docs/transformers.js). Two-stage scoring: 1-bit binary ITQ Hamming first-pass (~13ms) + int3 reranking of top-500 candidates.
- **Full mode (mobile)**: Same model, pure 1-bit binary ITQ scoring only. ~30 MB total download.
- **Lite mode**: Knowledge-distilled static embeddings (256d, fine-tuned from mxbai-embed-large via Model2Vec). Sub-1ms queries via WASM or pure JS. Automatic fallback when ONNX model can't load (e.g. iOS).

## Evaluation (67-query test set)

| Mode | Config | MRR | Hit@1 | Hit@6 |
|------|--------|:---:|:-----:|:-----:|
| Full | binary + int3 rerank (desktop) | 0.644 | 37/67 | 52/67 |
| Full | pure binary ITQ (mobile) | 0.563 | 30/67 | 52/67 |
| Lite | distilled-mxbai fine-tuned, int4 | 0.566 | 33/67 | 42/67 |

## Dictionary

350,000+ definitions from 5 public-domain sources: [Open English WordNet 2025](https://en-word.net/) (CC BY 4.0), Webster's 1913, GCIDE Webster portion, [Century Dictionary](https://en.wikipedia.org/wiki/Century_Dictionary) (1889-1911), and 61 LLM-augmented entries (CC0). Enriched with [Moby Thesaurus](https://en.wikipedia.org/wiki/Moby_Project) synonyms. [Wiktionary](https://kaikki.org/) (CC BY-SA 3.0) and [ConceptNet 5.7](https://conceptnet.io/) (CC BY-SA 4.0) used at build time for quality signals and embedding enrichment only; not redistributed.

## Privacy

Static files served from GitHub Pages through Cloudflare CDN. [GoatCounter](https://www.goatcounter.com/) for cookie-free analytics. No personal data collected.

<!--
## Updating Guide
- Update sitemap.txt version date
- Update transformers.js (https://cdn.jsdelivr.net/npm/@huggingface/transformers@4/dist/transformers.min.js -> Save as wordfor/vendor/transformers.min.js)
- Clear CloudFlare cache manually
-->

---

&copy; 2025 Zeeshan Khan Suri. Licensed under CC-BY-NC-ND-4.0.
