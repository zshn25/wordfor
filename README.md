# WordFor: A reverse dictionary that runs entirely in your browser

![wordfor](wordfor.gif)

**A free, private reverse dictionary using sentence embeddings, 1-bit quantization, and static model inference — with zero server-side compute.**

Read the full blog post: https://zshn25.github.io/wordfor-reverse-dictionary

## Quick Summary

WordFor is a reverse dictionary where you describe a concept and instantly get the word. It runs entirely in the browser: no server, no database, no cookies.

### Architecture

- **Full mode (desktop)**: Asymmetric retrieval — definitions encoded offline by [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) (1024d, MRL-truncated to 384d), queries encoded at runtime by [mdbr-leaf-mt](https://huggingface.co/MongoDB/mdbr-leaf-mt) (22M params) via [Transformers.js](https://huggingface.co/docs/transformers.js). Two-stage scoring: 1-bit binary Hamming first-pass (~13ms) + int4 reranking of top-500 candidates (with int8 fallback support).
- **Full mode (mobile)**: Same model, pure 1-bit binary scoring (no int8 rerank). ~30 MB total download vs ~95 MB on desktop.
- **Lite mode**: [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) static embeddings (bag-of-words). Sub-1ms queries, used as fallback.

### Evaluation (67-query test set)

| Mode | MRR | Hit@1 | Hit@6 |
|------|:---:|:-----:|:-----:|
| Full binary + int8 rerank | 0.6296 | 36/67 | 54/67 |
| Full pure binary (1-bit) | 0.5782 | 32/67 | 51/67 |
| Lite (potion int8) | 0.1248 | 4/67 | 14/67 |

### Dictionary

175,000+ definitions from [Open English WordNet](https://en-word.net/) (CC BY 4.0), Webster's 1913 (public domain), and Moby Thesaurus (public domain). 168K supplementary entries from [Wiktionary](https://en.wiktionary.org/) via [kaikki.org](https://kaikki.org/) (CC BY-SA 3.0).

### Privacy

Static files served from GitHub Pages through Cloudflare CDN. [GoatCounter](https://www.goatcounter.com/) for cookie-free analytics. No personal data collected.

---

© 2025 Zeeshan Khan Suri. Licensed under CC-BY-NC-ND-4.0.
