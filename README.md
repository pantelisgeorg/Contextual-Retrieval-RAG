# Contextual Retrieval RAG

A complete implementation of Anthropic's **Contextual Retrieval** technique with two interfaces — a CLI for scripting and a small FastAPI web UI for interactive use. Document parsing uses **Docling** for structure-preserving extraction (markdown, headings, tables, embedded images).

## Architecture

| Component | Original Guide | This Implementation |
|---|---|---|
| Document loader | — | **Docling** (PDF, DOCX, PPTX, XLSX, HTML) with markdown + image extraction; PyMuPDF fallback |
| LLM (contextualization) | Anthropic Claude | **OpenAI GPT-4o-mini** |
| Embeddings | Voyage AI `voyage-2` | **Ollama `nomic-embed-text-v2-moe`** |
| Vector DB | In-memory / pickle | **ChromaDB** (persistent) |
| BM25 Hybrid | Elasticsearch (Docker) | **`rank-bm25`** (pure Python) |
| Reranker | Cohere API | **Local `BAAI/bge-reranker-v2-m3`** |
| Query Expansion | — | **Ollama `hf.co/tobil/qmd-query-expansion-1.7b-GGUF:Q4_K_M`** |
| Web UI | — | **FastAPI** + a single static HTML page |

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager. `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Ollama** running locally — handles embeddings and query expansion. `curl -fsSL https://ollama.com/install.sh | sh`
- **An NVIDIA GPU (optional)** — accelerates the reranker, Docling, and EasyOCR. The torch stack is pinned to CUDA 12.1, supporting compute capability 5.0+ (see [GPU Support](#gpu-support)). Without one, everything runs on CPU.
- **An OpenAI API key** — used only for GPT-4o-mini chunk contextualization at ingest time.
- **Internet access on first ingest/query** — Docling models (~500 MB), the reranker (~600 MB), and embedding models are downloaded once and cached locally.

### Models at a glance

| Model | Role | Source | Disk |
|---|---|---|---|
| Docling layout + OCR | PDF/DOCX/PPTX → structured markdown | Auto-downloaded from Hugging Face on first ingest | ~500 MB |
| `gpt-4o-mini` | Chunk contextualization (ingest only) | OpenAI API — needs `OPENAI_API_KEY` | — |
| `nomic-embed-text-v2-moe` | Embeddings | `ollama pull` | ~500 MB |
| `hf.co/tobil/qmd-query-expansion-1.7b-GGUF:Q4_K_M` | Query expansion | `ollama pull` from Hugging Face | ~1 GB |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker | Auto-downloaded from Hugging Face on first query | ~600 MB |

## Quick Start

### 1. Clone and sync

```bash
git clone <repo-url>
cd "Contextual Retrieval RAG CLI"
uv sync
```

`uv sync` reads `pyproject.toml`, creates `.venv/`, and installs every dependency from the lockfile.

### 2. Pull the Ollama models

```bash
ollama pull nomic-embed-text-v2-moe
ollama pull hf.co/tobil/qmd-query-expansion-1.7b-GGUF:Q4_K_M
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — paste your real OPENAI_API_KEY
```

> ⚠️ `.env` contains a real secret. It is gitignored — never commit it. Use `.env.example` (placeholder values only) as the shared template.

### 4. Drop documents into `sources/`

```bash
cp my_documents/*.pdf sources/
```

Supported formats:
- With `USE_DOCLING=false` (default, fast): `.pdf` (text only) plus `.txt .md .py .js .json .yaml .html …`
- With `USE_DOCLING=true` (recommended): all of the above plus `.docx .pptx .xlsx .html` with structure and images preserved

### 5. Ingest

**CLI:**
```bash
uv run rag-system ingest --clear
```

**Web UI:** open the ingest panel and click "Ingest from sources/" (see below).

### 6. Query

**CLI:**
```bash
uv run rag-system query "How does authentication work?"
uv run rag-system chat
```

**Web UI:**
```bash
uv run rag-system serve
# Open http://127.0.0.1:8000
```

## Web UI

`uv run rag-system serve` starts a FastAPI server on `127.0.0.1:8000`. The single-page UI provides:

- **Search box** with adjustable top-k.
- **Result cards** — markdown rendered (headings, tables, lists, **images** when Docling is on); each card has a *Show contextualization* expander revealing the GPT-4o-mini-generated context for that chunk.
- **Model chips** at the top — at-a-glance view of which loader/contextualizer/embedder/reranker/expander is active.
- **Per-query pipeline indicator** — chips below the result count showing what *actually ran* for that query, e.g. `expand 4v/3b · hybrid bm25+vec · rerank 20 → 10`.
- **Ingest panel** (collapsible) — pick a single file from `sources/` or "All files", optional Clear-database-first checkbox, **live progress log** streamed via NDJSON. The retriever is rebuilt automatically when ingest finishes.

CLI flags:
```bash
uv run rag-system serve --host 127.0.0.1 --port 8000   # defaults
uv run rag-system serve --host 0.0.0.0 --port 8000     # LAN-accessible (no auth — careful)
```

API endpoints (for scripting):

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/health` | Returns `{"chunks": int}` |
| `GET`  | `/api/models` | Active models for each pipeline stage |
| `GET`  | `/api/sources` | List files in `sources/` |
| `POST` | `/api/query`  | `{query, top_k}` → `{results, pipeline}` |
| `POST` | `/api/ingest` | `{clear, file?}` → NDJSON stream of progress events |
| `GET`  | `/images/{slug}/{file}.png` | Diagrams extracted from PDFs (when Docling enabled) |

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** GPT-4o-mini API key. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Contextualization model. |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL. |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text-v2-moe` | Embedding model. Multilingual for Greek language as well |
| `OLLAMA_QUERY_EXPANSION_MODEL` | `hf.co/tobil/qmd-query-expansion-1.7b-GGUF:Q4_K_M` | Local query expansion model. |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker. |
| `CHROMA_DB_PATH` | `./data/chromadb` | Persistent vector DB path. |
| `SOURCES_DIR` | `./sources` | Input documents folder. |
| `USE_DOCLING` | `false` | Enable Docling for structure-preserving extraction (markdown, headings, tables, images). Recommended `true`. |
| `OCR_ENABLED` | `false` | Run OCR on image regions / scanned pages (Docling only). Born-digital PDFs don't need it. Enable for scanned docs. Requires Tesseract installed system-wide. |
| `OCR_LANGS` | `ell+eng` | `+`-separated [Tesseract language codes](https://github.com/tesseract-ocr/tessdata). Each language requires its `tesseract-ocr-<code>` system package (e.g. `sudo apt install tesseract-ocr-ell tesseract-ocr-eng`). |
| `DOCLING_MODEL` | `default` | Docling document model. `default` = standard layout pipeline + **TableFormer v2** (reads the embedded text layer, accurate text, fixed table alignment). `smoldocling` = 256M vision-language model (`docling-project/SmolDocling-256M`) running in-process via transformers on the GPU — best for **scanned** documents with no text layer. See [Document Models](#document-models). |
| `DOCLING_VLM_MAX_SIZE` | `1280` | SmolDocling only: longest image edge (px) fed to the VLM. Lower = less VRAM, lower = lower OCR quality. |
| `CHUNK_SIZE` | `800` | Characters per chunk. Larger → fewer GPT calls during ingest, but coarser retrieval granularity. 1500–3000 works well with Docling. Tables are never split across chunks. |
| `CHUNK_OVERLAP` | `100` | Overlap when a section exceeds chunk size and must be split. |
| `QUERY_EXPANSION` | `false` | Enable query expansion. |
| `RERANKING` | `true` | Enable reranking. |
| `BM25_HYBRID` | `true` | Enable BM25 + semantic fusion. |
| `SEMANTIC_WEIGHT` | `0.8` | RRF weight for semantic search. |
| `BM25_WEIGHT` | `0.2` | RRF weight for BM25. |
| `TOP_K` | `10` | Default number of results. |
| `ANSWER_SYNTHESIS` | `true` | Generate a written answer from the retrieved passages (with inline `[n]` citations). Disable to return raw chunks only. |
| `ANSWER_MODEL` | falls back to `OPENAI_MODEL` | LLM used for answer synthesis (e.g. `gpt-4o-mini`, `gpt-4o`). |
| `ANSWER_MAX_TOKENS` | `800` | Max tokens in the synthesized answer. |
| `CUDA_VISIBLE_DEVICES` | unset | Restrict visible GPUs (e.g. `"0"`). Set to an empty string (`CUDA_VISIBLE_DEVICES=`) to force CPU — only needed for GPUs below compute capability 5.0. See [GPU Support](#gpu-support). |

## Project Structure

```
.
├── rag_system/              # Core package
│   ├── config.py            # Reads .env into a Config dataclass
│   ├── document_loader.py   # Docling / PyMuPDF / plain-text loaders
│   ├── chunker.py           # Heading-aware chunker with section packing
│   ├── contextualizer.py    # GPT-4o-mini chunk context
│   ├── embedder.py          # Ollama embeddings
│   ├── database.py          # ChromaDB wrapper
│   ├── bm25_index.py        # Lightweight BM25
│   ├── reranker.py          # BGE cross-encoder reranker
│   ├── query_expander.py    # qmd-query-expansion-1.7b-GGUF (Qwen3-1.7B)
│   ├── answerer.py          # Strict-grounded answer synthesis with citations
│   ├── retriever.py         # RRF fusion + rerank pipeline
│   ├── ingest.py            # End-to-end ingest function with progress callback
│   ├── web.py               # FastAPI app (used by `serve`)
│   ├── static/              # Single-page UI (index.html + marked.min.js)
│   └── cli.py               # CLI entry point (ingest / query / chat / serve)
├── sources/                 # Drop raw documents here
├── data/                    # ChromaDB + BM25 index + Docling artifacts (gitignored)
│   ├── chromadb/
│   ├── images/{slug}/       # Per-doc image artifacts (referenced from markdown)
│   ├── markdown/{slug}.md   # Docling-extracted markdown with /images/... URLs (web-UI / RAG)
│   └── export/{slug}.md     # Same markdown but with images base64-embedded — single-file artifact for wikis
├── .venv/                   # uv-managed virtualenv (gitignored)
├── pyproject.toml           # Source of truth for deps; defines `rag-system` console script
├── uv.lock                  # Reproducible install
├── requirements.txt         # Mirror of pyproject for plain-pip users
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## How It Works

1. **Document loading** — when `USE_DOCLING=true`, PDFs/DOCX/PPTX/HTML go through Docling, producing markdown with real headings, tables, and `![Image](/images/...)` references. Two persistent copies are written per document: `data/markdown/<slug>.md` (image-referenced, fed into the RAG and the web UI) and `data/export/<slug>.md` (base64-embedded, a single self-contained file you can hand to a wiki). The default pipeline uses the **v2 TableFormer** and reads text from the embedded text layer (accurate, no OCR errors); a post-pass re-derives table column order from cell coordinates so each value lands under the correct header. For scanned documents with no text layer, set `DOCLING_MODEL=smoldocling` to run the 256M SmolDocling VLM (OCRs the page image end-to-end). With `OCR_ENABLED=true` on the default pipeline, Tesseract OCRs any region without an embedded text layer. Otherwise PyMuPDF extracts plain text from PDFs.
2. **Heading-aware chunking** — markdown is split on heading boundaries; small adjacent sections are greedily packed up to `CHUNK_SIZE` so each chunk is semantically coherent (a full section, not a mid-sentence cut). **Markdown tables are atomic** — a table is never split across chunks (an oversized table becomes one chunk rather than being broken mid-row). Heading paths are preserved as prefixes so the contextualizer and embedder see *where* each chunk lives.
3. **Contextualization** — for each chunk, GPT-4o-mini gets the full document plus that chunk and returns 1–2 sentences situating the chunk in the larger doc. That blurb is **prepended to the chunk before embedding** ([Anthropic's Contextual Retrieval technique](https://www.anthropic.com/news/contextual-retrieval)).
4. **Embedding** — the contextualized chunk is embedded via Ollama's `nomic-embed-text-v2-moe`.
5. **Storage** — embeddings + metadata go into ChromaDB. A parallel BM25 index is built over the same text for keyword search.
6. **Retrieval** —
   - Optional query expansion via local GGUF model (one query → multiple sub-queries for vec + BM25).
   - Semantic search (ChromaDB) ⨯ BM25 search (`rank-bm25`).
   - Reciprocal Rank Fusion combines both lists.
   - Optional reranking with `bge-reranker-v2-m3` re-scores the top candidates.
7. **Answer synthesis** — when `ANSWER_SYNTHESIS=true`, the retrieved passages and the question are sent to `ANSWER_MODEL` (default `gpt-4o-mini`) with a strict-grounding prompt: answer only from the passages, match the question's language, cite passages inline as `[1]`, `[3]`, etc. The answer is rendered above the source chunks.
8. **Display** — the web UI shows the synthesized answer at the top and renders source chunks as markdown so headings and diagrams are visible inline. To make sure figures show even when the cited chunk is text-only (the chunker can split a figure away from its descriptive paragraph), the server retrieves a larger candidate pool and surfaces up to 3 extra image-bearing chunks from the same document, ranked by the retriever's own score. Pipeline chips show what actually ran for the query.

## Document Models

`DOCLING_MODEL` selects how Docling converts documents to markdown:

| Model | How it works | Text source | Best for | GPU |
|---|---|---|---|---|
| `default` (default) | Standard layout pipeline + **TableFormer v2** | Embedded text layer (exact) | Born-digital PDFs/DOCX/PPTX | Optional |
| `smoldocling` | 256M vision-language model (`docling-project/SmolDocling-256M`) processes each page image → DocTags | VLM reads the image (OCR) | Scanned docs, image-only PDFs | Recommended |

**Tables.** With `default`, the v2 TableFormer detects the table grid and pulls each cell's text from the embedded layer (no OCR errors). Docling's TableFormer sometimes assigns correct cell bounding boxes but *wrong column indices* (a cyclic shift that lands a value under the wrong header); a built-in post-pass re-derives each column from the cells' x-coordinates so every value sits under the correct header. With `smoldocling`, table structure comes from the VLM's DocTags — clean, but the cell text is OCR'd, so non-Latin scripts may contain recognition errors.

**SmolDocling on the GPU.** SmolDocling ships as a bfloat16 model, but Pascal GPUs (sm_61, e.g. GTX 1070 Ti) have no native bfloat16, so it's forced to **float16** automatically (≈2 GB VRAM at `DOCLING_VLM_MAX_SIZE=1280`, ~0.4 s/page). It runs in-process via the transformers backend — **no vLLM required** (vLLM needs compute capability ≥ 7.0 anyway). Lower `DOCLING_VLM_MAX_SIZE` to reduce VRAM at the cost of OCR quality.

> For most documents — including this project's Greek PDFs — keep `DOCLING_MODEL=default`. Switch to `smoldocling` only when the source has no embedded text layer (scans, photographs of pages).

## GPU Support

The torch stack is pinned to **`torch==2.5.1+cu121`** (CUDA 12.1) in `pyproject.toml`, sourced from the [PyTorch cu121 wheel index](https://download.pytorch.org/whl/cu121). These prebuilt kernels cover **compute capabilities 5.0–9.0** (`sm_50`–`sm_90`), including older cards that newer torch builds dropped:

| GPU family | Example | Compute capability | Supported |
|---|---|---|---|

| Pascal | GTX 1070 Ti | 6.1 | Yes |
| Turing | RTX 2070 | 7.5 | Yes |
| Ampere | RTX 3080 | 8.6 | Yes |
| Ada | RTX 4090 | 8.9 | Yes (PTX JIT) |
| Hopper | H100 | 9.0 | Yes |

The reranker (`bge-reranker-v2-m3`), Docling layout/OCR models, EasyOCR, and the SmolDocling VLM run on the GPU automatically when one is visible, and fall back to CPU otherwise. CPU mode is fully functional — ingest just takes a minute or two longer (SmolDocling on CPU is noticeably slow; prefer the `default` model or a GPU).

> **Why the pin?** PyTorch ≥2.6 removed `sm_50`/`sm_60` from its CUDA wheels, so on a Pascal or Maxwell card you'd hit `cudaErrorNoKernelImageForDevice`. The 2.5.1+cu121 build is the last release that still ships those kernels. `uv sync` picks up the pin automatically; pip users need `--extra-index-url https://download.pytorch.org/whl/cu121` (see `requirements.txt`).

To force CPU (e.g. for a Kepler-class GPU below compute 5.0), set `CUDA_VISIBLE_DEVICES=` in `.env`.

## Notes

- **No Docker required**: BM25, reranker, and Docling all run locally in pure Python / PyTorch.
- **Local-first**: only contextualization (ingest) and HF model downloads (first run) need network.
- **Table extraction**: the default pipeline uses the **v2 TableFormer**, which keeps cell text intact (the older v1 model split cells mid-phrase). Because v1 and v2 each garble *different* tables (v2 sometimes overlaps columns; v1 sometimes splits cells), any table v2 garbles is automatically re-converted with **v1** and the cleaner version is kept (judged by cell-box overlap and duplicate text). A coordinate-based post-pass then re-derives each cell's column from its x-position, so every value lands under the correct header. SmolDocling (`DOCLING_MODEL=smoldocling`) produces clean table structure too, but reads text from the page image, so non-Latin scripts (e.g. Greek) may show OCR errors — prefer `default` for born-digital PDFs and `smoldocling` for scans.
- **uv vs pip**: `uv sync` is the supported path. A `requirements.txt` is kept alongside `pyproject.toml` for compatibility with plain-pip workflows, but if you `uv add <pkg>` later you'll need to mirror it manually (or run `uv export -o requirements.txt`).

## License

MIT — see [LICENSE](LICENSE).

The contextualization step is based on Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) technique.
