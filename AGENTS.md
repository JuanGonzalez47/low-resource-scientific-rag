# low-resource-scientific-rag — Agent Instructions

## Project Overview
Experimental evaluation of low-resource retrieval architectures for scientific literature. Uses adaptive ranking, semantic chunking, lightweight embeddings (MiniLM, BGE-small), and cross-encoder reranking.

## Pipeline (notebook execution order)

1. **corpus_preproc.ipynb** — PDF → clean text (`data/bronze/` → `data/silver/`)
2. **chunk_strategies.ipynb** — Three chunking strategies (`data/silver/` → `data/chunks/`)
   - Strategy A: Small fixed (100 tokens, 10% overlap) → `strategy_A_small_fixed.json`
   - Strategy B: Large fixed (256 tokens, 10% overlap) → `strategy_B_large_fixed.json`
   - Strategy C: Semantic (paragraph-aware, 40–256 tokens) → `strategy_C_semantic.json`
3. **RAG.ipynb** — Embeddings + FAISS + retrieval evaluation (`data/chunks/` → `data/gold/`, `data/indexes/`)
4. **retrieve_rerank.ipynb** — Same as RAG but adds cross-encoder reranking

## Key Dependencies
```bash
pip install -r requirements.txt
pip install pymupdf4llm
```
- Models: `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM: `ollama pull qwen2.5:1.5b`

## Data Directories
| Dir | Purpose | Git |
|-----|---------|-----|
| `data/bronze/` | 10 raw PDFs (arXiv papers) | Excluded |
| `data/silver/` | Cleaned .txt files | Excluded |
| `data/chunks/` | 3 JSON chunk files (A, B, C) | Excluded |
| `data/gold/` | 6 .npy embedding matrices (2 models × 3 strategies) | Excluded |
| `data/indexes/` | 6 FAISS IndexFlatIP indexes | Excluded |
| `data/metadata/` | corpus_stats.json, chunking_stats.json, retrieval_eval.json | Excluded |
| `data/benchmark/` | benchmark.json (30 queries with difficulty labels) | Included |

## Running Notebooks
- Execute cells sequentially in each notebook
- Notebooks cache embeddings/indexes to disk — re-running skips if files exist
- First run of RAG.ipynb takes ~5 min (embedding generation); subsequent runs are fast
- **Important**: Delete `data/gold/*.npy` and `data/indexes/*.faiss` before re-running after chunk changes

## Evaluation Metrics
- Recall@3, Recall@5, Precision@5, MRR (overall + per difficulty: easy/medium/hard)
- Adaptive Relevance Scoring (ARS): 35% lexical + 25% technical + 40% semantic
- Results saved to `data/metadata/retrieval_eval.json` and `retrieval_eval_detail.json`

## Key Fixes Applied
1. **Sigmoid activation**: Cross-encoder scores normalized to [0, 1]
2. **PyMuPDF4LLM migration**: Eliminated extraction bugs
3. **Equation detection**: Semantic chunking preserves equations
4. **Cache invalidation**: Fixed FAISS IndexError from stale embeddings

## Common Commands
```bash
# From repo root, run a single notebook via jupyter (or convert to .py first)
jupyter nbconvert --to notebook --execute notebooks/corpus_preproc.ipynb
jupyter nbconvert --to notebook --execute notebooks/chunk_strategies.ipynb
jupyter nbconvert --to notebook --execute notebooks/RAG.ipynb
jupyter nbconvert --to notebook --execute notebooks/retrieve_rerank.ipynb

# Launch chat app
streamlit run app/streamlit_app.py
```

## Gotchas
- **requirements.txt** — install all deps with `pip install -r requirements.txt`
- **PyMuPDF4LLM** — install separately: `pip install pymupdf4llm`
- **Windows paths** in notebooks use `..\data\...` — works on Linux too via pathlib
- **HF_TOKEN** recommended for faster model downloads (rate limits without)
- **Tokenizer** uses MiniLM's BPE for accurate token counting (not word split)
- **Equation detection** heuristic: preserves complete equations in semantic chunks
- **Benchmark** requires `query`, `relevant_paper_id`, `answer_contains`, `difficulty` fields
- **Cache invalidation** — delete `data/gold/*.npy` and `data/indexes/*.faiss` before re-running after chunk changes