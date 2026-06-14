# SciLitRAG

Low-resource RAG system for scientific literature. Bi-encoder retrieval + cross-encoder reranking, running entirely on CPU.

https://github.com/user-attachments/assets/ba369699-7753-49e3-8dc6-c62aa1e84bc9

## Results

| Config | MRR | Recall@5 | Precision@5 | Hard MRR |
|--------|-----|----------|-------------|----------|
| BGE-small + Fixed + Rerank | **0.929** | **1.000** | **0.853** | 0.817 |
| BGE-small + Fixed | 0.831 | 0.933 | 0.667 | 0.656 |
| MiniLM + Fixed + Rerank | 0.922 | 0.967 | 0.860 | 0.792 |

Reranking improves MRR by +9.3% and Precision@5 by +40.5% with ~310ms overhead.

## Project Structure

```
├── notebooks/
│   ├── corpus_preproc.ipynb        # PDF → clean text (PyMuPDF4LLM)
│   ├── chunk_strategies.ipynb      # 3 chunking strategies
│   ├── RAG.ipynb                   # Embeddings + FAISS + evaluation
│   └── retrieve_rerank.ipynb       # Reranking pipeline
├── app/
│   └── streamlit_app.py            # Chat interface with Ollama
├── data/
│   ├── bronze/                     # 10 raw PDFs (arXiv papers)
│   ├── silver/                     # Cleaned .txt files
│   ├── chunks/                     # 3 JSON chunk files (A, B, C)
│   ├── gold/                       # Embedding matrices (.npy)
│   ├── indexes/                    # FAISS indexes
│   ├── benchmark/                  # 30 evaluation queries
│   └── metadata/                   # Stats and eval results
├── requirements.txt
└── LICENSE
```

## Install

```bash
pip install -r requirements.txt
```

Also install PyMuPDF4LLM:
```bash
pip install pymupdf4llm
```

And Ollama (for the chat app):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b
```

## Run

### 1. Extract and clean PDFs

```bash
jupyter notebook notebooks/corpus_preproc.ipynb
```

Reads `data/bronze/*.pdf` → writes `data/silver/*.txt`.

### 2. Generate chunks

```bash
jupyter notebook notebooks/chunk_strategies.ipynb
```

Creates 3 chunk files in `data/chunks/`:
- Strategy A: small fixed (100 tokens, 10% overlap)
- Strategy B: large fixed (256 tokens, 10% overlap)
- Strategy C: semantic (paragraph-aware, 40–256 tokens)

### 3. Build embeddings and indexes

```bash
jupyter notebook notebooks/RAG.ipynb
```

Generates embeddings (`data/gold/`) and FAISS indexes (`data/indexes/`). ~5 min on first run.

### 4. Run retrieval + reranking evaluation

```bash
jupyter notebook notebooks/retrieve_rerank.ipynb
```

### 5. Launch chat app

```bash
streamlit run app/streamlit_app.py
```

## Models

| Component | Model |
|-----------|-------|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding | `BAAI/bge-small-en-v1.5` |
| Reranker | `ahmedfarazsyk/ms-marco-MiniLM-L6-v2-finetuned-scidocs` |
| LLM | `qwen2.5:1.5b` (Ollama) |

## Benchmark

30 queries across 10 papers, categorized by difficulty:
- **Easy** (10): Direct retrieval, exact keyword match
- **Medium** (10): Requires synthesis across sections
- **Hard** (10): Multi-hop reasoning, implicit answers

## License

MIT
