import streamlit as st
import json
import time
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

# --- Config ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME = "ahmedfarazsyk/ms-marco-MiniLM-L6-v2-finetuned-scidocs"
OLLAMA_MODEL = "qwen2.5:1.5b"
STRATEGY = "large_fixed"
TOP_K = 5
CANDIDATES = 50

# --- Load artifacts (cached) ---
@st.cache_resource
def load_artifacts():
    # Load chunks
    chunks = json.loads(Path("data/chunks/strategy_B_large_fixed.json").read_text())
    # Load FAISS index
    import faiss
    index = faiss.read_index("data/indexes/minilm_large_fixed.faiss")
    # Load models
    bi_encoder = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_NAME, activation_fn=torch.nn.Sigmoid())
    return chunks, index, bi_encoder, reranker

# --- Retrieve + Rerank ---
def retrieve(query, chunks, index, bi_encoder, reranker):
    q_vec = bi_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_vec, CANDIDATES)
    candidates = [chunks[idx] for idx in indices[0] if idx != -1]
    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    for i, s in enumerate(rerank_scores):
        candidates[i]["score"] = float(s)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:TOP_K]

# --- Generate answer via Ollama ---
def generate_answer(query, contexts):
    context_text = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)])
    prompt = f"""Answer the question based on the following context from scientific papers.
If the context doesn't contain enough information, say so.
Be concise and technical.

Context:
{context_text}

Question: {query}
Answer:"""
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# --- Streamlit UI ---
st.title("Scientific RAG Assistant")
st.sidebar.toggle("Show sources", key="show_sources")

chunks, index, bi_encoder, reranker = load_artifacts()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Ask a scientific question"):
    st.chat_message("user").write(query)
    
    with st.spinner("Retrieving..."):
        t0 = time.perf_counter()
        results = retrieve(query, chunks, index, bi_encoder, reranker)
        retrieval_time = (time.perf_counter() - t0) * 1000
    
    with st.spinner("Generating answer..."):
        t0 = time.perf_counter()
        answer = generate_answer(query, results)
        llm_time = (time.perf_counter() - t0) * 1000
    
    st.chat_message("assistant").write(answer)
    st.caption(f"Retrieval: {retrieval_time:.0f}ms | LLM: {llm_time:.0f}ms | Total: {retrieval_time + llm_time:.0f}ms")
    
    if st.session_state.show_sources:
        with st.expander("Sources"):
            for i, r in enumerate(results):
                st.write(f"**[{i+1}]** `{r['paper_id']}` — score: {r['score']:.4f}")
                st.write(r["text"][:300] + "...")
                st.divider()
    
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer})