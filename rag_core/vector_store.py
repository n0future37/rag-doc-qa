import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_core.config import CHUNKS_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL, ensure_dirs

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def build_index(chunks):
    embedder = get_embedder()
    texts = [chunk["content"] for chunk in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def save_vector_store(index, chunks):
    ensure_dirs()
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def load_vector_store():
    if not FAISS_INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def retrieve(question, index, chunks, top_k=4):
    embedder = get_embedder()
    query_embedding = embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1: continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)
    return results