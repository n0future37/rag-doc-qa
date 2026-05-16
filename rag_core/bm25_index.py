from rank_bm25 import BM25Okapi
import json
from rag_core.config import BM25_INDEX_PATH, ensure_dirs

bm25_model = None
bm25_corpus = []

def build_bm25_index(chunks):
    global bm25_model, bm25_corpus
    ensure_dirs()
    bm25_corpus = [chunk["content"].split() for chunk in chunks]
    bm25_model = BM25Okapi(bm25_corpus)
    with open(BM25_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump([chunk["content"] for chunk in chunks], f, ensure_ascii=False, indent=2)
    return bm25_model

def load_bm25_index():
    global bm25_model, bm25_corpus
    if not BM25_INDEX_PATH.exists():
        return None
    with open(BM25_INDEX_PATH, "r", encoding="utf-8") as f:
        texts = json.load(f)
    bm25_corpus = [text.split() for text in texts]
    bm25_model = BM25Okapi(bm25_corpus)
    return bm25_model

def bm25_retrieve(query, chunks, top_k=4):
    global bm25_model, bm25_corpus
    if bm25_model is None:
        load_bm25_index()
    if bm25_model is None:
        return []
    tokenized_query = query.split()
    scores = bm25_model.get_scores(tokenized_query)
    scored_chunks = [{"chunk": chunk, "score": float(score)} for chunk, score in zip(chunks, scores)]
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]