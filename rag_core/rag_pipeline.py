from rag_core.document_loader import load_uploaded_file
from rag_core.text_splitter import split_documents
from rag_core.vector_store import build_index, save_vector_store, load_vector_store
from rag_core.bm25_index import build_bm25_index, load_bm25_index, bm25_retrieve
from rag_core.llm_client import call_llm

def build_knowledge_base(uploaded_files, chunk_size, overlap):
    documents = []
    for uploaded_file in uploaded_files:
        documents.extend(load_uploaded_file(uploaded_file))
    chunks = split_documents(documents, chunk_size=chunk_size, overlap=overlap)
    index = build_index(chunks)
    save_vector_store(index, chunks)
    build_bm25_index(chunks)
    return index, chunks

def load_knowledge_base():
    index, chunks = load_vector_store()
    load_bm25_index()
    return index, chunks

def hybrid_retrieve(question, index, chunks, top_k=4, alpha=0.5):
    from rag_core.vector_store import retrieve as vector_retrieve
    vector_results = vector_retrieve(question, index, chunks, top_k*2)
    from rag_core.bm25_index import bm25_retrieve
    bm25_results = bm25_retrieve(question, chunks, top_k*2)
    combined = {}
    for item in vector_results:
        key = item["chunk_id"]
        combined[key] = combined.get(key,0)+alpha*item["score"]
    for item in bm25_results:
        key = item["chunk"]["chunk_id"]
        combined[key] = combined.get(key,0)+(1-alpha)*item["score"]
    sorted_chunks = sorted([(chunks[int(cid)], score) for cid, score in combined.items()], key=lambda x:x[1], reverse=True)
    results = []
    for chunk, score in sorted_chunks[:top_k]:
        c = chunk.copy()
        c["score"] = float(score)
        results.append(c)
    return results

def answer_question(question, index, chunks, top_k=4):
    retrieved_chunks = hybrid_retrieve(question, index, chunks, top_k=top_k)
    answer = call_llm(question, retrieved_chunks)
    return answer, retrieved_chunks