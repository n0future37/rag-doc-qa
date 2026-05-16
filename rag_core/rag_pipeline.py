from rag_core.document_loader import load_uploaded_file
from rag_core.llm_client import call_llm
from rag_core.text_splitter import split_documents
from rag_core.vector_store import (
    build_index,
    load_vector_store,
    retrieve,
    save_vector_store,
)


def build_knowledge_base(uploaded_files, chunk_size: int, overlap: int):
    documents = []

    for uploaded_file in uploaded_files:
        documents.extend(load_uploaded_file(uploaded_file))

    chunks = split_documents(
        documents=documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    index = build_index(chunks)
    save_vector_store(index, chunks)

    return index, chunks


def load_knowledge_base():
    return load_vector_store()


def answer_question(question: str, index, chunks: list[dict], top_k: int):
    retrieved_chunks = retrieve(
        question=question,
        index=index,
        chunks=chunks,
        top_k=top_k,
    )

    answer = call_llm(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    return answer, retrieved_chunks