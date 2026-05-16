def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)

def split_documents(documents, chunk_size=550, overlap=100):
    chunks = []
    global_chunk_id = 0
    for doc in documents:
        text = clean_text(doc["text"])
        start = 0
        local_chunk_id = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": global_chunk_id,
                    "local_chunk_id": local_chunk_id,
                    "file_name": doc["file_name"],
                    "page": doc.get("page"),
                    "content": chunk_text,
                })
                global_chunk_id += 1
                local_chunk_id += 1
            start += chunk_size - overlap
    return chunks