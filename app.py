import os
import json
from io import BytesIO

import faiss
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


load_dotenv()


APP_TITLE = "RAG 智能文档问答系统"


@st.cache_resource
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)


def read_uploaded_file(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        pages = []

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[文件：{uploaded_file.name}｜第 {page_idx + 1} 页]\n{text}")

        return "\n\n".join(pages)

    if file_name.endswith(".txt") or file_name.endswith(".md"):
        raw = uploaded_file.getvalue()
        return f"[文件：{uploaded_file.name}]\n" + raw.decode("utf-8", errors="ignore")

    raise ValueError("暂时只支持 PDF / TXT / Markdown 文件。")


def split_text(text: str, chunk_size: int = 550, overlap: int = 100) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_faiss_index(chunks: list[str], embedder):
    embeddings = embedder.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, embeddings


def retrieve(question: str, chunks: list[str], index, embedder, top_k: int = 4):
    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append(
            {
                "score": float(score),
                "content": chunks[idx],
            }
        )

    return results

def call_right_code_llm(question: str, retrieved_chunks: list[dict]) -> str:
    api_key = os.getenv("RIGHT_CODE_API_KEY")
    model = os.getenv("RIGHT_CODE_MODEL", "gpt-5.2")
    base_url = os.getenv(
        "RIGHT_CODE_BASE_URL",
        "https://www.right.codes/codex/v1/responses",
    )

    if not api_key:
        return "未检测到 RIGHT_CODE_API_KEY，请先在 .env 文件中配置。"

    context_text = "\n\n".join(
        [
            f"【资料片段 {idx + 1}｜相似度 {item['score']:.4f}】\n{item['content']}"
            for idx, item in enumerate(retrieved_chunks)
        ]
    )

    prompt = f"""
你是一个严谨的文档问答助手。请只根据下面的“参考资料”回答用户问题。

要求：
1. 只能使用参考资料中的信息，不要编造。
2. 如果资料中没有答案，请明确说“根据当前文档无法判断”。
3. 回答要清晰、分点，适合中文用户阅读。
4. 结尾列出你主要参考了哪些资料片段编号。

参考资料：
{context_text}

用户问题：
{question}

请给出答案：
""".strip()

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
        "stream": True,
    }

    try:
        response = requests.post(
            base_url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=120,
        )
    except requests.RequestException as exc:
        return f"请求 Right Code API 失败：{exc}"

    if response.status_code != 200:
        return f"Right Code API 调用失败，状态码：{response.status_code}\n\n{response.text}"

    answer_parts = []

    # 关键：不要让 requests 自动猜编码，手动按 utf-8 解码
    for raw_line in response.iter_lines(decode_unicode=False):
        if not raw_line:
            continue

        try:
            line = raw_line.decode("utf-8", errors="replace").strip()
        except Exception:
            continue

        if not line.startswith("data:"):
            continue

        line = line[len("data:"):].strip()

        if line == "[DONE]":
            break

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Right Code / responses 流式格式
        if data.get("type") == "response.output_text.delta":
            answer_parts.append(data.get("delta", ""))

        # 兜底：兼容部分 OpenAI chat.completions 流式格式
        choices = data.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})
            if isinstance(delta, dict) and delta.get("content"):
                answer_parts.append(delta["content"])

            message = choice.get("message", {})
            if isinstance(message, dict) and message.get("content"):
                answer_parts.append(message["content"])

    final_answer = "".join(answer_parts).strip()

    if not final_answer:
        return "模型没有返回有效内容。请检查模型名、API Key 权限、余额，或 Right Code 后台的模型可用权限。"

    return final_answer

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.markdown(
        """
        这是一个第 1 天版本的 RAG 文档问答 Demo：  
        上传文档 → 切分文本 → BGE 向量化 → FAISS 检索 → Right Code 大模型回答。
        """
    )

    with st.sidebar:
        st.header("参数设置")
        chunk_size = st.slider("文本块大小", 300, 1000, 550, 50)
        overlap = st.slider("文本块重叠", 0, 300, 100, 20)
        top_k = st.slider("召回片段数量 Top-K", 1, 8, 4)

        st.caption("第 1 天版本优先保证能跑，后续再做多路召回、重排和评测。")

    uploaded_files = st.file_uploader(
        "上传文档",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("请先上传 PDF / TXT / Markdown 文档。")
        return

    all_texts = []

    for uploaded_file in uploaded_files:
        try:
            text = read_uploaded_file(uploaded_file)
            all_texts.append(text)
        except Exception as exc:
            st.error(f"读取文件 {uploaded_file.name} 失败：{exc}")

    corpus = "\n\n".join(all_texts)

    if not corpus.strip():
        st.warning("没有从文档中读取到有效文本。")
        return

    chunks = split_text(corpus, chunk_size=chunk_size, overlap=overlap)

    st.success(f"文档读取完成，共切分为 {len(chunks)} 个文本片段。")

    embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    with st.spinner("正在加载向量模型并构建索引，首次运行会稍慢..."):
        embedder = load_embedding_model(embedding_model_name)
        index, _ = build_faiss_index(chunks, embedder)

    question = st.text_input("请输入你的问题", placeholder="例如：这份文档的核心观点是什么？")

    if st.button("开始问答", type="primary"):
        if not question.strip():
            st.warning("请先输入问题。")
            return

        with st.spinner("正在检索相关文档片段..."):
            retrieved_chunks = retrieve(
                question=question,
                chunks=chunks,
                index=index,
                embedder=embedder,
                top_k=top_k,
            )

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("检索到的资料片段")
            for idx, item in enumerate(retrieved_chunks):
                with st.expander(f"资料片段 {idx + 1}｜相似度 {item['score']:.4f}"):
                    st.write(item["content"])

        with right_col:
            st.subheader("模型回答")
            with st.spinner("正在调用 Right Code 大模型生成回答..."):
                answer = call_right_code_llm(question, retrieved_chunks)
            st.write(answer)


if __name__ == "__main__":
    main()