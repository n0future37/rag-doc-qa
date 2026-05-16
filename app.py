import streamlit as st

from rag_core.rag_pipeline import (
    answer_question,
    build_knowledge_base,
    load_knowledge_base,
)


APP_TITLE = "RAG 智能文档问答系统"


def init_session_state():
    if "index" not in st.session_state:
        st.session_state.index = None

    if "chunks" not in st.session_state:
        st.session_state.chunks = None


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session_state()

    st.title(APP_TITLE)

    st.markdown(
        """
        本项目是一个基于 RAG 的智能文档问答系统，支持文档上传、文本切分、
        向量检索、来源引用和大模型问答。
        """
    )

    with st.sidebar:
        st.header("参数设置")

        chunk_size = st.slider("文本块大小", 300, 1000, 550, 50)
        overlap = st.slider("文本块重叠", 0, 300, 100, 20)
        top_k = st.slider("召回片段数量 Top-K", 1, 8, 4)

        st.divider()

        if st.button("加载本地知识库"):
            index, chunks = load_knowledge_base()

            if index is None or chunks is None:
                st.warning("本地还没有保存过知识库，请先上传文档并构建。")
            else:
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.success(f"已加载本地知识库，共 {len(chunks)} 个文本片段。")

    uploaded_files = st.file_uploader(
        "上传文档",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("构建并保存知识库", type="primary"):
            with st.spinner("正在解析文档、切分文本并构建向量索引..."):
                index, chunks = build_knowledge_base(
                    uploaded_files=uploaded_files,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

            st.session_state.index = index
            st.session_state.chunks = chunks

            st.success(f"知识库构建完成，已保存到本地，共 {len(chunks)} 个文本片段。")

    if st.session_state.index is None or st.session_state.chunks is None:
        st.info("请先上传文档并构建知识库，或从侧边栏加载本地知识库。")
        return

    st.divider()

    st.subheader("开始问答")

    question = st.text_input(
        "请输入你的问题",
        placeholder="例如：这份文档的核心观点是什么？",
    )

    if st.button("提交问题"):
        if not question.strip():
            st.warning("请先输入问题。")
            return

        with st.spinner("正在检索相关片段并调用大模型生成回答..."):
            answer, retrieved_chunks = answer_question(
                question=question,
                index=st.session_state.index,
                chunks=st.session_state.chunks,
                top_k=top_k,
            )

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("检索到的资料片段")

            for idx, item in enumerate(retrieved_chunks):
                page_text = f"第 {item['page']} 页" if item.get("page") else "无页码"

                title = (
                    f"资料片段 {idx + 1} | "
                    f"{item['file_name']} | "
                    f"{page_text} | "
                    f"相似度 {item['score']:.4f}"
                )

                with st.expander(title):
                    st.markdown(f"**文件名：** {item['file_name']}")
                    st.markdown(f"**页码：** {page_text}")
                    st.markdown(f"**片段编号：** {item['chunk_id']}")
                    st.markdown(f"**相似度：** {item['score']:.4f}")
                    st.write(item["content"])

        with right_col:
            st.subheader("模型回答")
            st.write(answer)


if __name__ == "__main__":
    main()