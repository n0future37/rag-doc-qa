# RAG 智能文档问答系统

## 项目简介
本项目是一个基于 RAG（Retrieval-Augmented Generation，检索增强生成）的智能文档问答系统，支持用户上传 PDF、TXT、Markdown 文档，并结合向量检索 + BM25 多路召回，提供高质量问答服务。  
系统能够显示来源文件、页码、chunk 编号和相似度评分，保证回答的可追溯性。

## 核心功能
- PDF / TXT / Markdown 文档上传解析
- 文本切分与 chunk 管理
- BGE 中文向量模型生成文本嵌入
- FAISS 构建本地向量索引
- BM25 倒排索引多路召回
- 向量库 + BM25 索引本地持久化
- Top-K 相似度检索 + 可选 Reranker 重排序
- 回答中展示来源信息与相似度
- Streamlit 前端展示问答界面

## 技术栈
- Python
- Streamlit
- FAISS
- rank_bm25
- Sentence-Transformers
- BGE Embedding
- pypdf
- Right Code API

## 项目结构
```text
rag-doc-qa/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── rag_core/
│ ├── init.py
│ ├── config.py
│ ├── document_loader.py
│ ├── text_splitter.py
│ ├── vector_store.py
│ ├── bm25_index.py
│ ├── llm_client.py
│ └── rag_pipeline.py
│
├── data/
└── storage/