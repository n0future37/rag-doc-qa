# RAG 智能文档问答系统

## 项目简介

本项目是一个基于 RAG（Retrieval-Augmented Generation，检索增强生成）的智能文档问答系统。

系统支持用户上传 PDF、TXT、Markdown 文档，对文档进行文本解析、切分、向量化和相似度检索，并结合大语言模型生成基于原文依据的问答结果。

## 核心功能

- 支持 PDF / TXT / Markdown 文档上传
- 支持文档文本切分与 chunk 管理
- 使用 BGE 中文向量模型生成文本嵌入
- 使用 FAISS 构建本地向量索引
- 支持向量库本地持久化存储
- 支持 Top-K 相似度检索
- 接入大模型 API 生成文档问答结果
- 展示检索来源、页码、片段编号与相似度分数

## 技术栈

- Python
- Streamlit
- FAISS
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
│   ├── config.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── vector_store.py
│   ├── llm_client.py
│   └── rag_pipeline.py
│
├── data/
└── storage/