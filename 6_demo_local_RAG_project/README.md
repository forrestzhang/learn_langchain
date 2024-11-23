# 6_demo_RAG_project

## 1. 项目介绍

- 项目名称：基于 langchain 的 RAG 智能问答系统
- 项目目标：实现一个基于 langchain 的 RAG 智能问答系统，能够根据用户的问题，从知识库中检索出相关的信息，并给出回答。

## 2. 项目架构

### 2.1 数据层

- 使用 langchain 的 RAG 技术，从知识库中检索相关信息
- 原始数据为 markdown 格式，加载文档
- 使用 RecursiveCharacterTextSplitter 逐个对文档进行分割，并进行向量化
- 使用 FAISS 向量数据库存储文档向量，数据按照增量方式存储，不添加重复数据

### 2.2 模型层

- LLM：使用 Ollama 部署的 Qwen2.5-7B-32k
- Embedding：使用 Ollama 部署的 mxbai-embed-large

### 2.3 检索层

- 使用 FAISS 进行向量检索
- 支持相似度搜索
- 支持上下文关联

## 3. 代码结构

- src/data_processing.py：数据处理模块，负责数据加载、分割、向量化、存储
- src/query.py：查询模块，负责根据用户的问题进行检索、问答
