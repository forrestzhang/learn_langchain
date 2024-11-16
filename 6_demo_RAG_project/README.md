# 6_demo_RAG_project

## 1. 项目介绍

- 项目名称：基于langchain的RAG的智能问答系统
- 项目目标：实现一个基于langchain的RAG的智能问答系统，能够根据用户的问题，从知识库中检索出相关的信息，并给出回答。

## 2. 项目架构

- 使用langchain的RAG技术，从知识库中检索出相关的信息。
- 使用langgraph将agent技术串联起来，形成一个完整的流程。
- 使用marimo来可视化整个流程。

## 3. 项目实现

环境配置信息存放在.env文件中，请根据实际情况进行配置。

### 3.1 数据层

- 使用langchain的RAG技术，从知识库中检索出相关的信息。
- 原始数据为markdown格式，使用langchain的loader技术，将markdown文件加载为document。数据存放在data/markdown_files/目录下。
- 使用langchain的text_splitter技术，将document切分为小的chunks。
- 使用langchain的vectorstore技术，将chunks向量化，并存储到faiss vectorstore中, faiss vectorstore存放在data/faiss_vectorstore/目录下。
- 使用langchain的retriever技术，从faiss vectorstore中检索出相关的信息。

### 3.2 模型层

- 大语言模型平台为Ollama，模型为qwen2.5:7b-32k, 可以在.env文件中进行配置，预留base_url。
- Embedding模型平台为Ollama，模型为mxbai-embed-large, 可以在.env文件中进行配置，预留base_url。

### 3.3 流程层

- 使用langgraph将agent技术串联起来，形成一个完整的流程。
- 使用marimo来可视化整个流程。
