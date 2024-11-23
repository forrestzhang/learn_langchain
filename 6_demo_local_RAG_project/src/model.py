from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

class ModelManager:
    def __init__(self, vectorstore):
        self.llm = Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("LLM_MODEL"),
            temperature=0.7,
            top_k=10,
            top_p=0.95,
            num_ctx=32768  # 32k context window
        )
        self.vectorstore = vectorstore
        
    def create_qa_chain(self):
        """创建增强的问答链"""
        prompt_template = """请基于以下已知信息，简洁专业地回答用户的问题。
如果无法从中得到答案，请说 "抱歉，我无法从已知信息中找到相关答案。"
请不要编造任何信息。

已知信息：
{context}

用户问题：{question}

回答："""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 4,  # 检索 4 个最相关的文档片段
                    "score_threshold": 0.5  # 相似度阈值
                }
            ),
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": True
            },
            return_source_documents=True  # 返回源文档信息
        )
        
        return chain 