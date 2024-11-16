from data_processing import DataProcessor
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import glob

def build_vectorstore():
    """构建向量数据库"""
    try:
        processor = DataProcessor()
        vectorstore = processor.init_or_load_vectorstore()
        if not vectorstore:
            raise ValueError("无法初始化向量存储！")
    except Exception as e:
        print(f"构建向量数据库时出错: {str(e)}")

if __name__ == "__main__":
    build_vectorstore() 