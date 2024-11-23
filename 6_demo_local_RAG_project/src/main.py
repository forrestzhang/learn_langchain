from data_processing import DataProcessor
from model import ModelManager
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv

# Ensure loading .env file in the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, '.env')

# Load environment variables and print loading status
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"\n=== Environment variables loaded from: {env_path} ===")
else:
    print(f"\nWarning: Environment variables file not found: {env_path}")

# Print actual used environment variables
print("\n=== Environment variable configuration ===")
print(f"OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL')}")
print(f"LLM_MODEL: {os.getenv('LLM_MODEL')}")
print(f"EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL')}")
print(f"VECTORSTORE_PATH: {os.getenv('VECTORSTORE_PATH')}")
print(f"MARKDOWN_PATH: {os.getenv('MARKDOWN_PATH')}")
print("==================\n")

def init_vectorstore():
    """Initialize vector store"""
    processor = DataProcessor()
    vectorstore = processor.init_or_load_vectorstore()
    
    if vectorstore is None:
        raise ValueError("Unable to initialize vector store!")
    
    return vectorstore

def main():
    # Initialize vector store
    vectorstore = init_vectorstore()
    
    # Initialize model manager
    model_manager = ModelManager(vectorstore)
    
    # Create QA chain
    qa_chain = model_manager.create_qa_chain()
    
    # Interactive Q&A
    print("\n=== RAG Smart Q&A System Started ===")
    print("Tip: Enter your question to get answers, type 'quit' to exit")
    
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'quit':
            print("\nThank you for using! Goodbye!")
            break
        
        try:
            response = qa_chain.invoke({"query": question})
            print("\nAnswer:", response['result'])
            
            # Display source document information
            if 'source_documents' in response:
                print("\nReferences:")
                for i, doc in enumerate(response['source_documents'], 1):
                    print(f"{i}. {doc.metadata.get('source', 'Unknown source')}")
        except Exception as e:
            print(f"\nError occurred: {e}")
            print("Please try again or contact administrator")

if __name__ == "__main__":
    main() 