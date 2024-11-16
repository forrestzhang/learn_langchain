from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import glob
import hashlib
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

class DataProcessor:
    def __init__(self):
        self.embedding_model = os.getenv('EMBEDDING_MODEL')
        self.vectorstore_path = os.getenv('VECTORSTORE_PATH')
        self.markdown_path = os.getenv('MARKDOWN_PATH')
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def process_single_file(self, file_path):
        """Process a single markdown file"""
        try:
            # Load document
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            # Split document
            split_docs = self.text_splitter.split_documents(documents)
            print(f"Generated {len(split_docs)} text chunks from {os.path.basename(file_path)}")
            
            return split_docs
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def init_or_load_vectorstore(self):
        """Initialize or load vector store incrementally"""
        try:
            # Get list of markdown files
            markdown_files = glob.glob(os.path.join(self.markdown_path, "**/*.md"), recursive=True)
            if not markdown_files:
                print(f"Error: No markdown files found in {self.markdown_path}")
                return None
            
            print(f"\nFound {len(markdown_files)} markdown files")
            
            # Initialize embeddings
            embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.ollama_base_url
            )
            
            # Load existing vectorstore if it exists
            if os.path.exists(self.vectorstore_path):
                print("Loading existing vector store...")
                vectorstore = FAISS.load_local(
                    self.vectorstore_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                print("Creating new vector store...")
                vectorstore = None
            
            # Process files incrementally
            for i, file_path in enumerate(markdown_files, 1):
                print(f"\nProcessing file {i}/{len(markdown_files)}: {os.path.basename(file_path)}")
                
                # Process single file
                split_docs = self.process_single_file(file_path)
                if not split_docs:
                    continue
                
                # Create or update vectorstore
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(split_docs, embeddings)
                else:
                    vectorstore.add_documents(split_docs)
                
                # Save after each file
                print("Saving vector store...")
                os.makedirs(self.vectorstore_path, exist_ok=True)
                vectorstore.save_local(self.vectorstore_path)
                
                # Add delay to prevent overloading
                time.sleep(1)
            
            print("\nVector store processing completed!")
            return vectorstore
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return None