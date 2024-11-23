from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

def print_env_config():
    """Print environment configuration"""
    print("\n=== Environment Configuration ===")
    print(f"OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'Not set')}")
    print(f"LLM_MODEL: {os.getenv('LLM_MODEL', 'Not set')}")
    print(f"EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL', 'Not set')}")
    print(f"VECTORSTORE_PATH: {os.getenv('VECTORSTORE_PATH', 'Not set')}")
    print(f"RETRIEVAL_K: {os.getenv('RETRIEVAL_K', '4 (default)')}")
    print(f"RETRIEVAL_SCORE_THRESHOLD: {os.getenv('RETRIEVAL_SCORE_THRESHOLD', '0.5 (default)')}")
    print(f"RETRIEVAL_LAMBDA_MULT: {os.getenv('RETRIEVAL_LAMBDA_MULT', '0.7 (default)')}  # Diversity factor for MMR search")
    print("=" * 30 + "\n")

def validate_env_vars():
    """Validate required environment variables"""
    required_vars = [
        'OLLAMA_BASE_URL',
        'LLM_MODEL',
        'EMBEDDING_MODEL',
        'VECTORSTORE_PATH'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please check your .env file."
        )

class QueryEngine:
    def __init__(self):
        # Load and validate environment variables
        load_dotenv()
        validate_env_vars()
        print_env_config()
        
        # Load retrieval parameters from env
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", 4))
        self.retrieval_score_threshold = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", 0.5))
        self.retrieval_lambda_mult = float(os.getenv("RETRIEVAL_LAMBDA_MULT", 0.7))
        
        # Initialize components
        try:
            self._initialize_components()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize components: {str(e)}")
    
    def _initialize_components(self):
        """Initialize LLM, embeddings, and vector store"""
        # Initialize LLM for translation and QA
        self.llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("LLM_MODEL"),
            temperature=0.7
        )
        
        # Initialize embedding model
        self.embedding_model = OllamaEmbeddings(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("EMBEDDING_MODEL")
        )
        
        # Load vector store
        self.vectorstore = FAISS.load_local(
            os.getenv("VECTORSTORE_PATH"),
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        
        # Create QA chain
        self.qa_chain = self._create_qa_chain()
    
    def _translate_to_english(self, chinese_query: str) -> str:
        """Translate Chinese query to English using Ollama"""
        system_prompt = """You are a professional translator. 
        Translate the Chinese text to English accurately, keeping scientific terms unchanged.
        Only return the translated text, without any explanations."""
        
        try:
            response = self.llm.invoke(
                f"Translate this Chinese text to English: {chinese_query}"
            )
            translated = response.content.strip()
            print(f"\nTranslated query: {translated}")
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            return chinese_query
    
    def _is_chinese(self, text: str) -> bool:
        """Check if the text contains Chinese characters"""
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    def _create_qa_chain(self):
        """Create QA chain with configurable retrieval parameters"""
        prompt_template = """Use the following pieces of information to answer the question. If you cannot find the answer, say "I don't have enough information to answer that."
        Keep your answers professional and concise.

Context:
{context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Configure retriever with MMR search
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Using MMR instead of simple similarity
            search_kwargs={
                "k": self.retrieval_k,
                "fetch_k": self.retrieval_k * 4,  # Fetch more candidates for MMR
                "lambda_mult": self.retrieval_lambda_mult,  # Diversity factor from env
            }
        )
        
        # Debug: Test retriever directly
        def debug_retriever(query):
            print(f"\nExecuting search for query: {query}")
            docs = retriever.get_relevant_documents(query)
            print(f"\nDebug - Found {len(docs)} relevant documents")
            for i, doc in enumerate(docs, 1):
                print(f"\nDocument {i}:")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Content preview: {doc.page_content[:200]}...")
                if hasattr(doc, 'similarity_score'):
                    print(f"Similarity score: {doc.similarity_score}")
            return docs
        
        # Store retriever and debug function
        self.retriever = retriever
        self.debug_retriever = debug_retriever
        
        # Create chain with more specific configuration
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": True,
                "document_separator": "\n\n",  # Clear separation between documents
            },
            return_source_documents=True
        )
        
        return chain
    
    def _format_answer_with_citations(self, answer: str, sources: list) -> str:
        """Format answer with data sources only"""
        # Create citation map for unique sources
        unique_sources = list(dict.fromkeys(sources))  # Remove duplicates while preserving order
        citation_map = {
            os.path.basename(source): f"[{i+1}]" 
            for i, source in enumerate(unique_sources)
        }
        
        # Format answer with citations
        cited_answer = answer
        for source, citation in citation_map.items():
            if source in answer:
                cited_answer = cited_answer.replace(source, citation)
        
        # Add Data Sources section
        output = cited_answer + "\n\nData Sources:\n"
        for source in unique_sources:  # Use unique sources
            output += f"- {os.path.basename(source)}\n"
        
        return output
    
    def query(self, question: str):
        """Execute query with enhanced error handling and debugging"""
        try:
            # Normalize query
            normalized_query = question.strip().lower()
            
            # Translate Chinese query to English if needed
            if self._is_chinese(normalized_query):
                english_query = self._translate_to_english(normalized_query)
            else:
                english_query = normalized_query
            
            # Debug: Test retriever directly
            print("\nExecuting retrieval...")
            relevant_docs = self.debug_retriever(english_query)
            
            if not relevant_docs:
                return {
                    "answer": "No relevant documents found in the knowledge base. Please try rephrasing your question.",
                    "sources": []
                }
            
            # Get response from QA chain
            response = self.qa_chain.invoke({
                "query": english_query,
                "context": "\n\n".join(doc.page_content for doc in relevant_docs)
            })
            
            # Get sources from response and remove duplicates
            sources = []
            if response.get('source_documents'):
                sources = list(dict.fromkeys([
                    doc.metadata.get('source', 'Unknown source') 
                    for doc in response['source_documents']
                ]))
            
            # Format answer with citations
            formatted_answer = self._format_answer_with_citations(
                response['result'], 
                sources
            )
            
            return {
                "answer": formatted_answer,
                "sources": sources
            }
            
        except Exception as e:
            print(f"Query error: {str(e)}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return {
                "answer": "An error occurred during the query process. Please try again.",
                "sources": []
            }

def main():
    """Interactive query interface"""
    try:
        engine = QueryEngine()
        print("=== RAG Smart Q&A System Started ===")
        print("Tip: Enter your question to get answers, type 'quit' to exit")
        
        while True:
            question = input("\nQuestion: ")
            if question.lower() == 'quit':
                print("\nThank you for using! Goodbye!")
                break
            
            result = engine.query(question)
            print("\nAnswer:", result['answer'])
            
            # if result['sources']:
            #     print("\nReferences:")
            #     for i, source in enumerate(result['sources'], 1):
            #         print(f"{i}. {os.path.basename(source)}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main() 