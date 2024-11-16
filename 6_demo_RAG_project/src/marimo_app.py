import marimo

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __():
    import os
    from dotenv import load_dotenv
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    return FAISS, ChatOllama, OllamaEmbeddings, PromptTemplate, RetrievalQA, load_dotenv, os

@app.cell
def __(FAISS, ChatOllama, OllamaEmbeddings, PromptTemplate, RetrievalQA, load_dotenv, os):
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    embedding_model = OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=os.getenv("EMBEDDING_MODEL")
    )
    
    # Load vector store
    vectorstore = FAISS.load_local(
        os.getenv("VECTORSTORE_PATH"),
        embedding_model,
        allow_dangerous_deserialization=True
    )
    
    # Initialize LLM
    llm = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.7
    )
    
    # Create prompt template
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
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": int(os.getenv("RETRIEVAL_K", 4)),
                "fetch_k": int(os.getenv("RETRIEVAL_K", 4)) * 4,
                "lambda_mult": float(os.getenv("RETRIEVAL_LAMBDA_MULT", 0.7)),
            }
        ),
        chain_type_kwargs={
            "prompt": PROMPT,
            "verbose": True
        },
        return_source_documents=True
    )
    return qa_chain, vectorstore

@app.cell
def __(qa_chain):
    # Interactive query interface
    question = "What is uORF?"
    result = qa_chain({"query": question})
    
    # Format answer with sources
    answer = result["result"]
    sources = [doc.metadata.get('source', 'Unknown source') 
              for doc in result.get('source_documents', [])]
    
    # Display results
    print("\nQuestion:", question)
    print("\nAnswer:", answer)
    print("\nData Sources:")
    for source in sources:
        print(f"- {os.path.basename(source)}")
    return answer, sources

if __name__ == "__main__":
    app.run() 