import os 

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# embedding = OllamaEmbeddings(model='nomic-embed-text')
model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': True}
model_name = "BAAI/bge-base-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=model_name, 
                                       model_kwargs=model_kwargs,
                                        encode_kwargs=encode_kwargs,
                                       show_progress=True)

# db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)


def query_vector_store(store_name, query, embedding_funcation,
                       search_type, search_kwargs):
    if os.path.exists(persistent_directory):
        print(f"----- Querying the Vector Store {store_name} -----")

        db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_funcation)

        retriever = db.as_retriever(search_type=search_type, 
                                    search_kwargs=search_kwargs)
        
        relevant_docs = retriever.invoke(query)

        print(f"--- Relevant Documents from {store_name} ---")

        for i,doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")

            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
                if doc.metadata:
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist")

query = "How did Juliet die?"

print("\n--- Using Similarity Search ---")
query_vector_store("chroma_db_with_metadata", query, embeddings,
                   "similarity", {"k":3})


print("\n--- Using Max Marginal Relevance (MMR) ---")
query_vector_store("chroma_db_with_metadata", query, embeddings,
                     "mmr", 
                     {"k":3, "fetch_k":20, "lambda_mult": 0.5})


print("\n--- Using Similarity Score Threshold ---")
query_vector_store("chroma_db_with_metadata", query, embeddings,
                     "similarity_score_threshold",
                     {"k":3, "score_threshold": 0.3})