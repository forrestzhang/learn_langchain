import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
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

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)


query = "How did Juliet die?"


retriever = db.as_retriever(search_type="similarity_score_threshold", 
                            search_kwargs={"k":3, "score_threshold": 0.4})

relevant_docs = retriever.invoke(query)

print("\n ----Relevant Documents----")
for i,doc in enumerate(relevant_docs, 1):
    
    print(f"Document {i}:\n{doc.page_content}\n")

    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


# docs = db.similarity_search(query)

# for i,doc in enumerate(docs, 1):
    
#     print(f"Document {i}:\n{doc.page_content}\n")

#     if doc.metadata:
#         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")