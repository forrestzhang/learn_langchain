import os

from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_ollama import ChatOllama

current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
# encode_kwargs = {'normalize_embeddings': True}
# model_name = "BAAI/bge-base-en-v1.5"
# embeddings = HuggingFaceEmbeddings(model_name=model_name, 
#                                     model_kwargs=model_kwargs,
#                                     encode_kwargs=encode_kwargs,
#                                     show_progress=True)
embeddings = OllamaEmbeddings(model='nomic-embed-text',
                                  base_url="http://gpuserver:11434",
                                  show_progress=True)

db = Chroma(persist_directory=persistent_directory, 
            embedding_function=embeddings)


query ="How to learn LangChain?"

retriever = db.as_retriever(search_type="mmr", 
                            search_kwargs={"k":3, "score_threshold": 0.4})

# retriever = db.as_retriever(search_type="mmr", 
#                             search_kwargs={"k":2})

relevant_docs = retriever.invoke(query)


print("\n ----Relevant Documents----")
for i, doc in enumerate(relevant_docs, 1):
    
    print(f"Document {i}:\n{doc.page_content}\n")

    if doc.metadata:

        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an anser base only on the provided documents. If hte answer is not in the documents, please say 'I don't know'"
)

print("\n--- Combined Input ---")
print(combined_input)

model = ChatOllama(model="llama3")

message = [
    SystemMessage(content="Your are a helpful assistant."),
    HumanMessage(content=combined_input)
]

result = model.invoke(message)

print("\n--- Generated Response ---")
print("Full result:")
print(result)
print("Countent only:")
print(result.content)