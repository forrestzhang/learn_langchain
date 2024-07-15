import os 

from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter,
                                     SentenceTransformersTokenTextSplitter,
                                     TextSplitter,
                                     TokenTextSplitter)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings


current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(db_dir):
    raise FileNotFoundError(f"DB directory {db_dir} not found")


# model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
# encode_kwargs = {'normalize_embeddings': True}
# model_name = "BAAI/bge-base-en-v1.5"
# embeddings = HuggingFaceEmbeddings(model_name=model_name, 
#                                        model_kwargs=model_kwargs,
#                                         encode_kwargs=encode_kwargs,
#                                        show_progress=True)
embeddings = OllamaEmbeddings(model='nomic-embed-text',
                                  base_url="http://gpuserver:11434",
                                  show_progress=True)

db = Chroma(persist_directory=db_dir, embedding_function=embeddings)


loader = TextLoader(file_path)
documents = loader.load()

def create_vector_store(docs, store_name):
    
    persistent_directory = os.path.join(db_dir, store_name)

    if not os.path.exists(persistent_directory):
        print(f"\n--- Crating vector sotre {store_name} ---")
        db = Chroma.from_documents(documents=docs, 
                                   embedding=embeddings, 
                                   persist_directory=persistent_directory)
        
        print(f"--- Vector store {store_name} created ---")
    else:
        print(f"Vector store {store_name} already exists")

# Useful for consisten chunk sizes regardless of content structure
print("\n --- Using Character-base Splitting ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")

# Ideal for maintaining semantic coherence within chunks
print("\n --- Using Sentence-base Splitting ---")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000, chunk_overlap=100)
sent_docs = sent_splitter.split_documents(documents)
create_vector_store(sent_docs, "chroma_db_sent")


# Useful for transformer models with strict token limits
print("\n --- Using Token-base Splitting ---")
token_splitter = TokenTextSplitter(chunk_overlap=100, chunk_size=500)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")

# Balances between maintaining coherence and adhering to character limits
# suggest for use
print("\n --- Using Recursive Character-base Splitting ---")
rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vector_store(rec_char_docs, "chroma_db_rec_char")



# Custome Splitting
# Allows creating custom splitting logic based on specific requirements.
# Useful for documents with unique structure that standard splitters can't handle.
print("\n --- Using Custom Splitting ---")

class CustomTextSplitter(TextSplitter):
    
    def split_text(self, text):
        # Custom splitting logic
        return text.split("\n\n")
    
custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom")


# Funcation to query a vector store
def query_vector_store(store_name, query):
    
    persistent_directory = os.path.join(db_dir, store_name)

    if os.path.exists(persistent_directory):
        print(f"\n --- Querying vector store {store_name} ---")
        
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

        retriever = db.as_retriever(search_type="similarity_score_threshold",
                                    search_kwargs={"k":3, "score_threshold": 0.4})
        
        relevant_docs = retriever.invoke(query)

        print(f"\n ----Relevant Documents for {store_name}----")

        for i,doc in enumerate(relevant_docs, 1):

            print(f"Document {i}:\n{doc.page_content}\n")

            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist")


query = "How did Juliet die?"
query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_rec_char", query)
query_vector_store("chroma_db_custom", query)
