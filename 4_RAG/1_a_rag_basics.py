import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


# check chroma db exists
if not os.path.exists(persistent_directory):
    print("Presistent directory does not exits. Creating Chroma DB")
    
    #os.makedirs(persistent_directory)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    
    # Read the text from file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print("\n ----Document Chunks Information----")
    print(f"Number of documents: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create a Chroma Vector Store

    embedding = OllamaEmbeddings(model='nomic-embed-text')

    db = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persistent_directory)
    print("-------Chroma DB created-------")

else:
    print("Chroma DB already exists")
 