import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.realpath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# check chroma db exists
if not os.path.exists(persistent_directory):

    print("Presistent directory does not exits. Creating Chroma DB")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"Books directory {books_dir} not found")
    
    book_fils = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    documents = []

    for book_file in book_fils:
        file_path = os.path.join(books_dir, book_file)

        loader = TextLoader(file_path)

        book_docs = loader.load()

        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = text_splitter.split_documents(documents)

    print("\n ----Document Chunks Information----")
    print(f"Number of documents: {len(docs)}")

    # embeddings = OllamaEmbeddings(model='nomic-embed-text',
    #                               base_url="http://gpuserver:11434",
    #                               show_progress=True)
    model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}
    model_name = "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, 
                                       model_kwargs=model_kwargs,
                                        encode_kwargs=encode_kwargs,
                                       show_progress=True)

    db = Chroma.from_documents(documents=docs, 
                               embedding=embeddings, 
                               persist_directory=persistent_directory)
    
else:
    print("Chroma DB already exists")    