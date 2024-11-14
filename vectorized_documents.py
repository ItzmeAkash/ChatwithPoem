from langchain_core.documents import Document
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings()

# Load data from the directory
loader = DirectoryLoader(path="poemsTxtFile", glob="*.txt", loader_cls=UnstructuredFileLoader)

# Load the documents
documents = loader.load()

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

# Split the documents into chunks
text_chunks = text_splitter.split_documents(documents)

# Define the connection string and collection name
CONNECTION_STRING = "postgresql+psycopg2://postgres:admin@localhost:5432/vectordb"
COLLECTION_NAME = "poems_vector"



# Initialize the PGVector vector store
vector_store = PGVector.from_documents(
    embedding=embeddings,
    documents=text_chunks,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING)

print("Vector store creation complete.")

