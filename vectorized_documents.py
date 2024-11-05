from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import nltk
nltk.download('averaged_perceptron_tagger')

#loading the embedding models
embeddings = HuggingFaceEmbeddings()

#loading the Data from  Directory
loader = DirectoryLoader(path="poemsTxtFile",glob="./*.txt",loader_cls=UnstructuredFileLoader)

# loading the Document
documents  = loader.load()

text_spliter = CharacterTextSplitter(chunk_size=2000,chunk_overlap=500)

text_chunks = text_spliter.split_documents(documents)

vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="vector_db"
)

print("Documents Vectorized")
