import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.vectorstores.pgvector import PGVector
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
# Load environment variables from .env file
load_dotenv()


llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    
embeddings = HuggingFaceEmbeddings()

COLLECTION_NAME = "poems_vector"
CONNECTION_STRING = "postgresql+psycopg2://postgres:admin@localhost:5432/vectordb"

db = PGVector.from_existing_index(
    embedding=embeddings,
    
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
)


retriver = db.as_retriever(search_type="similarity", search_kwargs={"k":1})

prompt_template = """
You are a Conversational AI assistant that provides responses based on the given context. If the answer is not present in the provided context, please respond with: "I'm sorry, but I don't have that information in the current context. If I receive an update, I will let you know. Please check back later."

Context:
{context}

Question:
{question}
"""

chat_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriver,
    return_source_documents=True,
    verbose=True,
    chain_type_kwargs={"prompt": chat_prompt}
)



response = qa.invoke("tell me a  poem that has 43 lines and there summary of the poem "

)

print(response['result'])