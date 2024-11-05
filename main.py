import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()



# To fetch the directory
working_dir = os.path.dirname(os.path.abspath(__file__))

def vector_store():
    persist_directory = f"{working_dir}/vector_db"
    embeddings = HuggingFaceEmbeddings()
    # Load the vectordb
    vectorstore_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore_db

# Create a Conversational Chain with Memory
def conversational_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    retriever = vectorstore.as_retriever()
    
    memory = ConversationBufferMemory(llm=llm, output_key="answer", memory_key="chat_history", return_messages=True)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True
    )
    
    return chain

# Frontend using Streamlit
st.set_page_config(
    page_title="Chat with Poem",
    layout="centered"
)

st.title("Chat with Poem 📑")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = vector_store()
    
if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = conversational_chain(st.session_state.vectorstore)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for user question
user_input = st.chat_input("Ask anything...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain({"question": user_input})
        assistant_response = response['answer']
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
