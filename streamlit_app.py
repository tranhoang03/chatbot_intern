import streamlit as st
from models.rag_system import OptimizedRAGSystem
from config import Config
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG system
@st.cache_resource
def get_rag_system():
    config = Config()
    return OptimizedRAGSystem(config)

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 RAG Chatbot")

# Initialize RAG system
rag_system = get_rag_system()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_system.answer_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Add a sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a RAG (Retrieval-Augmented Generation) chatbot that can answer questions based on the knowledge base.
    
    ### Features:
    - Semantic search
    - Context-aware responses
    - Knowledge base integration
    
    ### How to use:
    1. Type your question in the chat input
    2. Wait for the system to process
    3. Get your answer with relevant context
    """) 