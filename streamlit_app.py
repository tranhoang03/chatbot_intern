import streamlit as st
from models.rag_system import OptimizedRAGSystem
from config import Config
import os
from dotenv import load_dotenv

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Load custom CSS
def load_css():
    with open('static/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load CSS
load_css()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG system
@st.cache_resource
def get_rag_system():
    config = Config()
    return OptimizedRAGSystem(config)

# Add a logo and title
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("<h1 class='stTitle'>🤖 RAG Chatbot</h1>", unsafe_allow_html=True)

# Initialize RAG system
rag_system = get_rag_system()

# Create a container for chat messages
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Add some spacing before the chat input
st.markdown("<br>" * 2, unsafe_allow_html=True)

# Chat input with custom styling
if prompt := st.chat_input("What would you like to know? 🤔"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = rag_system.answer_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Enhanced sidebar with better styling
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
        This is a RAG (Retrieval-Augmented Generation) chatbot that can answer questions based on the knowledge base.
    """)
    
    st.markdown("### Features")
    st.markdown("""
        - 🔍 Semantic search
        - 🧠 Context-aware responses
        - 📚 Knowledge base integration
    """)
    
    st.markdown("### How to use")
    st.markdown("""
        1. Type your question in the chat input
        2. Wait for the system to process
        3. Get your answer with relevant context
    """)
    
    # Add a separator
    st.markdown("---")
    
    # Add contact information
    st.markdown("""
        Need help? Contact us:
        
        📧 support@example.com
    """) 