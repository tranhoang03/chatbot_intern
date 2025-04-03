import streamlit as st
from models.rag_system import OptimizedRAGSystem
from models.face_auth import authenticate_user
from config import Config
import os
from dotenv import load_dotenv
import sqlite3
from datetime import datetime

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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = None
if "purchase_history" not in st.session_state:
    st.session_state.purchase_history = None

def get_purchase_history(user_id: int) -> list:
    """Get purchase history for a user"""
    try:
        conn = sqlite3.connect("Database.db")
        cursor = conn.cursor()
        
        query = """
        SELECT o.Order_date, p.Name, od.Quantity, od.Price, od.Rate
        FROM Orders o
        JOIN Order_detail od ON o.Id = od.Order_id
        JOIN Product p ON od.Product_id = p.Id
        WHERE o.Customer_id = ?
        ORDER BY o.Order_date DESC
        LIMIT 5
        """
        
        cursor.execute(query, (user_id,))
        results = cursor.fetchall()
        conn.close()
        
        return results
    except Exception as e:
        print(f"Error getting purchase history: {e}")
        return []

# Initialize RAG system
@st.cache_resource
def get_system():
    config = Config()
    return OptimizedRAGSystem(config)

# Add a logo and title
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("<h1 class='stTitle'>🤖 RAG Chatbot</h1>", unsafe_allow_html=True)

# Initialize system
rag_system = get_system()

# Authentication section
# Check if running on Streamlit Cloud by checking for a specific environment variable
is_streamlit_cloud = os.getenv("STREAMLIT_SERVER_PORT") is not None

if not st.session_state.authenticated:
    if is_streamlit_cloud:
        # Bypass authentication on Streamlit Cloud
        st.session_state.authenticated = True
        st.session_state.user_info = {"name": "Cloud User", "id": 0} # Example default user
        st.warning("Face authentication is disabled on Streamlit Cloud. Proceeding as default user.")
    else:
        # Face authentication (only run locally)
        user_info = authenticate_user()

        if user_info:
            st.session_state.user_info = user_info
            st.session_state.authenticated = True

            # Get purchase history
            purchase_history = get_purchase_history(user_info['id'])
            st.session_state.purchase_history = purchase_history

            # Set up personalized system prompt
            purchase_history_text = ""
            if purchase_history:
                purchase_history_text = "\nLịch sử mua hàng gần đây:\n"
                for date, product, quantity, price, rate in purchase_history:
                    purchase_history_text += f"- {date}: {product} (SL: {quantity}, Giá: {price}đ, Đánh giá: {rate}⭐)\n"

            st.session_state.system_prompt = f"""
            Bạn đang trò chuyện với khách hàng {user_info['name']} (ID: {user_info['id']}).\n
            Lịch sử mua hàng gần đây của khách: {purchase_history_text}
           
            """
        else:
            # Keep authenticated as False if face auth fails locally
            st.error("Authentication failed.")

# Main chat interface
if st.session_state.authenticated:
    # Create a container for chat messages
    chat_container = st.container()

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Bạn cần tôi giúp gì? 🤔"):
        # Add user message and display immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang xử lý..."):
                enhanced_prompt = f"{st.session_state.system_prompt}\n\nCâu hỏi của khách hàng: {prompt}"
                response = rag_system.answer_query(enhanced_prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar with user information
    with st.sidebar:
        if st.session_state.user_info: # Check if user_info exists before displaying
            st.markdown("### 👤 Thông tin người dùng")
            st.markdown(f"""
            **Tên:** {st.session_state.user_info['name']}
            **ID:** {st.session_state.user_info['id']}
            """)

            # Display purchase history
            if st.session_state.purchase_history:
                st.markdown("### 🛍️ Lịch sử mua hàng")
                for date, product, quantity, price, rate in st.session_state.purchase_history:
                    st.markdown(f"""
                    <div class="purchase-history-item">
                        <strong>{date}</strong><br>
                        Sản phẩm: {product}<br>
                        Số lượng: {quantity}<br>
                        Giá: {price}đ<br>
                        Đánh giá: {rate}⭐
                    </div>
                    """, unsafe_allow_html=True)

            # Add a logout button
            if st.button("🚪 Đăng xuất"):
                st.session_state.authenticated = False
                st.session_state.user_info = None
                st.session_state.system_prompt = None
                st.session_state.purchase_history = None
                st.session_state.messages = []
                st.rerun()
        else: # Handle case where authentication was bypassed
             st.markdown("### 👤 Thông tin người dùng")
             st.markdown("Running on Streamlit Cloud (No specific user authenticated).")

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
        - 👤 Personalized recommendations
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