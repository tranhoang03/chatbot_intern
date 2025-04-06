import streamlit as st
from models.rag_system import OptimizedRAGSystem
from models.face_auth import FaceAuthTransformer
from config import Config
import os
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import time

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session state ---
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
if "webrtc_active" not in st.session_state:
    st.session_state.webrtc_active = False
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()
if "show_greeting" not in st.session_state:
    st.session_state.show_greeting = False

load_dotenv()

def load_css():
    try:
        with open('static/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found.")

load_css()

def get_purchase_history(user_id: int) -> list:
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

@st.cache_resource
def get_system():
    config = Config()
    return OptimizedRAGSystem(config)

def create_transformer(q: queue.Queue):
    return FaceAuthTransformer(result_queue=q)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 class='stTitle'>🤖 RAG Chatbot</h1>", unsafe_allow_html=True)

rag_system = get_system()

if not st.session_state.authenticated:
    st.markdown("### 👤 Vui lòng nhìn vào camera để xác thực")

    current_queue = st.session_state.result_queue

    webrtc_ctx = webrtc_streamer(
        key="face-auth",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": {"frameRate": {"ideal": 10}},
            "audio": False,
        },
        video_processor_factory=lambda: create_transformer(current_queue),
        async_processing=True
    )

    try:
        user_info = st.session_state.result_queue.get(block=False)
        
        if user_info: 
            st.session_state.user_info = user_info
            st.session_state.authenticated = True
            st.session_state.show_greeting = True
            if webrtc_ctx and webrtc_ctx.state.playing:
                webrtc_ctx.stop()
                st.session_state.webrtc_active = False
                
            st.rerun()
        else:
            st.error("Xác thực thất bại. Không tìm thấy thông tin hoặc có lỗi xảy ra.")
            if webrtc_ctx and webrtc_ctx.state.playing:
                 webrtc_ctx.stop()

    except queue.Empty:
        if webrtc_ctx and webrtc_ctx.state.playing:
            st.info("Đang chờ nhận diện khuôn mặt...") 
        pass 
    except Exception as e:
        st.error(f"Lỗi trong quá trình xác thực: {e}")
        if webrtc_ctx and webrtc_ctx.state.playing:
            webrtc_ctx.stop()

if st.session_state.authenticated:
    if st.session_state.get("show_greeting", False):
        user_info = st.session_state.user_info 
        if user_info:
            st.markdown(f"""
            <div style='text-align: center; padding: 20px;'>
                <h2>👋 Xin chào {user_info['name']}!</h2>
                <p style='font-size: 1.2em;'>Rất vui được gặp lại bạn.</p>
            </div>
            """, unsafe_allow_html=True)

            purchase_history = get_purchase_history(user_info['id'])
            st.session_state.purchase_history = purchase_history
            purchase_history_text = ""
            if purchase_history:
                purchase_history_text = "\nLịch sử mua hàng gần đây:\n"
                for date, product, quantity, price, rate in purchase_history:
                    purchase_history_text += f"- {date}: {product} (SL: {quantity}, Giá: {price}đ, Đánh giá: {rate}⭐)\n"
            st.session_state.system_prompt = f"""
            Bạn đang trò chuyện với khách hàng {user_info['name']} (ID: {user_info['id']}).\n
            {purchase_history_text}
            """
            st.session_state.show_greeting = False
        else: 
            st.session_state.show_greeting = False 
            st.warning("Đã xảy ra lỗi khi hiển thị lời chào.")

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Bạn cần tôi giúp gì? 🤔"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang xử lý..."):
                query_to_send = prompt
                if st.session_state.system_prompt:
                    query_to_send = f"{st.session_state.system_prompt}\n\nCâu hỏi của khách hàng: {prompt}"
                response = rag_system.answer_query(query_to_send)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        st.markdown("### 👤 Thông tin người dùng")
        if st.session_state.user_info:
            st.markdown(f"**Tên:** {st.session_state.user_info['name']}\n\n**ID:** {st.session_state.user_info['id']}")
        else:
            st.markdown("Không có thông tin người dùng.")

        st.markdown("### 🛍️ Lịch sử mua hàng")
        if st.session_state.purchase_history:
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
        else:
            st.markdown("Chưa có lịch sử mua hàng.")

        if st.button("🚪 Đăng xuất"):
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.session_state.system_prompt = None
            st.session_state.purchase_history = None
            st.session_state.messages = []
            st.session_state.webrtc_active = False
            while not st.session_state.result_queue.empty():
                try:
                    st.session_state.result_queue.get(block=False)
                except queue.Empty:
                    continue
            st.rerun()

with st.sidebar:
    if st.session_state.authenticated:
        st.markdown("***")

    st.markdown("### About")
    st.markdown("This is a RAG (Retrieval-Augmented Generation) chatbot that can answer questions based on the knowledge base.")

    st.markdown("### Features")
    st.markdown("""
        - 🔍 Semantic search  
        - 🧠 Context-aware responses  
        - 📚 Knowledge base integration  
        - 👤 Personalized recommendations (via Face Auth)  
        - 📸 Camera Access via WebRTC
    """)

    st.markdown("### How to use")
    st.markdown("""
        1. Allow camera access for face authentication.  
        2. Look at the camera until recognized.  
        3. Type your question in the chat input.  
        4. Wait for the system to process.  
        5. Get your answer with relevant context.
    """)

    st.markdown("---")
    st.markdown("Need help? Contact us:\n📧 tranhoang0320@gmail.com")
