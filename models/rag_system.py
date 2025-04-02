from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
import sqlite3

from config import Config
from utils import (
    load_table_data,
    execute_sql_query,
    format_sql_results,
    validate_sql_query
)
from .chat_history import ChatHistory
from .prompts import PromptManager

class OptimizedRAGSystem:
    def __init__(self, config: Config):
        self.config = config
        self.chat_history = ChatHistory()
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all necessary components"""
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            google_api_key=self.config.google_api_key
        )
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> FAISS:
        """Initialize FAISS vector store"""
        # Check if vector store exists
        if os.path.exists(self.config.vector_store_path):
            try:
                return FAISS.load_local(
                    self.config.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading vector store: {e}")
                # Nếu load thất bại, tạo mới
                return self._create_new_vector_store()
        
        # Create new vector store
        return self._create_new_vector_store()
    
    def _create_new_vector_store(self) -> FAISS:
        """Create new vector store from database"""
        try:
            # Create vector store directory if it doesn't exist
            os.makedirs(self.config.vector_store_path, exist_ok=True)
            
            # Load data from database
            documents = load_table_data(self.config.db_path)
            
            if not documents:
                print("No documents loaded from database")
                return None
            
            # Create vector store
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            print(f"Creating vector store with {len(texts)} documents")
            
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # Save vector store
            vector_store.save_local(self.config.vector_store_path)
            print("Vector store created and saved successfully")
            
            return vector_store
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None
    
    def _needs_calculation(self, query: str) -> bool:
        """Check if query requires calculation using LLM"""
        prompt = f"""
        Bạn là một chuyên gia trong việc lựa chọn phương pháp để trả lời người dùng.Phân tích câu hỏi sau và quyết định xem nên sử dụng phương pháp nào để trả lời:

        Câu hỏi: {query}

        Hệ thống có 2 phương pháp để trả lời:
        1. Database (SQL) - Sử dụng khi cần:
           - Tính toán số liệu (tổng, trung bình, đếm, v.v.)
           - So sánh dữ liệu
           - Thống kê
           - Liệt kê danh sách
           - Sắp xếp dữ liệu
           - Lọc dữ liệu theo điều kiện
           - Truy vấn dữ liệu có cấu trúc rõ ràng

        2. Vector Store - Sử dụng khi cần:
           - Tìm kiếm thông tin theo ngữ nghĩa
           - Trả lời câu hỏi về nội dung chi tiết
           - Tìm kiếm thông tin không có cấu trúc rõ ràng
           - Trả lời câu hỏi mô tả, giải thích
           - Tìm kiếm thông tin liên quan đến từ khóa

        Yêu cầu:
        1. Phân tích câu hỏi và quyết định phương pháp phù hợp nhất
        2. Chỉ trả về "true" nếu nên dùng Database (SQL)
        3. Chỉ trả về "false" nếu nên dùng Vector Store
        4. Không giải thích thêm
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Extract only the content from the response
            if hasattr(response, 'content'):
                result = response.content.strip().lower()
            else:
                result = str(response).strip().lower()
            
            return result == "true"
            
        except Exception as e:
            print(f"Error in _needs_calculation: {e}")
            # Fallback to simple keyword check if LLM fails
            calculation_keywords = [
                "tính", "tổng", "trung bình", "số lượng", "count", "sum", "average",
                "nhiều nhất", "ít nhất", "max", "min", "so sánh", "thống kê",
                "danh sách", "liệt kê", "hiển thị", "show", "list", "display"
            ]
            return any(keyword in query.lower() for keyword in calculation_keywords)
    
    def _get_database_schema(self) -> str:
        """Get database schema information"""
        try:
            conn = sqlite3.connect(self.config.db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            schema_info = []
            for table in tables:
                table_name = table[0]
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # Get foreign keys
                cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                foreign_keys = cursor.fetchall()
                
                # Get indexes (including primary keys)
                cursor.execute(f"PRAGMA index_list({table_name})")
                indexes = cursor.fetchall()
                
                # Format column information
                column_info = []
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    is_pk = col[5] == 1  # Check if column is primary key
                    pk_info = " (PRIMARY KEY)" if is_pk else ""
                    column_info.append(f"{col_name} ({col_type}){pk_info}")
                
                # Format foreign key information
                fk_info = []
                for fk in foreign_keys:
                    ref_table = fk[2]  # Referenced table
                    from_col = fk[3]   # Column in this table
                    to_col = fk[4]     # Column in referenced table
                    fk_info.append(f"FOREIGN KEY ({from_col}) REFERENCES {ref_table}({to_col})")
                
                # Format index information
                index_info = []
                for idx in indexes:
                    idx_name = idx[1]
                    is_unique = idx[2] == 1
                    if not idx_name.startswith('sqlite_autoindex'):  # Skip auto-generated indexes
                        index_info.append(f"{'UNIQUE ' if is_unique else ''}INDEX {idx_name}")
                
                # Combine all information
                table_info = [f"Bảng {table_name}:"]
                table_info.extend(column_info)
                if fk_info:
                    table_info.append("\nKhóa ngoại:")
                    table_info.extend(fk_info)
                if index_info:
                    table_info.append("\nChỉ mục:")
                    table_info.extend(index_info)
                
                schema_info.append("\n".join(table_info))
            
            conn.close()
            return "\n\n".join(schema_info)
            
        except Exception as e:
            print(f"Error getting database schema: {e}")
            return ""

    
    def _answer_with_vector(self, query: str) -> str:
        """Answer query using only vector search"""
        try:
            # Get relevant documents
            docs = self.vector_store.similarity_search(
                query,
                k=self.config.top_k_results
            )
            
            # Extract context
            context = [doc.page_content for doc in docs]
            
            # Get recent chat history
            recent_history = self.chat_history.get_recent_history()
            
            # Generate natural language response
            prompt = PromptManager.get_vector_prompt(context, query, recent_history)
            
            response = self.llm.invoke(prompt)
            # Extract only the content from the response
            if hasattr(response, 'content'):
                return response.content.strip()
            return str(response).strip()
            
        except Exception as e:
            return f"Lỗi khi xử lý câu hỏi: {str(e)}"
    
    def _answer_with_sql(self, query: str) -> str:
        """Answer query using SQL"""
        try:
            # Generate SQL query directly from the question
            sql_query = PromptManager.get_sql_generation_prompt(query, self._get_database_schema())# Print the SQL query
            # Execute SQL query
            results = execute_sql_query(
                self.config.db_path,
                sql_query,
                self.config.db_timeout
            )
            
            # Format results
            formatted_results = format_sql_results(results)
            
            # Get recent chat history
            recent_history = self.chat_history.get_recent_history()
            
            # Generate natural language response using PromptManager
            prompt = PromptManager.get_sql_response_prompt(
                query=query,
                results=formatted_results,
                history=recent_history
            )
            response = self.llm.invoke(prompt)
            # Extract only the content from the response
            if hasattr(response, 'content'):
                return response.content.strip()
            return str(response).strip()
            
        except Exception as e:
            return f"Lỗi khi xử lý câu hỏi: {str(e)}"
    
    def answer_query(self, query: str) -> str:
        """Process query and return answer"""
        try:
            # Determine if calculation is needed
            needs_sql = self._needs_calculation(query)
            print(f"LLM decision: {'1' if needs_sql else '0'}")  # Print 1 for SQL, 0 for vector search
            
            if needs_sql:
                response = self._answer_with_sql(query)
            else:
                response = self._answer_with_vector(query)
            # Save to chat history
            self.chat_history.add_chat(query, response)
            
            return response
                
        except Exception as e:
            error_msg = f"Lỗi hệ thống: {str(e)}"
            self.chat_history.add_chat(query, error_msg)
            return error_msg 