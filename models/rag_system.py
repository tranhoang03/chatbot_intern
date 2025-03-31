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
                
                # Format column information
                column_info = []
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    column_info.append(f"{col_name} ({col_type})")
                
                schema_info.append(f"Bảng {table_name}:\n" + "\n".join(column_info))
            
            conn.close()
            return "\n\n".join(schema_info)
            
        except Exception as e:
            print(f"Error getting database schema: {e}")
            return ""

    def _generate_sql_from_context(self, query: str, context: List[str]) -> str:
        """Generate SQL query based on context"""
        # Get database schema
        schema_info = self._get_database_schema()
        
        # Create prompt with schema information
        prompt = f"""
        Bạn là một chuyên gia SQL. Hãy tạo một truy vấn SQL chính xác để trả lời câu hỏi của người dùng.

        Câu hỏi từ người dùng:
        "{query}"

        **Cấu trúc database hiện có:**
        {schema_info}

        **Yêu cầu:**
        1. Chỉ sử dụng các bảng và cột có trong cấu trúc database trên
        2. Chỉ tạo câu truy vấn SELECT
        3. Không sử dụng các từ khóa nguy hiểm (DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE)
        4. Đảm bảo truy vấn chạy được trên SQLite
        5. Nếu có điều kiện lọc dữ liệu, sử dụng `WHERE`
        6. Nếu cần sắp xếp, sử dụng `ORDER BY`
        7. Nếu cần nối bảng, sử dụng `JOIN` hợp lý
        8. Không giả định bất kỳ giá trị nào không có trong bảng

        **Quy tắc:**
        1. Chỉ trả về mã SQL, không có giải thích
        2. Không chứa Markdown code block
        3. Không thêm comment hay giải thích gì thêm
        4. Đảm bảo câu SQL đúng cú pháp
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Extract only the content from the response
            if hasattr(response, 'content'):
                sql_query = response.content.strip()
            else:
                sql_query = str(response).strip()
            
            print(f"\nGenerated SQL query: {sql_query}")  # Debug print
            
            # Validate SQL query
            if not validate_sql_query(sql_query):
                print(f"Invalid SQL query: {sql_query}")  # Debug print
                raise ValueError(f"Invalid SQL query generated: {sql_query}")
            
            # Test the query before returning
            test_results = execute_sql_query(self.config.db_path, sql_query)
            if test_results is None:
                raise ValueError(f"SQL query execution failed: {sql_query}")
                
            return sql_query
            
        except Exception as e:
            print(f"Error in SQL generation: {str(e)}")  # Debug print
            raise
    
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
            sql_query = self._generate_sql_from_context(query, [])
            print(f"Generated SQL query: {sql_query}")  # Print the SQL query
            
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
            
            # Generate natural language response
            prompt = f"""
            Dựa trên câu hỏi và kết quả SQL sau:

            Câu hỏi: {query}
            Kết quả SQL: {formatted_results}

            Yêu cầu:
            1. Trả lời câu hỏi dựa trên kết quả SQL
            2. Nếu câu hỏi về số lượng bảng, trả lời chính xác số lượng
            3. Nếu câu hỏi về thông tin khác, trả lời dựa trên kết quả SQL
            4. Không thêm thông tin không liên quan
            5. Trả lời ngắn gọn, rõ ràng
            """
            
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