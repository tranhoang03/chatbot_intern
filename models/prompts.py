from typing import List

class PromptManager:
    @staticmethod
    def get_sql_prompt(tables: str, query: str) -> str:
        """Generate SQL prompt with table definitions"""
        return f"""
        Based on the context and question, create an appropriate SQL query.

        Context:
        {tables}

        Question: {query}

        Requirements:
        1. Only use tables and columns from the context
        2. Create accurate and efficient SQL query
        3. Use aggregation functions when needed
        4. Add appropriate WHERE conditions
        5. Sort results if necessary
        6. Use table aliases for clarity
        7. Add LIMIT for large result sets
        8. Use JOIN when needed
        9. Use GROUP BY for aggregations
        10. Add HAVING for post-aggregation filters
        11. Return only the SQL query, no explanations or metadata
        12. For list queries, use SELECT * with LIMIT
        13. For counting queries, use COUNT(*)
        14. For sorting, use ORDER BY with appropriate columns

        Return only the SQL query, no explanations or metadata.
        """
    
    @staticmethod
    def get_vector_prompt(context: list, query: str, history: str) -> str:
        """Generate vector search prompt"""
        return f"""
        {history}
        
        Dựa trên thông tin sau:
        {context}

        Câu hỏi: {query}

        Yêu cầu:
        1. Trả lời ngắn gọn và tự nhiên
        2. Chỉ sử dụng thông tin từ context
        3. Sử dụng ngôn ngữ đa dạng và thân thiện
        4. Nếu thiếu thông tin, nói "Không có đủ thông tin"
        5. Duy trì tính nhất quán với các câu trả lời trước
        6. Với danh sách, hiển thị rõ ràng từng mục
        7. Với kết quả tính toán, hiển thị số liệu cụ thể
        8. Tránh lặp lại cấu trúc câu trả lời
        9. Sử dụng dấu câu phù hợp
        10. Thêm từ ngữ thân thiện khi phù hợp
        """
    
    @staticmethod
    def get_sql_response_prompt(context: list, query: str, results: str, history: str) -> str:
        """Generate SQL response prompt"""
        return f"""
        {history}

        Dựa trên thông tin sau:
        {context}

        Kết quả tính toán:
        {results}

        Câu hỏi: {query}

        Yêu cầu:
        1. Trả lời ngắn gọn và tự nhiên
        2. Sử dụng cả thông tin từ context và kết quả tính toán
        3. Sử dụng ngôn ngữ đa dạng và thân thiện
        4. Nếu thiếu thông tin, nói "Không có đủ thông tin"
        5. Duy trì tính nhất quán với các câu trả lời trước
        6. Với danh sách, hiển thị rõ ràng từng mục
        7. Với kết quả tính toán, hiển thị số liệu cụ thể
        8. Tránh lặp lại cấu trúc câu trả lời
        9. Sử dụng dấu câu phù hợp
        10. Thêm từ ngữ thân thiện khi phù hợp
        """ 