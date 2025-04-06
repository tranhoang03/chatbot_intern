import json
import sqlite3
from typing import List, Dict, Any, Tuple
import base64
import os

def load_table_data(db_path: str) -> List[Dict[str, Any]]:
    """Load data from all tables in the database and format for vector store"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        documents = []
        
        # Define the Vietnamese column name mapping with diacritics and spaces
        column_name_mapping = {
            # Bảng Categories
            "Id": "id danh mục",
            "Name": "tên danh mục", 
            "Description": "mô tả danh mục",

            # Bảng Product
            "Categories_id": "id danh mục",
            # "Id": "id sản phẩm",          # Trùng key "Id"
            # "Name": "tên sản phẩm",        # Trùng key "Name"
            "Product_Prep": "thành phần sản phẩm",
            "Calories": "calo",
            "Dietary_Fibre_g": "chất xơ",
            "Sugars_g": "đường",
            "Protein_g": "protein",
            "Vitamin_A": "vitamin A",
            "Vitamin_C": "vitamin C",
            "Caffeine_mg": "caffeine",
            "Rating": "đánh giá",
            "Descriptions": "mô tả sản phẩm",
            "Link_Image": "link ảnh",

            # Bảng Store
            # "Id": "id cửa hàng",          # Trùng key "Id"
            # "Name": "tên cửa hàng",        # Trùng key "Name"
            "Address": "địa chỉ",
            "Phone": "số điện thoại",
            "Open_Close": "giờ mở cửa đóng cửa",

            # Bảng Orders
            # "Id": "id đơn hàng",          # Trùng key "Id"
            "Customer_id": "id khách hàng",
            "Store_id": "id cửa hàng",
            "Order_date": "ngày đặt hàng",

            # Bảng Order_detail
            "Order_id": "id đơn hàng",
            "Product_id": "id sản phẩm",
            "Quantity": "số lượng",
            "Price": "đơn giá",
            "Rate": "đánh giá", # Hoặc "đánh giá"

            # Bảng Customer_preferences
            # "Customer_id": "id khách hàng", # Đã có
            "Preferred_categories": "danh mục ưa thích",
            "Max_price": "giá tối đa",

            # Bảng customers
            "id": "id khách hàng", 
            "name": "tên khách hàng",
            "sex": "giới tính",
            "age": "tuổi",
            "location": "địa chỉ", # Giả định location là địa chỉ
            "picture": "ảnh",             # Sẽ được bỏ qua
            "embedding": "embedding"      # Sẽ được bỏ qua
        }

        print("\n=== Loading Data for Vector Store ===")
        for table_tuple in tables:
            table_name = table_tuple[0]
            # Standardize table name for lookup (e.g., lowercase)
            # lookup_table_name = table_name.lower()
            print(f"Processing table: {table_name}")
            
            # Skip sqlite sequence table
            if table_name == 'sqlite_sequence':
                continue

            cursor.execute(f"PRAGMA table_info({table_name});")
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]
            
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()
            
            print(f"  - Found {len(rows)} rows")
            
            # Convert each row to a document
            for row_idx, row in enumerate(rows):
                # Create a dictionary of column names and values
                row_dict = {}
                for col_name, val in zip(column_names, row):
                    # Skip specific columns like embeddings or pictures
                    # Check against original column names
                    if (table_name == "customers" and col_name in ["embedding", "picture"]):
                        continue
                    # Add other skips if needed, e.g., embedding in products
                    # if table_name == "Product" and col_name == "embedding":
                    #     continue
                    row_dict[col_name] = val
                
                # Create content string with Vietnamese names if available
                content_parts = []
                for k, v in row_dict.items():
                    # Use Vietnamese name from mapping, fallback to original name
                    # Use original column name (k) for lookup in mapping
                    display_name = column_name_mapping.get(k, k) 
                    # Format nicely, handling None values
                    value_str = str(v) if v is not None else "không có"
                    content_parts.append(f"{display_name}: {value_str}")

                # Include table name for context
                content = f"Bảng {table_name}: " + ", ".join(content_parts)
                
                # Create metadata
                metadata = {
                    "table": table_name,
                    # Store original column names in metadata
                    "columns": list(row_dict.keys()), 
                    "data": row_dict, # Original data with original keys
                    "original_row_index": row_idx 
                }
                
                documents.append({
                    "content": content,
                    "metadata": metadata
                })
        
        print("\n=== Summary ===")
        print(f"Total documents created: {len(documents)}")
        print("="*50)
        
        conn.close()
        return documents
        
    except Exception as e:
        print(f"Error loading table data: {e}")
        return []

def create_document_content(table_name: str, columns: List[str], row: Tuple) -> str:
    """Create a text representation of a database row"""
    content = [f"Table: {table_name}"]
    for col, val in zip(columns, row):
        content.append(f"{col}: {val}")
    return "\n".join(content)


def execute_sql_query(db_path: str, query: str, timeout: int = 30) -> List[Dict[str, Any]]:
    """Execute SQL query and return results"""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query)
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Fetch results
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        results = []
        for row in rows:
            result_dict = dict(zip(columns, row))
            results.append(result_dict)
        
        conn.close()
        return results
        
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return []

def format_sql_results(results: List[Dict[str, Any]]) -> str:
    """Format SQL results into a readable string"""
    if not results:
        return "Không tìm thấy kết quả"
    
    formatted_results = []
    for result in results:
        # Format each result as a string of key-value pairs
        result_str = ", ".join([
            f"{k}: {v}" for k, v in result.items()
        ])
        formatted_results.append(result_str)
    
    return "\n".join(formatted_results)

def validate_sql_query(query: str) -> bool:
    """Validate SQL query"""
    try:
        # Basic validation
        if not query or not query.strip():
            print("Empty query")
            return False
        
        # Remove comments and whitespace
        query = ' '.join(query.split())
        query_upper = query.upper()
        
        # Check for dangerous keywords
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
        if any(keyword in query_upper for keyword in dangerous_keywords):
            print(f"Dangerous keyword found in query: {query}")
            return False
        
        # Check if it's a SELECT query
        if not query_upper.startswith("SELECT"):
            print(f"Query is not a SELECT statement: {query}")
            return False
            
        # Check for basic SQL syntax
        if "FROM" not in query_upper:
            print(f"Missing FROM clause in query: {query}")
            return False
            
        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            print(f"Unbalanced parentheses in query: {query}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error validating SQL query: {str(e)}")
        return False

def serialize_row(row: Tuple) -> List[Any]:
    """Serialize a database row, handling different data types including BLOBs."""
    serialized = []
    for item in row:
        if isinstance(item, bytes):
            # Assuming BLOB data is base64 encoded image or similar
            # You might need different logic depending on the actual BLOB content
            try:
                # Try decoding if it's text, otherwise encode to base64 string
                serialized.append(item.decode('utf-8')) 
            except UnicodeDecodeError:
                serialized.append(base64.b64encode(item).decode('utf-8'))
        elif item is None:
            serialized.append(None) # Or use a placeholder like 'NULL'
        else:
            # Convert other types to string or keep as is if JSON serializable
            serialized.append(item) 
    return serialized 