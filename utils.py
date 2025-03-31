import json
import sqlite3
from typing import List, Dict, Any, Tuple
import base64

def load_table_data(db_path: str) -> List[Dict[str, Any]]:
    """Load data from all tables in the database"""
    documents = []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print("\n=== Database Tables Information ===")
        print(f"Found tables: {[table[0] for table in tables]}")
        print(f"Total number of tables: {len(tables)}")
        
        for table in tables:
            table_name = table[0]
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            print(f"\nTable: {table_name}")
            print(f"Columns: {column_names}")
            
            # Get all data from the table
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            print(f"Number of rows: {len(rows)}")
            
            # Convert each row to a document
            for row in rows:
                # Create a dictionary of column names and values
                row_dict = {}
                for col_name, val in zip(column_names, row):
                    # Skip specific columns for customers table
                    if table_name == "customers" and col_name in ["embedding", "picture"]:
                        continue
                    row_dict[col_name] = val
                
                # Create content string with all information
                content = f"Bảng {table_name}: " + ", ".join([
                    f"{k}: {v}" for k, v in row_dict.items()
                ])
                
                # Create metadata
                metadata = {
                    "table": table_name,
                    "columns": list(row_dict.keys()),
                    "data": row_dict
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

def serialize_row(row: Tuple) -> List[Any]:
    """Convert row data to JSON-serializable format"""
    return list(row)  # Since we don't have binary data, we can return as is

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