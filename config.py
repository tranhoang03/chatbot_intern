import os
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
# import streamlit as st # Removed import as st.secrets is no longer used

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration for RAG system"""
    # Base directory
    base_dir: Path = Path(__file__).parent
    
    # Database configuration
    db_path: str = os.getenv("DB_PATH", "Database.db")
    db_timeout: int = int(os.getenv("DB_TIMEOUT", 30))
    
    # Vector store configuration
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "vector_store")
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", 3))
    
    # Model configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-1.5-flash-latest")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
    
    # API Keys - Read directly from environment variables (loaded from .env)
    google_api_key: str = os.getenv("GOOGLE_API_KEY")
    huggingface_hub_token: str = os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    def __post_init__(self):
        """Ensure paths exist and essential keys are loaded"""
        # Ensure vector store directory exists, handle potential base_dir issue
        vector_store_full_path = os.path.join(self.base_dir, self.vector_store_path)
        os.makedirs(vector_store_full_path, exist_ok=True)
        self.vector_store_path = vector_store_full_path # Update path to be absolute

        # Ensure db path is absolute
        db_full_path = os.path.join(self.base_dir, self.db_path)
        if not os.path.exists(db_full_path):
            raise ValueError(f"Database file not found at {db_full_path}")
        self.db_path = db_full_path # Update path to be absolute
            
        # --- Removed dynamic loading logic for keys --- 

        # --- Check if keys were loaded from environment --- 
        if not self.google_api_key:
             print("ERROR: GOOGLE_API_KEY not found in environment variables (check .env file). Application might fail.")
             # Optionally raise error to stop execution if critical
             # raise ValueError("GOOGLE_API_KEY is required but not found.")
        else:
             print("Loaded GOOGLE_API_KEY from environment.")
             
        if not self.huggingface_hub_token:
             # This might be acceptable if models are cached, but downloads will fail/be slow
             print("WARNING: HUGGINGFACE_HUB_TOKEN not found in environment variables (check .env file). Model downloads might be rate-limited or fail if not cached.")
        else:
             print("Loaded HUGGINGFACE_HUB_TOKEN from environment.")
             # No need to explicitly set os.environ here, load_dotenv handles it.

