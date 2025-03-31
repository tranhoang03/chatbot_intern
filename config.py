import os
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration for RAG system"""
    # Base directory
    base_dir: Path = Path(__file__).parent
    
    # Database configuration
    db_path: str = str(base_dir / "Database.db")
    db_timeout: int = 30
    
    # Vector store configuration
    vector_store_path: str = str(base_dir / "vector_store")
    top_k_results: int = 5
    
    # Model configuration
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "gemini-1.5-pro"
    llm_temperature: float = 0.7
    
    # API Keys
    google_api_key: str = os.getenv("GOOGLE_API_KEY")
    
    def __post_init__(self):
        """Ensure paths exist"""
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        if not os.path.exists(self.db_path):
            raise ValueError(f"Database file not found at {self.db_path}")
            
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables") 