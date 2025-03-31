from typing import List, Dict, Any
import os
import json
from datetime import datetime

class ChatHistory:
    def __init__(self, history_file: str = "chat_history.json", max_history: int = 5):
        self.history_file = history_file
        self.max_history = max_history
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load chat history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading chat history: {e}")
                return []
        return []
    
    def _save_history(self):
        """Save chat history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def add_chat(self, query: str, response: str):
        """Add a new chat entry"""
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response
        }
        self.history.append(chat_entry)
        
        # Keep only the last max_history entries
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            
        self._save_history()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get all chat history"""
        return self.history
    
    def get_recent_history(self) -> str:
        """Get recent chat history as formatted string"""
        if not self.history:
            return ""
            
        history_text = "Lịch sử trò chuyện gần đây:\n"
        for entry in self.history:
            history_text += f"Q: {entry['query']}\n"
            history_text += f"A: {entry['response']}\n"
        return history_text
    
    def clear_history(self):
        """Clear all chat history"""
        self.history = []
        self._save_history() 