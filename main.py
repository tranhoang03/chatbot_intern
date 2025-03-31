from config import Config
from models.rag_system import OptimizedRAGSystem


def main():
    # Initialize configuration
    config = Config()
    
    # Create RAG system
    rag = OptimizedRAGSystem(config)
    
    # Interactive loop
    print("Enhanced RAG System Ready!")
    print("Enter your questions (type 'quit' to exit)")
    print("Type 'history' to view chat history")
    print("Type 'clear' to clear chat history")
    print("="*50)
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() == 'quit':
            break
        elif query.lower() == 'history':
            history = rag.chat_history.get_history()
            if history:
                print("\nChat History:")
                for entry in history:
                    print(f"\nTime: {entry['timestamp']}")
                    print(f"Q: {entry['query']}")
                    print(f"A: {entry['response']}")
            else:
                print("\nNo chat history available.")
            continue
        elif query.lower() == 'clear':
            rag.chat_history.clear_history()
            print("\nChat history cleared.")
            continue
            
        print("\nProcessing...")
        response = rag.answer_query(query)
        print("\nAnswer:", response)
        print("\n" + "="*50)

if __name__ == "__main__":
    main() 
    