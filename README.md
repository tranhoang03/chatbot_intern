# RAG Chatbot with Streamlit

This is a Retrieval-Augmented Generation (RAG) chatbot system deployed on Streamlit Cloud.

## Features

- Semantic search using FAISS
- Context-aware responses using LangChain
- Google Generative AI integration
- SQLite database for document storage
- Beautiful Streamlit UI

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   ```
5. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Deployment to Streamlit Cloud

1. Create a GitHub repository and push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin your_github_repo_url
   git push -u origin main
   ```

2. Set up Streamlit Cloud:
   - Go to [Streamlit Cloud](https://share.streamlit.io/)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the main file path to `streamlit_app.py`

3. Configure Secrets:
   - In Streamlit Cloud, go to your app's settings
   - Click on "Secrets"
   - Add your environment variables:
     ```toml
     GOOGLE_API_KEY = "your_google_api_key"
     ```

4. Deploy your app!

## Project Structure

```
enhanced_rag/
├── streamlit_app.py    # Main Streamlit application
├── main.py            # RAG system implementation
├── config.py          # Configuration settings
├── utils.py           # Utility functions
├── models/           # Model files
├── vector_store/     # Vector store files
├── requirements.txt  # Project dependencies
└── .env             # Environment variables (not in git)
```

## Security Notes

- Never commit your `.env` file or any files containing API keys
- Use Streamlit Cloud's secrets management for sensitive data
- Keep your API keys secure and rotate them regularly

## Contributing

Feel free to submit issues and enhancement requests! 