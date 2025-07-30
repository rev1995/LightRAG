# Running LightRAG Gemini from the LightRAG Directory

## 📁 Current Directory Structure
You are now inside the LightRAG directory with this structure:
```
LightRAG/                           <- You are here
├── 🔧 Configuration & Scripts
│   ├── .env                        # Environment configuration
│   ├── requirements.txt            # Python dependencies
│   └── start_server.py            # Server startup script
├── 🤖 AI Components
│   ├── gemini_llm.py              # Gemini LLM integration
│   ├── gemini_embeddings.py       # Gemini embeddings
│   ├── gemma_tokenizer.py         # Gemma tokenizer
│   └── llm_reranker.py            # LLM-based reranking
├── 🚀 Application
│   ├── lightrag_gemini_server.py  # FastAPI server
│   └── streamlit_app.py           # Streamlit interface
├── 📚 LightRAG Source
│   └── lightrag/                  # LightRAG framework source
└── 📖 Documentation
    ├── README.md
    └── PIPELINE_DOCUMENTATION.md
```

## 🚀 Quick Start

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Configure Environment
```bash
# Copy example config and edit
cp env.example .env
# Edit .env and add your GEMINI_API_KEY
```

#### 3. Start the API Server
```bash
python lightrag_gemini_server.py
# or
python start_server.py
```

#### 4. Start the Streamlit Interface (in a new terminal)
```bash
streamlit run streamlit_app.py
```

## 🔧 Path Changes Made

Since you moved into the LightRAG directory, the following path updates were made:

### ✅ Fixed Import Paths
- **lightrag_gemini_server.py**: `sys.path.insert(0, './lightrag')` 
- **gemini_llm.py**: Added LightRAG path setup
- **gemini_embeddings.py**: Added LightRAG path setup  
- **gemma_tokenizer.py**: Added LightRAG path setup
- **llm_reranker.py**: Added LightRAG path setup

### ✅ Removed WebUI References
- Removed `./webui` directory checks
- Updated redirects to go to API docs instead

### ✅ Relative Paths Work
These paths work correctly from the LightRAG directory:
- `./inputs` - Input documents directory
- `./rag_storage` - RAG data storage  
- `./logs` - Application logs
- `./tokenizer_cache` - Tokenizer cache
- `./embedding_cache` - Embedding cache

## 🌐 Access Points

After starting the system:

- **🖥️ Streamlit Interface**: http://localhost:8501
  - Chat interface
  - Knowledge graph visualizer
  - Document management
  - System monitoring

- **📚 API Documentation**: http://localhost:9621/docs
  - Complete API reference
  - Interactive testing

- **💚 Health Check**: http://localhost:9621/health
  - System status

## 📝 Configuration

Your `.env` file should have at minimum:
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (with defaults)
HOST=0.0.0.0
PORT=9621
WORKSPACE=gemini_production
WORKING_DIR=./rag_storage
```

## 🔍 Troubleshooting

### Import Errors
If you get import errors related to LightRAG modules:
- Make sure you're running from the LightRAG directory
- The path fixes should handle this automatically

### Port Conflicts
If port 9621 or 8501 are in use:
```bash
# Change in .env file
PORT=9622
STREAMLIT_PORT=8502
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### API Key Issues
- Get your Gemini API key from: https://makersuite.google.com/app/apikey
- Add it to your `.env` file as `GEMINI_API_KEY=your_key_here`

## 🎯 Next Steps

1. **Configure your API key** in the `.env` file
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Start the API server**: `python lightrag_gemini_server.py`
4. **Start the Streamlit interface**: `streamlit run streamlit_app.py`
5. **Upload documents** and start chatting!

The system is now properly configured to run from within the LightRAG directory! 🚀 