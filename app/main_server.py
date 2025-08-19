"""
LightRAG API Server with Gemini Integration
Using local LightRAG source code
"""

# Setup LightRAG path first
from setup_lightrag import setup_lightrag_path, verify_lightrag_import
setup_lightrag_path()
verify_lightrag_import()

import os
import asyncio
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# Now import from local LightRAG
from lightrag.api.lightrag_server import create_app
from lightrag.api.config import global_args
from lightrag.utils import setup_logger

# Load environment variables
load_dotenv()

def setup_environment():
    """Setup environment for Gemini integration"""
    
    # Set default environment variables for Gemini
    env_defaults = {
        "LLM_BINDING": "gemini",
        "LLM_MODEL": "gemini-2.0-flash",
        "EMBEDDING_BINDING": "gemini", 
        "EMBEDDING_MODEL": "text-embedding-004",
        "EMBEDDING_DIM": "768",
        "ENABLE_RERANK": "true",
        "RERANK_BINDING": "gemini",
        "TEMPERATURE": "0.0",
        "TOP_K": "40",
        "CHUNK_TOP_K": "10",
        "MAX_TOTAL_TOKENS": "30000",
        "ENABLE_LLM_CACHE": "true",
        "HOST": "0.0.0.0",
        "PORT": "9621",
        "WEBUI_TITLE": "LightRAG Gemini RAG System",
        "WEBUI_DESCRIPTION": "Advanced RAG with Gemini Integration",
        "WORKING_DIR": "./storage/rag_storage",
        "INPUT_DIR": "./storage/documents"
    }
    
    for key, default_value in env_defaults.items():
        if not os.getenv(key):
            os.environ[key] = default_value
            print(f"Set default {key}={default_value}")

def main():
    """Main server entry point"""
    
    print("🚀 Starting LightRAG Gemini Server...")
    
    # Setup environment
    setup_environment()
    
    # Verify required API keys
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ ERROR: GEMINI_API_KEY environment variable is required!")
        print("Please set your Gemini API key in .env file:")
        print("GEMINI_API_KEY=your_gemini_api_key_here")
        return
    
    # Setup logging
    setup_logger("lightrag_server", level="INFO")
    
    # Create storage directories
    storage_dir = Path("storage")
    (storage_dir / "rag_storage").mkdir(parents=True, exist_ok=True)
    (storage_dir / "documents").mkdir(parents=True, exist_ok=True)
    (storage_dir / "exports").mkdir(parents=True, exist_ok=True)
    
    # Parse arguments (using LightRAG's existing argument parser)
    args = global_args()
    
    # Override with our Gemini settings
    args.llm_binding = "gemini"
    args.llm_model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    args.embedding_binding = "gemini"  
    args.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    
    # Create FastAPI app using LightRAG's existing server
    app = create_app(args)
    
    print(f"✅ Server configured with:")
    print(f"   - LLM Model: {args.llm_model}")
    print(f"   - Embedding Model: {args.embedding_model}")
    print(f"   - Host: {os.getenv('HOST', '0.0.0.0')}")
    print(f"   - Port: {os.getenv('PORT', '9621')}")
    print(f"   - Working Dir: {os.getenv('WORKING_DIR')}")
    
    # Start server
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 9621)),
        log_level="info"
    )

if __name__ == "__main__":
    main() 