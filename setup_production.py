#!/usr/bin/env python3
"""
Production LightRAG Setup Script

This script helps users quickly set up the production RAG pipeline with:
- Environment configuration
- Directory creation
- Dependency checking
- Initial testing
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🚀 Production LightRAG Pipeline Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check Python version compatibility"""
    print("📋 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def check_dependencies():
    """Check and install required dependencies"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "google-genai",
        "sentence-transformers", 
        "sentencepiece",
        "requests",
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "numpy",
        "asyncio"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📥 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            sys.exit(1)
    else:
        print("✅ All dependencies are available")

def setup_environment():
    """Setup environment configuration"""
    print("\n🔧 Setting up environment...")
    
    # Check if .env exists
    env_file = Path(".env")
    env_production = Path(".env.production")
    
    if not env_file.exists():
        if env_production.exists():
            print("📋 Copying .env.production to .env...")
            shutil.copy(env_production, env_file)
            print("✅ Environment file created")
        else:
            print("❌ .env.production file not found")
            print("Please create .env file manually with required configuration")
            return False
    else:
        print("✅ .env file already exists")
    
    # Load and validate environment
    load_dotenv()
    
    required_vars = ["GEMINI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing required environment variables: {', '.join(missing_vars)}")
        print("Please update your .env file with the required API keys")
        return False
    
    print("✅ Environment configuration validated")
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "rag_storage",
        "inputs", 
        "logs"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {directory}/")
        else:
            print(f"✅ {directory}/ already exists")

def test_imports():
    """Test if all imports work correctly"""
    print("\n🧪 Testing imports...")
    
    try:
        # Test LightRAG imports
        from lightrag import LightRAG, QueryParam
        print("✅ LightRAG core imports")
        
        # Test production pipeline imports
        from production_rag_pipeline import ProductionRAGPipeline, RAGConfig
        print("✅ Production pipeline imports")
        
        # Test other dependencies
        import google.genai
        print("✅ Google GenAI imports")
        
        import sentence_transformers
        print("✅ Sentence Transformers imports")
        
        import sentencepiece
        print("✅ SentencePiece imports")
        
        import requests
        print("✅ Requests imports")
        
        import fastapi
        print("✅ FastAPI imports")
        
        # Test LLM reranking
        from lightrag.llm_rerank import LLMReranker
        print("✅ LLM reranking imports")
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    print("✅ All imports successful")
    return True

def create_sample_documents():
    """Create sample documents for testing"""
    print("\n📄 Creating sample documents...")
    
    sample_docs = [
        {
            "name": "lightrag_introduction.txt",
            "content": """LightRAG is a powerful retrieval-augmented generation system that combines knowledge graphs with vector search for enhanced information retrieval and generation.

Key Features:
- Knowledge Graph Integration: Builds and utilizes knowledge graphs for structured information
- Vector Search: Efficient similarity search using embeddings
- Multiple Query Modes: Supports naive, local, global, hybrid, mix, and bypass modes
- Advanced Caching: Multi-mode caching for improved performance
- Reranker Support: Integration with reranking models for better relevance
- Token Tracking: Comprehensive token usage monitoring
- Data Isolation: Workspace-based isolation between instances

The system is designed for production use with robust error handling, comprehensive logging, and scalable architecture."""
        },
        {
            "name": "rag_technology.txt", 
            "content": """Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation to create more accurate and contextually relevant responses.

RAG Process:
1. Query Processing: User query is analyzed and processed
2. Information Retrieval: Relevant documents are retrieved from knowledge base
3. Context Assembly: Retrieved information is assembled into context
4. Response Generation: LLM generates response using retrieved context
5. Quality Enhancement: Optional reranking improves response relevance

Benefits of RAG:
- Improved Accuracy: Responses are grounded in retrieved information
- Reduced Hallucination: Less likely to generate false information
- Up-to-date Information: Can access current knowledge base
- Transparency: Sources can be traced and verified
- Cost Efficiency: Reduces need for large context windows"""
        }
    ]
    
    inputs_dir = Path("inputs")
    inputs_dir.mkdir(exist_ok=True)
    
    for doc in sample_docs:
        doc_path = inputs_dir / doc["name"]
        if not doc_path.exists():
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(doc["content"])
            print(f"✅ Created {doc['name']}")
        else:
            print(f"✅ {doc['name']} already exists")

def run_quick_test():
    """Run a quick test of the pipeline"""
    print("\n🧪 Running quick test...")
    
    try:
        # Import and test basic functionality
        from production_rag_pipeline import ProductionRAGPipeline, RAGConfig
        
        print("✅ Production pipeline imports successful")
        print("✅ Setup completed successfully!")
        print("\n🎉 Your production LightRAG pipeline is ready!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        print("Please check your configuration and try again")
        return False
    
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n📋 Next Steps:")
    print("1. Update your .env file with your API keys:")
    print("   - GEMINI_API_KEY: Your Google Gemini API key")
    print()
    print("2. Start the production pipeline:")
    print("   python production_rag_pipeline.py")
    print()
    print("3. Or start the API server:")
    print("   python production_api_server.py")
    print()
    print("4. Access the WebUI at: http://localhost:9621")
    print()
    print("5. Check the documentation: PRODUCTION_README.md")
    print()

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Check and install dependencies
    check_dependencies()
    
    # Setup environment
    env_ok = setup_environment()
    
    # Create directories
    create_directories()
    
    # Test imports
    imports_ok = test_imports()
    
    # Create sample documents
    create_sample_documents()
    
    # Run quick test
    test_ok = run_quick_test()
    
    # Print next steps
    print_next_steps()
    
    if not env_ok:
        print("⚠️  Please configure your API keys in the .env file")
    
    if not imports_ok or not test_ok:
        print("❌ Setup completed with errors. Please check the configuration.")
        sys.exit(1)
    
    print("✅ Setup completed successfully!")

if __name__ == "__main__":
    main() 