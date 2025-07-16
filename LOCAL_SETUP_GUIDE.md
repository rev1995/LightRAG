# Local Setup Guide - LightRAG Production Pipeline

This guide shows how to run the LightRAG production pipeline locally without Docker, using the source code directly.

## 🏗️ Architecture Overview

The system consists of three main components:
1. **LightRAG Core** (`lightrag/`) - Python backend with RAG logic
2. **API Server** (`lightrag/api/`) - FastAPI server for backend API
3. **WebUI** (`lightrag_webui/`) - React/TypeScript frontend

## 📋 Prerequisites

### System Requirements
- **Python 3.9+** (recommended: 3.11+)
- **Node.js 18+** (for WebUI)
- **Git** (for cloning)

### Required API Keys
- **GEMINI_API_KEY**: Google Gemini API key

## 🚀 Step-by-Step Setup

### Step 1: Clone and Navigate
```bash
# Navigate to your project directory
cd LightRAG

# Verify you're in the right directory
ls -la
# Should see: lightrag/, lightrag_webui/, production_rag_pipeline.py, etc.
```

### Step 2: Python Environment Setup

#### Option A: Using venv (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify activation
which python  # Should point to venv directory
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n lightrag python=3.11
conda activate lightrag
```

### Step 3: Install Python Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install additional dependencies for production
pip install google-genai sentence-transformers sentencepiece requests fastapi uvicorn python-dotenv nest-asyncio

# Optional: Install multimodal support
pip install raganything
```

### Step 4: Environment Configuration

```bash
# Copy production environment file
cp .env.production .env

# Edit the .env file with your API key
# On Windows:
notepad .env
# On macOS/Linux:
nano .env
```

#### Required .env Configuration:
```bash
# Required API Keys
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=9621
WORKING_DIR=./rag_storage
WORKSPACE=production_workspace

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs

# LLM Configuration
LLM_MODEL=gemini-2.0-flash
LLM_MAX_OUTPUT_TOKENS=5000
LLM_TEMPERATURE=0.1
LLM_TOP_K=10

# Embedding Configuration (Gemini Native)
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIM=768
EMBEDDING_MAX_TOKEN_SIZE=512

# Reranker Configuration
ENABLE_RERANK=true
RERANK_STRATEGY=semantic_scoring
RERANK_BATCH_SIZE=5
RERANK_MAX_CONCURRENT=3
RERANK_CACHE_ENABLED=true

# Cache Configuration
ENABLE_LLM_CACHE=true
ENABLE_EMBEDDING_CACHE=true
EMBEDDING_CACHE_SIMILARITY_THRESHOLD=0.90

# Query Configuration
DEFAULT_QUERY_MODE=mix
TOP_K=40
CHUNK_TOP_K=10
MAX_ENTITY_TOKENS=10000
MAX_RELATION_TOKENS=10000
MAX_TOTAL_TOKENS=32000

# Tokenizer Configuration
TOKENIZER_DIR=./tokenizer_cache
```

### Step 5: Create Directories

```bash
# Create necessary directories
mkdir -p rag_storage inputs logs tokenizer_cache

# Verify directories
ls -la
# Should see: rag_storage/, inputs/, logs/, tokenizer_cache/
```

### Step 6: Setup WebUI

```bash
# Navigate to WebUI directory
cd lightrag_webui

# Install Node.js dependencies
npm install
# OR if you prefer yarn:
# yarn install

# Verify installation
npm run build
# Should complete without errors

# Return to main directory
cd ..
```

### Step 7: Test Core Functionality

```bash
# Test Python imports
python -c "
from lightrag import LightRAG, QueryParam
from production_rag_pipeline import ProductionRAGPipeline, RAGConfig
import google.genai
print('✅ All imports successful')
"

# Test environment configuration
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    print('✅ Environment configured')
else:
    print('❌ GEMINI_API_KEY not found')
"
```

## 🚀 Running the System

### Option 1: Production Pipeline (Direct Python)

```bash
# Run the production pipeline demo
python production_rag_pipeline.py
```

This will:
- Initialize the RAG pipeline
- Process sample documents
- Run example queries
- Show token usage statistics

### Option 2: API Server + WebUI (Recommended)

#### Terminal 1: Start API Server
```bash
# Start the LightRAG API server
python -m lightrag.api.lightrag_server --config .env
```

The API server will start on `http://localhost:9621`

#### Terminal 2: Start WebUI (Development Mode)
```bash
# Navigate to WebUI directory
cd lightrag_webui

# Start development server
npm run dev
# OR
yarn dev
```

The WebUI will start on `http://localhost:3000` (or another port if 3000 is busy)

#### Terminal 3: Build WebUI for Production (Optional)
```bash
# Navigate to WebUI directory
cd lightrag_webui

# Build for production
npm run build

# The built files will be in dist/ directory
# You can serve them with any static file server
```

### Option 3: Production API Server

```bash
# Start the production API server
python production_api_server.py
```

## 🔧 Development Workflow

### Running Tests

```bash
# Test the production pipeline
python production_rag_pipeline.py

# Test LLM reranking
python test_llm_rerank.py

# Test tokenizer
python test_tokenizer.py
```

### Adding Documents

```bash
# Create sample documents
echo "LightRAG is a powerful retrieval-augmented generation system." > inputs/sample1.txt
echo "It combines knowledge graphs with vector search for enhanced results." > inputs/sample2.txt

# Or use the setup script
python setup_production.py
```

### Monitoring

```bash
# Check logs
tail -f logs/lightrag.log

# Check storage
ls -la rag_storage/

# Check cache
ls -la rag_storage/cache/
```

## 🌐 Access Points

### API Endpoints
- **Health Check**: `http://localhost:9621/health`
- **Query**: `http://localhost:9621/query`
- **Insert**: `http://localhost:9621/insert`
- **Cache Management**: `http://localhost:9621/cache/clear`
- **Statistics**: `http://localhost:9621/stats/tokens`

### WebUI
- **Main Interface**: `http://localhost:3000` (development)
- **API Documentation**: `http://localhost:9621/docs` (Swagger UI)

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Install missing dependencies
pip install google-genai sentence-transformers sentencepiece
```

#### 2. API Key Issues
```bash
# Check environment file
cat .env | grep GEMINI_API_KEY

# Test API key
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('API Key:', os.getenv('GEMINI_API_KEY')[:10] + '...' if os.getenv('GEMINI_API_KEY') else 'Not found')
"
```

#### 3. Port Conflicts
```bash
# Check if ports are in use
netstat -an | grep 9621
netstat -an | grep 3000

# Kill processes if needed
# Windows:
netstat -ano | findstr 9621
taskkill /PID <PID> /F

# macOS/Linux:
lsof -i :9621
kill -9 <PID>
```

#### 4. Node.js Issues
```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

#### 5. Python Path Issues
```bash
# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run with explicit path
PYTHONPATH=. python production_rag_pipeline.py
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export VERBOSE=true

# Run with debug output
python production_rag_pipeline.py
```

## 📊 Performance Tips

### For Development
```bash
# Use smaller models for faster testing
LLM_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=text-embedding-004

# Reduce token limits
MAX_TOTAL_TOKENS=16000
CHUNK_SIZE=800
```

### For Production
```bash
# Use larger models for better quality
LLM_MODEL=gemini-2.0-flash
EMBEDDING_MODEL=text-embedding-004

# Increase limits for complex queries
MAX_TOTAL_TOKENS=32000
CHUNK_SIZE=1200
```

## 🔄 Update Workflow

### Update Source Code
```bash
# Pull latest changes
git pull origin main

# Reinstall dependencies if needed
pip install -r requirements.txt
cd lightrag_webui && npm install && cd ..
```

### Update Environment
```bash
# Compare with production config
diff .env .env.production

# Update if needed
cp .env.production .env
# Then edit .env with your API keys
```

## 📝 Next Steps

1. **Test the system** with sample documents
2. **Configure your API keys** in the `.env` file
3. **Add your documents** to the `inputs/` directory
4. **Customize the configuration** based on your needs
5. **Monitor performance** and adjust settings
6. **Scale up** with production storage backends if needed

## 🆘 Getting Help

1. **Check logs**: `tail -f logs/lightrag.log`
2. **Test components**: Run individual test scripts
3. **Verify environment**: Check all required variables
4. **Review documentation**: Read `PRODUCTION_README.md`
5. **Check issues**: Look for similar problems in the repository

The system is now ready for local development and testing! 🎉 