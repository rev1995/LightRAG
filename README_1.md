# LightRAG Gemini 2.0 Flash Production Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Gemini 2.0](https://img.shields.io/badge/Gemini-2.0%20Flash-orange.svg)](https://ai.google.dev/)

A comprehensive, production-ready Knowledge Graph RAG (Retrieval-Augmented Generation) system powered by **Gemini 2.0 Flash**, featuring advanced document processing, intelligent querying, and robust monitoring capabilities.

## 🌟 Features

### 🤖 **Advanced AI Integration**
- **Gemini 2.0 Flash LLM** with token tracking and cost monitoring
- **Native Gemini embeddings** (text-embedding-001, 3072D) with intelligent caching
- **Gemma tokenizer** for accurate token counting
- **LLM-based reranking** using Gemini for intelligent document reordering

### 📚 **Document Processing**
- **Multi-format support**: PDF, DOCX, PPTX, XLSX, TXT, MD, JSON, CSV, and more
- **Advanced parsing** with Docling integration
- **Batch processing** with concurrent document handling
- **Automatic content extraction** and chunking

### 🧠 **Knowledge Graph RAG**
- **Multi-mode querying**: Local, Global, Hybrid, Mix, and Naive modes
- **Entity-relation extraction** with LLM-powered graph building
- **Vector similarity search** with multiple storage backends
- **Context-aware retrieval** with configurable parameters

### 🚀 **Production Features**
- **REST API** with comprehensive endpoints
- **Streamlit interface** with chat, knowledge graph visualization, and dashboards
- **Health monitoring** and metrics collection
- **Configurable storage** backends (JSON, Redis, Neo4j, PostgreSQL, etc.)
- **Production-ready deployment** configuration

### 📊 **Monitoring & Analytics**
- **Real-time metrics** and performance tracking
- **Token usage monitoring** and cost estimation
- **System health checks** with component status
- **Comprehensive logging** with structured output

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd lightrag-gemini-pipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the environment template and configure your settings:

```bash
cp .env.example .env
```

**Essential Configuration** (edit `.env`):

```bash
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-2.0-flash
EMBEDDING_MODEL=text-embedding-001

# Server Configuration
HOST=0.0.0.0
PORT=9621
WORKSPACE=gemini_production

# LLM-based Reranking (recommended for production)
ENABLE_RERANK=true
RERANK_MODE=llm
```

### 3. Start the Server

```bash
# Quick start with validation
python start_server.py

# Development mode with auto-reload
python start_server.py --dev

# Install dependencies automatically
python start_server.py --install
```

### 4. Access the System

```bash
# Start the Streamlit interface
streamlit run streamlit_app.py
```

- **Streamlit Interface**: http://localhost:8501
- **API Documentation**: http://localhost:9621/docs  
- **Health Check**: http://localhost:9621/health

## 📖 Detailed Setup

### Prerequisites

- **Python 3.8+**
- **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **8GB+ RAM** (recommended for production)
- **Storage space** for documents and embeddings

### Environment Variables

The system uses a comprehensive `.env` file for configuration. Key categories:

#### 🔑 **API Configuration**
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

#### 🤖 **LLM Configuration**
```bash
LLM_MODEL=gemini-2.0-flash
TEMPERATURE=0.1
TOP_K_SAMPLING=40
MAX_TOKENS=8000
TIMEOUT=300
```

#### 🔍 **Embedding Configuration**
```bash
EMBEDDING_MODEL=text-embedding-001
EMBEDDING_DIM=3072
EMBEDDING_BATCH_SIZE=32
EMBEDDING_CACHE_ENABLED=true
```

#### 📊 **RAG Parameters**
```bash
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=200
TOP_K=60
CHUNK_TOP_K=15
MAX_ENTITY_TOKENS=12000
MAX_RELATION_TOKENS=12000
```

#### 🏪 **Storage Configuration**
```bash
# Default storage (suitable for development)
LIGHTRAG_KV_STORAGE=JsonKVStorage
LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage
LIGHTRAG_GRAPH_STORAGE=NetworkXStorage

# Production storage options
# LIGHTRAG_KV_STORAGE=RedisKVStorage
# LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage
# LIGHTRAG_GRAPH_STORAGE=Neo4JStorage
```

### Advanced Configuration

#### 📈 **Performance Tuning**
```bash
MAX_ASYNC=6
MAX_PARALLEL_INSERT=3
EMBEDDING_FUNC_MAX_ASYNC=8
ENABLE_LLM_CACHE=true
```

## 🛠 Usage

### Streamlit Interface

The Streamlit interface provides a comprehensive way to interact with the system:

1. **💬 Chat Interface**
   - Natural language conversations
   - Multiple query modes (Mix, Local, Global, Hybrid, Naive)
   - Real-time streaming responses
   - Chat history management

2. **🕸️ Knowledge Graph Visualizer**
   - Interactive graph exploration
   - Entity relationship visualization
   - Configurable depth and node limits
   - Network analysis metrics

3. **📋 Document Dashboard**
   - Upload files via drag-and-drop
   - Add text content directly
   - Monitor processing status
   - Document management actions

4. **📈 System Monitoring**
   - Real-time performance metrics
   - API call tracking and analytics
   - Token usage monitoring
   - System health dashboard

### API Usage

#### Document Upload
```bash
curl -X POST "http://localhost:9621/documents/upload" \
     -F "file=@document.pdf"
```

#### Text Addition
```bash
curl -X POST "http://localhost:9621/documents/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your text content here"}'
```

#### Query System
```bash
curl -X POST "http://localhost:9621/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the main topic?", "mode": "hybrid"}'
```

#### Streaming Query
```bash
curl -X POST "http://localhost:9621/query/stream" \
     -H "Content-Type: application/json" \
     -d '{"query": "Explain the concept", "mode": "mix"}'
```

### Python SDK Usage

```python
from lightrag import LightRAG, QueryParam
from gemini_llm import gemini_model_complete
from gemini_embeddings import gemini_embed
from gemma_tokenizer import get_gemma_tokenizer

# Initialize RAG system
rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=gemini_model_complete,
    embedding_func=gemini_embed,
    tokenizer=get_gemma_tokenizer(),
)

# Add documents
await rag.ainsert("Your document content here")

# Query the system
response = await rag.aquery(
    "What are the main topics?",
    param=QueryParam(mode="hybrid", top_k=10)
)
```

## 🏗 Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │   FastAPI       │    │   LightRAG      │
│   (Frontend)    │────│   (API Server)  │────│   (Core Engine) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Gemini 2.0    │    │   Storage       │
                       │   (LLM+Embed)   │    │   Backends      │
                       └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Document Ingestion**: Multi-format documents → Text extraction → Chunking
2. **Knowledge Extraction**: Chunks → Entity/Relation extraction → Knowledge Graph
3. **Vector Storage**: Text chunks → Embeddings → Vector database
4. **Query Processing**: User query → Vector search + Graph traversal → LLM synthesis
5. **Response Generation**: Context + Query → Gemini 2.0 → Final response

### Query Modes

- **Mix Mode**: Combines multiple retrieval strategies (recommended)
- **Local Mode**: Entity-focused retrieval
- **Global Mode**: Community-based retrieval
- **Hybrid Mode**: Combines local and global
- **Naive Mode**: Simple vector similarity

## 🚦 Deployment

### Development Deployment

```bash
# Start with auto-reload
python start_server.py --dev

# With dependency installation
python start_server.py --install --dev
```

### Production Deployment

#### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 9621

CMD ["python", "lightrag_gemini_server.py"]
```

```bash
# Build and run
docker build -t lightrag-gemini .
docker run -p 9621:9621 --env-file .env lightrag-gemini
```

#### Systemd Service

```ini
[Unit]
Description=LightRAG Gemini Server
After=network.target

[Service]
Type=simple
User=lightrag
WorkingDirectory=/opt/lightrag-gemini
Environment=PATH=/opt/lightrag-gemini/venv/bin
ExecStart=/opt/lightrag-gemini/venv/bin/python lightrag_gemini_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

#### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:9621;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Cloud Deployment

#### Google Cloud Run

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: lightrag-gemini
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containers:
      - image: gcr.io/PROJECT-ID/lightrag-gemini
        ports:
        - containerPort: 9621
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: gemini-secrets
              key: api-key
```

#### AWS ECS

```json
{
  "family": "lightrag-gemini",
  "containerDefinitions": [
    {
      "name": "lightrag-gemini",
      "image": "your-account.dkr.ecr.region.amazonaws.com/lightrag-gemini:latest",
      "portMappings": [
        {
          "containerPort": 9621,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "GEMINI_API_KEY",
          "value": "your-api-key"
        }
      ]
    }
  ]
}
```

## 📊 Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:9621/health

# Detailed metrics
curl http://localhost:9621/metrics
```

### Metrics Collection

The system provides comprehensive metrics:

- **Request metrics**: Total requests, response times
- **Token usage**: Input/output tokens, costs
- **Document metrics**: Total documents, processing status
- **System metrics**: Memory usage, uptime

### Logging

Structured logging with multiple levels:

```bash
# Set log level
LOG_LEVEL=DEBUG

# Configure log directory
LOG_DIR=./logs

# Log rotation
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Reduce batch sizes
EMBEDDING_BATCH_SIZE=32
MAX_ASYNC=2
MAX_PARALLEL_INSERT=1
```

#### 2. Port Conflicts
```bash
# Change port
PORT=9622

# Check port usage
netstat -tulpn | grep 9621
```

#### 3. Tokenizer Download Issues
```bash
# Clear cache and retry
rm -rf ./tokenizer_cache
python start_server.py --validate
```

### Debug Mode

```bash
# Enable debug logging
DEBUG_MODE=true

# Start with validation
python start_server.py --validate

# Run tests
python start_server.py --test
```

### Performance Optimization

1. **Enable caching**:
   ```bash
   ENABLE_LLM_CACHE=true
   EMBEDDING_CACHE_ENABLED=true
   ```

2. **Optimize batch sizes**:
   ```bash
   EMBEDDING_BATCH_SIZE=64
   MAX_ASYNC=6
   ```

3. **Use production storage**:
   ```bash
   LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage
   LIGHTRAG_KV_STORAGE=RedisKVStorage
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python start_server.py --test

# Format code
black .
isort .

# Lint code
flake8 .
mypy .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LightRAG](https://github.com/HKUDS/LightRAG) - Core RAG framework
- [Google Gemini](https://ai.google.dev/) - LLM and embedding models
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [Gemma](https://github.com/google/gemma_pytorch) - Tokenizer models

## 📞 Support

For support and questions:

1. Check the [troubleshooting section](#🔧-troubleshooting)
2. Review the [API documentation](http://localhost:9621/docs)
3. Open an issue on GitHub
4. Check the health endpoint: `/health`

---

**Built with ❤️ for production RAG systems** 