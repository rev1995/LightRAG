# LightRAG

A lightweight, production-ready RAG (Retrieval-Augmented Generation) pipeline that integrates Gemini models, token tracking, and advanced caching for optimal performance.

## Features

- Gemini model integration for LLM capabilities
- Customizable tokenization with GemmaTokenizer (no tiktoken dependency)
- Embedding and vector search for knowledge retrieval
- Knowledge graph integration
- Token tracking and usage monitoring
- Advanced caching for improved performance
- Production-ready API server

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

The system can be configured using environment variables or a `.env` file. Key configuration options include:

### Tokenizer Configuration

By default, LightRAG uses the GemmaTokenizer which doesn't require tiktoken:

```
# Enable/disable GemmaTokenizer (default: True)
USE_GEMMA_TOKENIZER=True

# Directory to cache tokenizer models
TOKENIZER_DIR=./tokenizer_cache
```

### LLM Configuration

```
# Gemini API key
GEMINI_API_KEY=your_api_key_here

# Gemini model to use
GEMINI_MODEL=gemini-2.0-flash

# Output parameters
GEMINI_MAX_OUTPUT_TOKENS=5000
GEMINI_TEMPERATURE=0.1
GEMINI_TOP_K=10
```

### Storage Configuration

```
# Working directory for storage
WORKING_DIR=./rag_storage

# Storage types
KV_STORAGE=JsonKVStorage
VECTOR_STORAGE=NanoVectorDBStorage
GRAPH_STORAGE=NetworkXStorage
DOC_STATUS_STORAGE=JsonDocStatusStorage
```

## Usage

### Starting the API Server

```bash
python api_server.py
```

### Using the Client

```python
from lightrag_client import LightRAGClient

client = LightRAGClient("http://localhost:8000")
response = client.query("What is RAG?")
print(response)
```

## Advanced Usage

For advanced usage and customization, refer to the examples in the `lightrag/examples/` directory.

Build frontend and run the rag app:
cd lightrag_webui
bun install  # or npm install
bun run build  # or npm run build
python -m lightrag.api.lightrag_server