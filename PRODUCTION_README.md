# Production LightRAG Pipeline

A production-ready RAG (Retrieval-Augmented Generation) pipeline built with LightRAG, featuring Gemini LLM, efficient chunking, advanced caching, reranking, and comprehensive query controls.

## 🚀 Features

### Core Features
- **Gemini LLM Integration**: Powered by Google's Gemini models with token tracking
- **Efficient Text Chunking**: Uses Gemini tokenizer for optimal document splitting
- **Advanced Caching**: Multi-mode caching with robust implementation
- **Reranker Integration**: BAAI/bge-reranker-v2-m3 with mix mode as default
- **Query Parameter Controls**: Comprehensive query customization in WebUI
- **Data Isolation**: Workspace-based isolation between LightRAG instances
- **Multimodal Processing**: RAG-Anything integration for document processing
- **Token Usage Tracking**: Real-time token consumption monitoring

### WebUI Enhancements
- **User Prompt Control**: Custom prompts to guide LLM response processing
- **Cache Management**: Clear LLM response cache with different modes
- **Query Mode Selection**: Choose from naive, local, global, hybrid, mix, bypass
- **Rerank Toggle**: Enable/disable reranking per query
- **Response Format Control**: Multiple paragraphs, single paragraph, bullet points
- **Token Limits**: Configurable entity, relation, and total token limits

### Production Features
- **Environment Configuration**: All settings via environment variables
- **Comprehensive Logging**: Rotating file logs with configurable levels
- **Error Handling**: Robust error handling and recovery
- **Health Checks**: API health monitoring
- **CORS Support**: Cross-origin request handling
- **SSL Support**: HTTPS configuration
- **Authentication**: API key and JWT token support

## 📋 Requirements

### API Keys Required
- **GEMINI_API_KEY**: Google Gemini API key
- **GEMINI_API_KEY**: Google Gemini API key for LLM and embeddings

### Python Dependencies
```bash
pip install google-genai sentence-transformers sentencepiece requests fastapi uvicorn python-dotenv
```

### Optional Dependencies
```bash
pip install raganything  # For multimodal document processing
```

## 🛠️ Setup

### 1. Environment Configuration

Copy the production environment file:
```bash
cp .env.production .env
```

Edit `.env` and configure your API keys:
```bash
# Required API Keys
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=9621
WORKING_DIR=./rag_storage
WORKSPACE=production_workspace

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs
```

### 2. Directory Structure
```
LightRAG/
├── production_rag_pipeline.py    # Main RAG pipeline
├── production_api_server.py      # API server
├── .env                          # Environment configuration
├── rag_storage/                  # RAG data storage
├── inputs/                       # Input documents
└── logs/                         # Log files
```

### 3. Running the Pipeline

#### Option 1: Direct Python Execution
```bash
# Run the production pipeline demo
python production_rag_pipeline.py

# Run the API server
python production_api_server.py
```

#### Option 2: Using LightRAG API Server
```bash
# Start the LightRAG API server with production config
python -m lightrag.api.lightrag_server --config .env
```

#### Option 3: Docker (if available)
```bash
# Build and run with Docker
docker build -t lightrag-production .
docker run -p 9621:9621 --env-file .env lightrag-production
```

## 🔧 Configuration

### Environment Variables

#### Required
- `GEMINI_API_KEY`: Google Gemini API key
- `GEMINI_API_KEY`: Google Gemini API key for LLM and embeddings

#### Server Configuration
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 9621)
- `WORKING_DIR`: RAG storage directory (default: ./rag_storage)
- `WORKSPACE`: Data isolation workspace (default: production_workspace)

#### LLM Configuration
- `LLM_MODEL`: Gemini model name (default: gemini-1.5-flash)
- `LLM_MAX_OUTPUT_TOKENS`: Maximum output tokens (default: 5000)
- `LLM_TEMPERATURE`: Generation temperature (default: 0.1)

#### Embedding Configuration
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-004)
- `EMBEDDING_DIM`: Embedding dimension (default: 768)
- `EMBEDDING_MAX_TOKEN_SIZE`: Max tokens per embedding (default: 512)

#### Reranker Configuration
- `ENABLE_RERANK`: Enable reranking (default: true)
- `RERANK_STRATEGY`: Reranking strategy (default: semantic_scoring)
- `RERANK_BATCH_SIZE`: Batch size for processing (default: 5)
- `RERANK_MAX_CONCURRENT`: Max concurrent operations (default: 3)
- `RERANK_CACHE_ENABLED`: Enable reranking cache (default: true)
- `RERANK_MODEL`: External reranker model (optional)
- `RERANK_BINDING_HOST`: External reranker API host (optional)
- `RERANK_BINDING_API_KEY`: External reranker API key (optional)

#### Cache Configuration
- `ENABLE_LLM_CACHE`: Enable LLM response caching (default: true)
- `ENABLE_EMBEDDING_CACHE`: Enable embedding caching (default: true)
- `EMBEDDING_CACHE_SIMILARITY_THRESHOLD`: Cache similarity threshold (default: 0.90)

#### Query Configuration
- `DEFAULT_QUERY_MODE`: Default query mode (default: mix)
- `TOP_K`: Number of top results (default: 40)
- `CHUNK_TOP_K`: Number of chunks to retrieve (default: 10)
- `MAX_ENTITY_TOKENS`: Max entity tokens (default: 10000)
- `MAX_RELATION_TOKENS`: Max relation tokens (default: 10000)
- `MAX_TOTAL_TOKENS`: Max total tokens (default: 32000)

#### Tokenizer Configuration
- `TOKENIZER_DIR`: Directory for tokenizer cache (default: ./tokenizer_cache)

## 📖 Usage

### 1. Basic Usage

```python
from production_rag_pipeline import ProductionRAGPipeline, RAGConfig

# Initialize configuration
config = RAGConfig()

# Create and initialize pipeline
pipeline = ProductionRAGPipeline(config)
await pipeline.initialize()

# Insert documents
documents = [
    "LightRAG is a powerful retrieval-augmented generation system.",
    "It combines knowledge graphs with vector search for better results."
]
await pipeline.insert_documents(documents)

# Query with custom parameters
result = await pipeline.query(
    query="What is LightRAG?",
    mode="mix",
    user_prompt="Provide a comprehensive explanation with examples.",
    top_k=10,
    enable_rerank=True
)

print(result["response"])
```

### 2. API Usage

#### Health Check
```bash
curl http://localhost:9621/health
```

#### Query
```bash
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is LightRAG?",
    "mode": "mix",
    "user_prompt": "Provide a comprehensive explanation",
    "top_k": 10,
    "enable_rerank": true
  }'
```

#### Insert Documents
```bash
curl -X POST http://localhost:9621/insert \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "LightRAG is a powerful RAG system.",
      "It uses knowledge graphs and vector search."
    ]
  }'
```

#### Clear Cache
```bash
curl -X POST http://localhost:9621/cache/clear \
  -H "Content-Type: application/json" \
  -d '{
    "modes": ["query", "entity_extract"]
  }'
```

#### Get Token Statistics
```bash
curl http://localhost:9621/stats/tokens
```

### 3. WebUI Usage

1. Start the LightRAG API server
2. Access the WebUI at `http://localhost:9621`
3. Use the enhanced query settings panel to configure:
   - Query mode (naive, local, global, hybrid, mix, bypass)
   - User prompt for response guidance
   - Rerank enable/disable
   - Response format (multiple paragraphs, single paragraph, bullet points)
   - Token limits and other parameters

## 🔍 Query Modes

### Available Modes
- **naive**: Basic search without advanced techniques
- **local**: Context-dependent information retrieval
- **global**: Global knowledge utilization
- **hybrid**: Combines local and global methods
- **mix**: Integrates knowledge graph and vector retrieval (recommended with reranker)
- **bypass**: Bypasses knowledge retrieval, uses LLM directly

## 🎯 LLM-Based Reranking

### Overview
The system uses an efficient LLM-based reranking approach that leverages the same LLM for both generation and reranking, providing:

- **Cost-Effectiveness**: No additional API costs (uses existing LLM)
- **Semantic Consistency**: Same model for generation and reranking
- **Configurable Strategies**: Multiple reranking approaches
- **Built-in Caching**: Performance optimization for repeated queries
- **Fallback Mechanisms**: Reliable error handling

### Reranking Strategies
- **semantic_scoring**: Provides relevance scores (0.0-1.0) with reasoning
- **relevance_ranking**: Ranks documents by relevance position
- **hybrid**: Combines scoring and ranking for comprehensive evaluation

### Configuration
```bash
# LLM-based reranking settings
ENABLE_RERANK=true
RERANK_STRATEGY=semantic_scoring
RERANK_BATCH_SIZE=5
RERANK_MAX_CONCURRENT=3
RERANK_CACHE_ENABLED=true
```

### Usage Example
```python
# Query with LLM-based reranking
result = await pipeline.query(
    query="What is machine learning?",
    enable_rerank=True,  # Uses LLM-based reranking
    top_k=10
)

# The reranked documents include:
# - rerank_score: Relevance score (0.0-1.0)
# - rerank_reasoning: Explanation of relevance
# - rerank_confidence: Confidence in the assessment
```

### User Prompt vs Query
The `user_prompt` parameter guides the LLM on how to process retrieved results after the query is completed. It does not participate in the RAG retrieval phase.

Example:
```python
result = await pipeline.query(
    query="Please draw a character relationship diagram for Scrooge",
    user_prompt="For diagrams, use mermaid format with English/Pinyin node names and Chinese display labels"
)
```

## 🗄️ Cache Management

### Cache Modes
- **query**: LLM responses for queries
- **entity_extract**: Entity extraction results
- **relation_extract**: Relation extraction results
- **summary**: Document and entity summaries

### Clearing Cache
```python
# Clear specific cache modes
await pipeline.clear_cache(["query", "entity_extract"])

# Clear all caches
await pipeline.clear_cache()
```

## 📊 Token Tracking

### Token Statistics
```python
stats = await pipeline.get_token_usage_stats()
print(f"Total tokens: {stats['total_tokens']}")
print(f"Average per request: {stats['average_tokens_per_request']}")
```

### Cost Estimation
The system tracks token usage and provides cost estimation based on configured rates.

## 🔒 Data Isolation

### Workspace Configuration
Use the `WORKSPACE` environment variable to isolate data between different LightRAG instances:

```bash
# Instance 1
WORKSPACE=production_workspace

# Instance 2
WORKSPACE=development_workspace
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure `GEMINI_API_KEY` is set correctly
   - Check API key permissions and quotas

2. **Memory Issues**
   - Clear cache regularly using the cache management endpoints
   - Monitor token usage and adjust limits as needed

3. **Performance Issues**
   - Enable caching for better performance
   - Use appropriate chunk sizes for your documents
   - Consider using production storage backends (Redis, PostgreSQL, etc.)

4. **Logging Issues**
   - Check log directory permissions
   - Verify `LOG_LEVEL` configuration
   - Monitor log file sizes and rotation

### Debug Mode
Enable verbose logging:
```bash
LOG_LEVEL=DEBUG
VERBOSE=true
```

## 📈 Performance Optimization

### Recommendations
1. **Use mix mode** when reranker is enabled for best results
2. **Enable caching** for repeated queries
3. **Monitor token usage** to control costs
4. **Use appropriate chunk sizes** (500-1500 tokens recommended)
5. **Consider production storage** for large-scale deployments

### Storage Options
- **Default**: JSON files (good for development)
- **Production**: Redis, PostgreSQL, Neo4j, Milvus, Qdrant

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the same license as LightRAG.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the LightRAG documentation
3. Open an issue on the repository
4. Check the logs for detailed error information 