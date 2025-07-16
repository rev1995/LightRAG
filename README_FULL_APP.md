# LightRAG Full End-to-End Application

A complete RAG (Retrieval-Augmented Generation) application with API backend and web frontend, built with LightRAG and Gemini LLM.

## 🚀 Features

- **Production RAG Pipeline** with Gemini LLM integration
- **FastAPI Backend** with comprehensive REST API
- **React Frontend** with modern UI components
- **Token Tracking** and cost estimation
- **Advanced Caching** with multiple modes
- **LLM-based Reranking** for improved retrieval
- **Document Upload** and management
- **Real-time Health Monitoring**
- **Docker Support** for easy deployment

## 📁 Project Structure

```
LightRAG/
├── api_server.py              # FastAPI backend server
├── production_rag_pipeline.py # Production RAG pipeline
├── llm_rerank_robust.py      # Robust LLM reranking
├── start_backend.py          # Backend startup script
├── start_frontend.py         # Frontend startup script
├── requirements_api.txt      # Backend dependencies
├── .env.backend             # Backend environment config
├── lightrag_webui/          # React frontend
│   ├── src/
│   │   ├── api/lightrag.ts  # API client functions
│   │   ├── components/      # React components
│   │   └── ...
│   ├── .env.local          # Frontend environment
│   └── package.json
├── docker-compose.full.yml  # Docker Compose setup
├── Dockerfile.backend       # Backend Dockerfile
└── lightrag_webui/Dockerfile.frontend # Frontend Dockerfile
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Google Gemini API Key

### 1. Backend Setup

```bash
# Install backend dependencies
pip install -r requirements_api.txt

# Set up environment
cp .env.backend .env.backend.local
# Edit .env.backend.local with your GEMINI_API_KEY

# Start backend
python start_backend.py
```

Backend will be available at: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd lightrag_webui

# Install dependencies
npm install

# Start frontend
npm run dev
```

Frontend will be available at: http://localhost:5173

### 3. Using Docker (Alternative)

```bash
# Set environment variable
export GEMINI_API_KEY=your_api_key_here

# Start with Docker Compose
docker-compose -f docker-compose.full.yml up --build
```

## 🔧 Configuration

### Backend Environment (.env.backend)

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Working Directory
WORKING_DIR=./rag_storage
WORKSPACE=default

# LLM Configuration
LLM_MODEL=gemini-2.0-flash
LLM_MAX_OUTPUT_TOKENS=5000
LLM_TEMPERATURE=0.1

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIM=768

# Reranker Configuration
ENABLE_RERANK=true
RERANK_STRATEGY=semantic_scoring

# Cache Configuration
ENABLE_LLM_CACHE=true
ENABLE_EMBEDDING_CACHE=true
```

### Frontend Environment (lightrag_webui/.env.local)

```env
VITE_API_URL=http://localhost:8000
VITE_API_PROXY=http://localhost:8000
VITE_API_ENDPOINTS=http://localhost:8000
VITE_BACKEND_URL=http://localhost:8000
```

## 📡 API Endpoints

### Core Endpoints

- `POST /query` - Query the RAG system
- `POST /insert` - Insert documents
- `POST /clear_cache` - Clear cache
- `GET /token_stats` - Get token usage statistics
- `GET /health` - Health check

### File Upload

- `POST /upload` - Upload single document
- `POST /upload_batch` - Upload multiple documents

### Example API Usage

```bash
# Query the system
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is LightRAG?",
    "mode": "mix",
    "user_prompt": "Provide a comprehensive explanation"
  }'

# Insert documents
curl -X POST "http://localhost:8000/insert" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "LightRAG is a powerful RAG system that combines knowledge graphs with vector search."
    ]
  }'

# Get token statistics
curl "http://localhost:8000/token_stats"
```

## 🎯 Frontend Features

### Document Management
- Upload documents (text files, PDFs, etc.)
- View document status and processing
- Clear documents and cache

### Query Interface
- Multiple query modes (naive, local, global, hybrid, mix, bypass)
- Custom user prompts
- Conversation history
- Real-time streaming responses

### Knowledge Graph Visualization
- Interactive graph viewer
- Entity and relationship editing
- Graph search and filtering

### Monitoring
- Token usage tracking
- Cost estimation
- Health status monitoring
- Cache management

## 🔍 Usage Examples

### 1. Basic Query

```javascript
import { queryRAG } from '@/api/lightrag'

const result = await queryRAG({
  query: "What is the main advantage of LightRAG?",
  mode: "mix",
  user_prompt: "Focus on technical benefits"
})

console.log(result.response)
```

### 2. Document Insertion

```javascript
import { insertDocuments } from '@/api/lightrag'

const result = await insertDocuments([
  "LightRAG combines knowledge graphs with vector search for improved retrieval.",
  "The system supports multiple query modes and advanced caching."
])

console.log(`Inserted ${result.successful_insertions}/${result.total_documents} documents`)
```

### 3. Cache Management

```javascript
import { clearRAGCache, getTokenStats } from '@/api/lightrag'

// Clear specific cache modes
await clearRAGCache(['query', 'entity_extract'])

// Get token usage
const stats = await getTokenStats()
console.log(`Total tokens used: ${stats.total_tokens}`)
```

## 🐛 Troubleshooting

### Common Issues

1. **Backend won't start**
   - Check `GEMINI_API_KEY` is set
   - Verify Python dependencies are installed
   - Check logs in `./logs/` directory

2. **Frontend can't connect to backend**
   - Verify backend is running on port 8000
   - Check `.env.local` has correct API URL
   - Ensure CORS is properly configured

3. **Token usage errors**
   - Check Gemini API key permissions
   - Verify API quotas and limits
   - Monitor token usage in logs

4. **Document processing fails**
   - Check file format support
   - Verify working directory permissions
   - Monitor pipeline status

### Debug Mode

Enable verbose logging:

```bash
# Backend
VERBOSE=true python start_backend.py

# Frontend
npm run dev -- --debug
```

## 🚀 Production Deployment

### Environment Variables

Set production environment variables:

```bash
# Backend
export GEMINI_API_KEY=your_production_key
export LOG_LEVEL=WARNING
export WORKING_DIR=/data/rag_storage

# Frontend
export VITE_API_URL=https://your-api-domain.com
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.full.yml up -d

# View logs
docker-compose -f docker-compose.full.yml logs -f
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        proxy_pass http://localhost:5173;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Monitoring

### Health Checks

- Backend: `GET /health`
- Frontend: Built-in health monitoring
- Docker: Automatic health checks

### Logs

- Backend logs: `./logs/production_rag.log`
- Frontend logs: Browser console
- Docker logs: `docker-compose logs`

### Metrics

- Token usage statistics
- Response times
- Error rates
- Cache hit rates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LightRAG team for the core RAG framework
- Google for Gemini LLM
- FastAPI for the web framework
- React team for the frontend framework 