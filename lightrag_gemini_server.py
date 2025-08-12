#!/usr/bin/env python3
"""
LightRAG Gemini 2.0 Flash Production Server
Production-ready FastAPI server with comprehensive Gemini integration, monitoring, and all LightRAG features.
"""

import os
import sys
import asyncio
import logging
import uvicorn
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import time
import traceback

# FastAPI imports
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables first
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

# LightRAG imports (built from source) - append to avoid conflicts with stdlib
lightrag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lightrag'))
if lightrag_path not in sys.path:
    sys.path.append(lightrag_path)

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, logger, setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.api.routers.document_routes import create_document_routes, DocumentManager
from lightrag.api.routers.query_routes import create_query_routes
from lightrag.api.routers.graph_routes import create_graph_routes
# Authentication removed for simplicity

# Gemini integration imports
from gemini_llm import (
    GeminiLLM, 
    GeminiConfig,
    gemini_model_complete,
    validate_gemini_config,
    get_gemini_llm
)
from gemini_embeddings import (
    GeminiEmbeddings,
    GeminiEmbeddingConfig, 
    gemini_embed,
    validate_gemini_embedding_config,
    get_gemini_embeddings
)
from gemma_tokenizer import (
    GemmaTokenizer,
    get_gemma_tokenizer,
    validate_tokenizer_setup
)
from llm_reranker import create_llm_rerank_function


class ServerConfig:
    """Server configuration from environment variables"""
    
    def __init__(self):
        # Server settings
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "9621"))
        self.workers = int(os.getenv("WORKERS", "1"))
        self.cors_origins = os.getenv("CORS_ORIGINS", "*")
        
        # Directory settings
        self.input_dir = os.getenv("INPUT_DIR", "./inputs")
        self.working_dir = os.getenv("WORKING_DIR", "./rag_storage")
        self.log_dir = os.getenv("LOG_DIR", "./logs")
        
        # LightRAG settings
        self.workspace = os.getenv("WORKSPACE", "gemini_production")
        self.max_async = int(os.getenv("MAX_ASYNC", "6"))
        self.max_parallel_insert = int(os.getenv("MAX_PARALLEL_INSERT", "3"))
        self.top_k = int(os.getenv("TOP_K", "60"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
        self.chunk_overlap_size = int(os.getenv("CHUNK_OVERLAP_SIZE", "200"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "8000"))
        self.timeout = int(os.getenv("TIMEOUT", "300"))
        self.enable_llm_cache = os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true"
        self.enable_llm_cache_for_extract = os.getenv("ENABLE_LLM_CACHE_FOR_EXTRACT", "true").lower() == "true"
        self.cosine_threshold = float(os.getenv("COSINE_THRESHOLD", "0.2"))
        self.max_graph_nodes = int(os.getenv("MAX_GRAPH_NODES", "1000"))
        self.summary_language = os.getenv("SUMMARY_LANGUAGE", "English")
        self.max_gleaning = int(os.getenv("MAX_GLEANING", "2"))
        
        # Storage settings
        self.kv_storage = os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
        self.doc_status_storage = os.getenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage")
        self.graph_storage = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
        self.vector_storage = os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
        
        # Rerank settings
        self.enable_rerank = os.getenv("ENABLE_RERANK", "true").lower() == "true"
        self.min_rerank_score = float(os.getenv("MIN_RERANK_SCORE", "0.3"))
        self.rerank_mode = os.getenv("RERANK_MODE", "llm")
        self.rerank_llm_model = os.getenv("RERANK_LLM_MODEL", "gemini-2.0-flash")
        self.rerank_max_docs = int(os.getenv("RERANK_MAX_DOCS", "20"))
        
        # Monitoring settings
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.enable_token_tracking = os.getenv("ENABLE_TOKEN_TRACKING", "true").lower() == "true"
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        
        # Auto scan
        self.auto_scan_at_startup = os.getenv("AUTO_SCAN_AT_STARTUP", "false").lower() == "true"
        
        # UI settings
        self.webui_title = os.getenv("WEBUI_TITLE", "LightRAG Gemini 2.0 Flash Pipeline")
        self.webui_description = os.getenv("WEBUI_DESCRIPTION", "Production-Ready Knowledge Graph RAG System")


class HealthStatus(BaseModel):
    """Health status response model"""
    status: str
    version: str
    timestamp: float
    components: Dict[str, Any]
    configuration: Dict[str, Any]
    performance: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Metrics response model"""
    server_uptime: float
    total_requests: int
    total_documents: int
    total_queries: int
    token_usage: Dict[str, Any]
    performance_metrics: Dict[str, Any]


# Global variables
server_config = ServerConfig()
rag_instance = None
start_time = time.time()
request_counter = 0
query_counter = 0
document_counter = 0


async def setup_gemini_rag() -> LightRAG:
    """Setup LightRAG with Gemini 2.0 Flash integration"""
    logger.info("Initializing LightRAG with Gemini 2.0 Flash...")
    
    # Validate configurations
    if not validate_gemini_config():
        raise RuntimeError("Gemini LLM configuration validation failed")
    
    if not validate_gemini_embedding_config():
        raise RuntimeError("Gemini embedding configuration validation failed")
    
    if not validate_tokenizer_setup():
        raise RuntimeError("Gemma tokenizer setup validation failed")
    
    # Create tokenizer
    tokenizer = get_gemma_tokenizer(os.getenv("LLM_MODEL", "gemini-2.0-flash"))
    
    # Create embedding function
    embedding_config = GeminiEmbeddingConfig.from_env()
    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_config.embedding_dim,
        max_token_size=embedding_config.max_token_size,
        func=gemini_embed,
    )
    
    # Setup LLM-based rerank function if configured
    rerank_model_func = None
    if server_config.enable_rerank:
        rerank_model_func = create_llm_rerank_function(gemini_model_complete)
        logger.info(f"LLM-based reranking configured with model: {os.getenv('RERANK_LLM_MODEL', 'gemini-2.0-flash')}")
    else:
        logger.info("Reranking disabled")
    
    # Create LightRAG instance
    rag = LightRAG(
        working_dir=server_config.working_dir,
        workspace=server_config.workspace,
        llm_model_func=gemini_model_complete,
        llm_model_name=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
        llm_model_max_async=server_config.max_async,
        chunk_token_size=server_config.chunk_size,
        chunk_overlap_token_size=server_config.chunk_overlap_size,
        tokenizer=tokenizer,
        embedding_func=embedding_func,
        kv_storage=server_config.kv_storage,
        graph_storage=server_config.graph_storage,
        vector_storage=server_config.vector_storage,
        doc_status_storage=server_config.doc_status_storage,
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": server_config.cosine_threshold
        },
        enable_llm_cache=server_config.enable_llm_cache,
        enable_llm_cache_for_entity_extract=server_config.enable_llm_cache_for_extract,
        rerank_model_func=rerank_model_func,
        auto_manage_storages_states=False,
        max_parallel_insert=server_config.max_parallel_insert,
        max_graph_nodes=server_config.max_graph_nodes,
        summary_max_tokens=server_config.max_tokens,
        entity_extract_max_gleaning=server_config.max_gleaning,
        addon_params={"language": server_config.summary_language},
        llm_model_kwargs={
            "timeout": server_config.timeout,
        }
    )
    
    logger.info("LightRAG with Gemini 2.0 Flash initialized successfully")
    return rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global rag_instance
    
    try:
        # Setup logging
        setup_logger("lightrag", level=logging.INFO if not server_config.debug_mode else logging.DEBUG)
        
        # Create directories
        Path(server_config.working_dir).mkdir(parents=True, exist_ok=True)
        Path(server_config.input_dir).mkdir(parents=True, exist_ok=True)
        Path(server_config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize RAG
        rag_instance = await setup_gemini_rag()
        
        # Initialize storages
        await rag_instance.initialize_storages()
        await initialize_pipeline_status()
        
        logger.info("✅ LightRAG Gemini 2.0 Flash server startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        raise
    finally:
        # Cleanup
        if rag_instance:
            try:
                await rag_instance.finalize_storages()
                logger.info("✅ Server cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


# Create FastAPI app
app = FastAPI(
    title="LightRAG Gemini 2.0 Flash Server",
    description="Production-ready Knowledge Graph RAG system with Gemini 2.0 Flash",
    version="1.0.0",
    lifespan=lifespan,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
if server_config.cors_origins == "*":
    origins = ["*"]
else:
    origins = [origin.strip() for origin in server_config.cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request counter middleware
@app.middleware("http")
async def count_requests(request, call_next):
    global request_counter
    request_counter += 1
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Process-Time"] = str(process_time)
    return response


    # No authentication - simplified for ease of use


@app.get("/")
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Comprehensive health check endpoint"""
    global rag_instance, start_time
    
    try:
        # Component health checks
        components = {}
        
        # Check Gemini LLM
        try:
            gemini_llm = get_gemini_llm()
            components["gemini_llm"] = {
                "status": "healthy",
                "model": gemini_llm.config.model,
                "token_tracking": gemini_llm.config.enable_token_tracking
            }
        except Exception as e:
            components["gemini_llm"] = {"status": "unhealthy", "error": str(e)}
        
        # Check Gemini embeddings
        try:
            gemini_emb = get_gemini_embeddings()
            components["gemini_embeddings"] = {
                "status": "healthy",
                "model": gemini_emb.config.model,
                "dimension": gemini_emb.config.embedding_dim,
                "cache_enabled": gemini_emb.config.enable_caching
            }
        except Exception as e:
            components["gemini_embeddings"] = {"status": "unhealthy", "error": str(e)}
        
        # Check tokenizer
        try:
            tokenizer = get_gemma_tokenizer()
            components["tokenizer"] = {
                "status": "healthy",
                "model": tokenizer.model_name,
                "limits": tokenizer.get_token_limits()
            }
        except Exception as e:
            components["tokenizer"] = {"status": "unhealthy", "error": str(e)}
        
        # Check RAG instance
        if rag_instance:
            components["rag"] = {
                "status": "healthy",
                "workspace": rag_instance.workspace,
                "working_dir": rag_instance.working_dir,
                "storages": {
                    "kv_storage": type(rag_instance.text_chunks).__name__,
                    "vector_storage": type(rag_instance.chunks_vdb).__name__,
                    "graph_storage": type(rag_instance.chunk_entity_relation_graph).__name__,
                }
            }
        else:
            components["rag"] = {"status": "unhealthy", "error": "RAG instance not initialized"}
        
        # Configuration summary
        configuration = {
            "server": {
                "host": server_config.host,
                "port": server_config.port,
                "workspace": server_config.workspace,
                "cors_origins": origins[:3] if len(origins) > 3 else origins,  # Limit output
            },
            "llm": {
                "model": os.getenv("LLM_MODEL", "gemini-2.0-flash"),
                "max_async": server_config.max_async,
                "enable_cache": server_config.enable_llm_cache,
                "timeout": server_config.timeout,
            },
            "embeddings": {
                "model": os.getenv("EMBEDDING_MODEL", "text-embedding-004"),
                "dimension": int(os.getenv("EMBEDDING_DIM", "768")),
                "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
            },
            "rag": {
                "chunk_size": server_config.chunk_size,
                "top_k": server_config.top_k,
                "enable_rerank": server_config.enable_rerank,
                "max_graph_nodes": server_config.max_graph_nodes,
            }
        }
        
        # Performance metrics
        uptime = time.time() - start_time
        performance = {
            "uptime_seconds": uptime,
            "total_requests": request_counter,
            "requests_per_second": request_counter / uptime if uptime > 0 else 0,
            "total_queries": query_counter,
            "total_documents": document_counter,
        }
        
        # Add token usage if available
        if server_config.enable_token_tracking:
            try:
                gemini_llm = get_gemini_llm()
                token_usage = gemini_llm.get_token_usage()
                performance["token_usage"] = token_usage
            except:
                pass
        
        # Overall status
        overall_status = "healthy" if all(
            comp.get("status") == "healthy" for comp in components.values()
        ) else "degraded"
        
        return HealthStatus(
            status=overall_status,
            version="1.0.0",
            timestamp=time.time(),
            components=components,
            configuration=configuration,
            performance=performance
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get detailed server metrics"""
    global start_time, request_counter, query_counter, document_counter
    
    try:
        uptime = time.time() - start_time
        
        # Get token usage
        token_usage = {}
        if server_config.enable_token_tracking:
            try:
                gemini_llm = get_gemini_llm()
                token_usage = gemini_llm.get_token_usage()
            except:
                token_usage = {"error": "Token tracking not available"}
        
        # Performance metrics
        performance_metrics = {
            "requests_per_second": request_counter / uptime if uptime > 0 else 0,
            "queries_per_second": query_counter / uptime if uptime > 0 else 0,
            "documents_per_second": document_counter / uptime if uptime > 0 else 0,
            "average_request_time": 0,  # Would need request tracking
        }
        
        return MetricsResponse(
            server_uptime=uptime,
            total_requests=request_counter,
            total_documents=document_counter,
            total_queries=query_counter,
            token_usage=token_usage,
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


@app.post("/admin/clear-token-usage")
async def clear_token_usage():
    """Clear token usage statistics"""
    try:
        if server_config.enable_token_tracking:
            gemini_llm = get_gemini_llm()
            gemini_llm.reset_token_tracker()
            return {"status": "success", "message": "Token usage statistics cleared"}
        else:
            return {"status": "info", "message": "Token tracking is disabled"}
    except Exception as e:
        logger.error(f"Error clearing token usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/clear-embedding-cache")
async def clear_embedding_cache():
    """Clear embedding cache"""
    try:
        gemini_emb = get_gemini_embeddings()
        await gemini_emb.clear_cache()
        return {"status": "success", "message": "Embedding cache cleared"}
    except Exception as e:
        logger.error(f"Error clearing embedding cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Include LightRAG routers
doc_manager = DocumentManager(server_config.input_dir, workspace=server_config.workspace)

async def get_rag_instance():
    """Dependency to get RAG instance"""
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_instance

# Custom wrapper to track queries and documents
@app.middleware("http")
async def track_operations(request, call_next):
    global query_counter, document_counter
    
    response = await call_next(request)
    
    # Track query operations
    if "/query" in str(request.url):
        query_counter += 1
    
    # Track document operations
    if "/documents" in str(request.url) and request.method in ["POST", "PUT"]:
        document_counter += 1
    
    return response

# Add RAG routes
app.include_router(
    create_document_routes(lambda: rag_instance, doc_manager, None),
    prefix="/documents",
    tags=["documents"]
)

app.include_router(
    create_query_routes(lambda: rag_instance, None, server_config.top_k),
    prefix="",
    tags=["query"]
)

app.include_router(
    create_graph_routes(lambda: rag_instance, None),
    prefix="/graph",
    tags=["graph"]
)

# API setup completed - no additional Ollama compatibility needed


# Web UI replaced with Streamlit interface
logger.info("Web UI replaced with Streamlit interface (run: streamlit run streamlit_app.py)")


def main():
    """Main entry point"""
    try:
        logger.info(f"🚀 Starting LightRAG Gemini 2.0 Flash Server...")
        logger.info(f"📊 Configuration: {server_config.host}:{server_config.port}")
        logger.info(f"🤖 Model: {os.getenv('LLM_MODEL', 'gemini-2.0-flash')}")
        logger.info(f"🔍 Embeddings: {os.getenv('EMBEDDING_MODEL', 'text-embedding-004')}")
        logger.info(f"💾 Workspace: {server_config.workspace}")
        
        # Uvicorn configuration
        uvicorn_config = {
            "app": app,
            "host": server_config.host,
            "port": server_config.port,
            "log_level": "info" if not server_config.debug_mode else "debug",
            "access_log": True,
        }
        
        # Start server
        uvicorn.run(**uvicorn_config)
        
    except KeyboardInterrupt:
        logger.info("👋 Server shutdown requested")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 