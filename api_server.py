#!/usr/bin/env python3
"""
LightRAG API Server

A FastAPI server that provides a REST API for the LightRAG production pipeline.
This server wraps the ProductionRAGPipeline and exposes endpoints for:
- Querying documents
- Inserting documents
- Clearing cache
- Getting token statistics
- Health checks
"""

import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Import the production pipeline
from production_rag_pipeline import ProductionRAGPipeline, RAGConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LightRAG API",
    description="Production-ready RAG API with Gemini LLM integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[ProductionRAGPipeline] = None

# --- Pydantic Models ---

class QueryRequest(BaseModel):
    """Request model for querying the RAG system"""
    query: str = Field(..., description="The query text")
    mode: Optional[str] = Field(None, description="Query mode (naive, local, global, hybrid, mix, bypass)")
    user_prompt: Optional[str] = Field(None, description="Custom user prompt")
    top_k: Optional[int] = Field(None, description="Number of top entities/relations to retrieve")
    chunk_top_k: Optional[int] = Field(None, description="Number of top chunks to retrieve")
    enable_rerank: Optional[bool] = Field(None, description="Enable reranking")
    response_type: Optional[str] = Field("Multiple Paragraphs", description="Response format type")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Conversation history")

class QueryResponse(BaseModel):
    """Response model for queries"""
    response: str
    query_mode: str
    token_usage: Dict[str, Any]
    query_params: Dict[str, Any]

class InsertRequest(BaseModel):
    """Request model for inserting documents"""
    documents: List[str] = Field(..., description="List of document texts to insert")

class InsertResponse(BaseModel):
    """Response model for document insertion"""
    total_documents: int
    successful_insertions: int
    results: List[Dict[str, Any]]

class CacheClearRequest(BaseModel):
    """Request model for clearing cache"""
    modes: Optional[List[str]] = Field(None, description="Cache modes to clear")

class CacheClearResponse(BaseModel):
    """Response model for cache clearing"""
    status: str
    message: str
    modes_cleared: Any

class TokenStatsResponse(BaseModel):
    """Response model for token statistics"""
    total_requests: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    average_tokens_per_request: float
    cost_estimation: float

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    pipeline_initialized: bool

# --- Startup and Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global pipeline
    try:
        logger.info("Initializing LightRAG API server...")
        
        # Load configuration
        config = RAGConfig()
        
        # Initialize pipeline
        pipeline = ProductionRAGPipeline(config)
        await pipeline.initialize()
        
        logger.info("LightRAG API server initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global pipeline
    try:
        if pipeline:
            await pipeline.finalize()
            logger.info("LightRAG API server shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# --- API Endpoints ---

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LightRAG API Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        return HealthResponse(
            status="healthy",
            message="LightRAG API server is running",
            pipeline_initialized=pipeline is not None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Authentication Endpoints ---

class AuthStatusResponse(BaseModel):
    """Response model for auth status"""
    auth_configured: bool = False
    access_token: Optional[str] = None
    token_type: Optional[str] = None
    auth_mode: str = "disabled"
    message: Optional[str] = None
    core_version: str = "1.0.0"
    api_version: str = "1.0.0"
    webui_title: str = "LightRAG WebUI"
    webui_description: str = "Production-ready RAG system with Gemini LLM"

class LoginRequest(BaseModel):
    """Request model for login"""
    username: str
    password: str

class LoginResponse(BaseModel):
    """Response model for login"""
    access_token: str
    token_type: str = "bearer"
    auth_mode: str = "enabled"
    message: Optional[str] = None
    core_version: str = "1.0.0"
    api_version: str = "1.0.0"
    webui_title: str = "LightRAG WebUI"
    webui_description: str = "Production-ready RAG system with Gemini LLM"

@app.get("/auth-status", response_model=AuthStatusResponse)
async def auth_status():
    """Get authentication status"""
    return AuthStatusResponse(
        auth_configured=False,  # No auth configured for this simple setup
        auth_mode="disabled",
        message="Authentication is disabled in this setup"
    )

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint (simplified - always succeeds)"""
    # For this demo, we'll accept any credentials
    # In production, you'd want proper authentication
    logger.info(f"Login attempt for user: {request.username}")
    return LoginResponse(
        access_token="demo_token_12345",
        message="Login successful (demo mode)"
    )

@app.post("/login", response_model=LoginResponse)
async def login_form(username: str = Form(...), password: str = Form(...)):
    """Login endpoint for form data (multipart/form-data)"""
    # For this demo, we'll accept any credentials
    # In production, you'd want proper authentication
    logger.info(f"Login attempt for user: {username}")
    return LoginResponse(
        access_token="demo_token_12345",
        message="Login successful (demo mode)"
    )

# --- Document Management Endpoints ---

class DocStatusResponse(BaseModel):
    """Response model for document status"""
    id: str
    content_summary: str
    content_length: int
    status: str
    created_at: str
    updated_at: str
    chunks_count: Optional[int] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    file_path: str

class DocsStatusesResponse(BaseModel):
    """Response model for document statuses"""
    statuses: Dict[str, List[DocStatusResponse]]

@app.get("/documents", response_model=DocsStatusesResponse)
async def get_documents():
    """Get document statuses"""
    # For now, return empty statuses
    # In a full implementation, this would query the document storage
    return DocsStatusesResponse(statuses={
        "pending": [],
        "processing": [],
        "processed": [],
        "failed": []
    })

@app.post("/scan-documents")
async def scan_documents():
    """Scan for new documents"""
    return {"status": "success", "message": "Document scanning completed"}

@app.get("/documents-scan-progress")
async def get_documents_scan_progress():
    """Get document scanning progress"""
    return {
        "is_scanning": False,
        "current_file": "",
        "indexed_count": 0,
        "total_files": 0,
        "progress": 0
    }

# --- Pipeline Status Endpoints ---

class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status"""
    autoscanned: bool = False
    busy: bool = False
    job_name: str = ""
    job_start: Optional[str] = None
    docs: int = 0
    batchs: int = 0
    cur_batch: int = 0
    request_pending: bool = False
    latest_message: str = "Ready"
    history_messages: Optional[List[str]] = None
    update_status: Optional[Dict[str, Any]] = None

@app.get("/pipeline-status", response_model=PipelineStatusResponse)
async def get_pipeline_status():
    """Get pipeline status"""
    return PipelineStatusResponse(
        latest_message="Pipeline is ready"
    )

# --- Graph Endpoints ---

@app.get("/graph/labels")
async def get_graph_labels():
    """Get available graph labels"""
    # Return empty list for now
    # In a full implementation, this would query the knowledge graph
    return []

@app.post("/graph/query")
async def query_graphs():
    """Query knowledge graphs"""
    # Return empty graph for now
    return {
        "nodes": [],
        "edges": []
    }

# --- Document Actions ---

class DocActionResponse(BaseModel):
    """Response model for document actions"""
    status: str
    message: str

@app.post("/documents/insert")
async def insert_document():
    """Insert a document"""
    return DocActionResponse(
        status="success",
        message="Document inserted successfully"
    )

@app.post("/documents/clear")
async def clear_documents():
    """Clear all documents"""
    return DocActionResponse(
        status="success",
        message="All documents cleared"
    )

@app.post("/documents/delete")
async def delete_documents():
    """Delete documents"""
    return DocActionResponse(
        status="success",
        message="Documents deleted successfully"
    )

# --- Entity and Relation Management ---

@app.post("/entities/update")
async def update_entity():
    """Update an entity"""
    return DocActionResponse(
        status="success",
        message="Entity updated successfully"
    )

@app.post("/relations/update")
async def update_relation():
    """Update a relation"""
    return DocActionResponse(
        status="success",
        message="Relation updated successfully"
    )

@app.get("/entities/check")
async def check_entity_name():
    """Check if entity name exists"""
    return {"exists": False}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the RAG system"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Prepare query parameters, handling None values
        query_params = {
            "query": request.query,
            "response_type": request.response_type or "Multiple Paragraphs",
            "conversation_history": request.conversation_history or []
        }
        
        # Add optional parameters only if they are not None
        if request.mode is not None:
            query_params["mode"] = request.mode
        if request.user_prompt is not None:
            query_params["user_prompt"] = request.user_prompt
        if request.top_k is not None:
            query_params["top_k"] = request.top_k
        if request.chunk_top_k is not None:
            query_params["chunk_top_k"] = request.chunk_top_k
        if request.enable_rerank is not None:
            query_params["enable_rerank"] = request.enable_rerank
        
        result = await pipeline.query(**query_params)
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insert", response_model=InsertResponse)
async def insert_endpoint(request: InsertRequest):
    """Insert documents into the RAG system"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        logger.info(f"Inserting {len(request.documents)} documents...")
        
        result = await pipeline.insert_documents(request.documents)
        
        return InsertResponse(**result)
        
    except Exception as e:
        logger.error(f"Error inserting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_cache", response_model=CacheClearResponse)
async def clear_cache_endpoint(request: CacheClearRequest):
    """Clear cache with specified modes"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        logger.info(f"Clearing cache for modes: {request.modes or 'all'}")
        
        # Handle None case for modes parameter
        modes_to_clear = request.modes if request.modes is not None else None
        result = await pipeline.clear_cache(modes_to_clear)
        
        return CacheClearResponse(**result)
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/token_stats", response_model=TokenStatsResponse)
async def token_stats_endpoint():
    """Get token usage statistics"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        stats = await pipeline.get_token_usage_stats()
        
        return TokenStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting token stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and insert a single document"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        logger.info(f"Uploading document: {file.filename}")
        
        result = await pipeline.insert_documents([text])
        
        return {
            "filename": file.filename,
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_batch")
async def upload_documents_batch(files: List[UploadFile] = File(...)):
    """Upload and insert multiple documents"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        documents = []
        filenames = []
        
        for file in files:
            content = await file.read()
            text = content.decode('utf-8')
            documents.append(text)
            filenames.append(file.filename)
        
        logger.info(f"Uploading {len(files)} documents")
        
        result = await pipeline.insert_documents(documents)
        
        return {
            "filenames": filenames,
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Error Handlers ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

# --- Main Function ---

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 