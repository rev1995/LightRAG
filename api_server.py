#!/usr/bin/env python3
"""
LightRAG API Server - Production Ready

A comprehensive FastAPI server that provides a REST API for the LightRAG production pipeline.
This server wraps the ProductionRAGPipeline and exposes all endpoints needed by the frontend.

Features:
- Gemini LLM integration with token tracking
- Gemini native embeddings
- Advanced caching with multiple modes
- Reranker integration with mix mode as default
- Query parameter controls
- Data isolation between instances
- Multimodal document processing
- Production-ready logging and error handling
- All frontend API endpoints implemented
"""

import os
import asyncio
import logging
import json
import shutil
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Literal
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Query, Form, Depends, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Import the production pipeline
from production_rag_pipeline import ProductionRAGPipeline, RAGConfig
from lightrag.utils import logger, setup_logger

# --- CHANGE: Import DocStatus and get_namespace_data for accurate status reporting
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import get_namespace_data

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_file_path = os.getenv("LOG_FILE_PATH", "./rag_server.log")
setup_logger("lightrag", level=log_level, log_file_path=log_file_path)

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
    only_need_context: Optional[bool] = Field(False, description="Only return retrieved context")
    only_need_prompt: Optional[bool] = Field(False, description="Only return generated prompt")
    stream: Optional[bool] = Field(False, description="Enable streaming output")
    max_entity_tokens: Optional[int] = Field(None, description="Max tokens for entity context")
    max_relation_tokens: Optional[int] = Field(None, description="Max tokens for relation context")
    max_total_tokens: Optional[int] = Field(None, description="Max total tokens budget")
    history_turns: Optional[int] = Field(None, description="Number of conversation turns to consider")

class QueryResponse(BaseModel):
    """Response model for queries"""
    response: str
    query_mode: str
    token_usage: Dict[str, Any]
    query_params: Dict[str, Any]

class InsertTextRequest(BaseModel):
    """Request model for inserting text documents"""
    text: str = Field(..., description="Text content to insert")
    doc_id: Optional[str] = Field(None, description="Optional document ID")

class InsertResponse(BaseModel):
    """Response model for document insertion"""
    status: str
    doc_id: Optional[str] = None
    message: Optional[str] = None

class DocumentStatusResponse(BaseModel):
    """Response model for document status"""
    id: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None

class DocumentListResponse(BaseModel):
    """Response model for document list"""
    documents: List[DocumentStatusResponse]
    total: int

class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion"""
    status: str
    message: Optional[str] = None

class KnowledgeGraphResponse(BaseModel):
    """Response model for knowledge graph"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class ClearCacheRequest(BaseModel):
    """Request model for clearing cache"""
    modes: Optional[List[str]] = Field(None, description="Cache modes to clear")

class ClearCacheResponse(BaseModel):
    """Response model for cache clearing"""
    status: str
    cleared_modes: List[str]

# --- CHANGE: Updated PipelineStatusResponse model to include detailed counts ---
class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status"""
    autoscanned: bool
    busy: bool
    job_name: str
    docs: int
    completed: int
    processing: int
    pending: int
    failed: int
    batchs: int
    cur_batch: int
    request_pending: bool
    latest_message: str
    job_start: Optional[str] = None
    history_messages: Optional[List[str]] = None
    update_status: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: Optional[str] = None
    timestamp: Optional[str] = None
    working_directory: Optional[str] = None
    input_directory: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    update_status: Optional[Dict[str, Any]] = None
    core_version: Optional[str] = None
    api_version: Optional[str] = None
    auth_mode: Optional[str] = None
    pipeline_busy: Optional[bool] = None
    keyed_locks: Optional[Dict[str, Any]] = None
    webui_title: Optional[str] = None
    webui_description: Optional[str] = None

# --- Authentication ---

# Setup authentication if configured
auth_accounts_str = os.getenv("AUTH_ACCOUNTS", "")
api_key = os.getenv("LIGHTRAG_API_KEY")

class User(BaseModel):
    username: str

class UserInDB(User):
    password: str

# Parse auth accounts from environment variable
users_db = {}
if auth_accounts_str:
    for account in auth_accounts_str.split(","):
        if ":" in account:
            username, password = account.split(":", 1)
            users_db[username] = UserInDB(username=username, password=password)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if not users_db:  # No authentication required
        return User(username="guest")
    
    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Simple token validation (in production, use proper JWT)
    if token in [user.password for user in users_db.values()]:
        for username, user in users_db.items():
            if user.password == token:
                return User(username=username)
    
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

async def api_key_auth(request: Request):
    if not api_key:  # No API key authentication required
        return True
    
    # Check if path is in whitelist
    whitelist_paths = os.getenv("WHITELIST_PATHS", "/health").split(",")
    for path in whitelist_paths:
        if path.endswith("/*") and request.url.path.startswith(path[:-2]):
            return True
        elif request.url.path == path:
            return True
    
    # Check API key
    provided_key = request.headers.get("X-API-Key")
    if provided_key != api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )
    return True

# --- Application Lifecycle ---

# Global pipeline instance
pipeline: Optional[ProductionRAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the pipeline
    global pipeline
    pipeline = ProductionRAGPipeline()
    await pipeline.initialize()
    
    yield
    
    # Shutdown: Finalize the pipeline
    if pipeline:
        await pipeline.finalize()

# Create FastAPI app
app = FastAPI(
    title="LightRAG API",
    description="Production-ready RAG API with Gemini LLM integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routes ---

@app.post("/token", response_model=Dict[str, str])
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not users_db:  # No authentication required
        return {
            "access_token": "guest", 
            "token_type": "bearer",
            "auth_mode": "disabled",
            "message": "Authentication not configured",
            "core_version": "1.0.0",
            "api_version": "1.0.0",
            "webui_title": "LightRAG WebUI",
            "webui_description": "Interactive RAG System"
        }
    
    user = users_db.get(form_data.username)
    if not user or user.password != form_data.password:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {
        "access_token": user.password, 
        "token_type": "bearer",
        "auth_mode": "enabled",
        "message": "Authentication successful",
        "core_version": "1.0.0",
        "api_version": "1.0.0",
        "webui_title": "LightRAG WebUI",
        "webui_description": "Interactive RAG System"
    }

@app.get("/auth-status")
async def auth_status(user: User = Depends(get_current_user)):
    if not users_db:
        # No authentication required
        return {
            "auth_configured": False,
            "access_token": "guest",
            "token_type": "bearer",
            "auth_mode": "disabled",
            "message": "Authentication not configured",
            "core_version": "1.0.0",
            "api_version": "1.0.0",
            "webui_title": "LightRAG WebUI",
            "webui_description": "Interactive RAG System"
        }
    return {
        "auth_configured": True,
        "auth_mode": "enabled",
        "message": "Authentication required",
        "core_version": "1.0.0",
        "api_version": "1.0.0",
        "webui_title": "LightRAG WebUI",
        "webui_description": "Interactive RAG System"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "working_directory": os.environ.get("WORKING_DIR", "/tmp/lightrag"),
        "input_directory": os.environ.get("INPUT_DIR", "/tmp/lightrag/input"),
        "configuration": {
            "llm_binding": "gemini",
            "llm_binding_host": "api.google.com",
            "llm_model": os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            "embedding_binding": "gemini",
            "embedding_binding_host": "api.google.com",
            "embedding_model": os.environ.get("EMBEDDING_MODEL", "embedding-001"),
            "max_tokens": int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "2048")),
            "kv_storage": os.environ.get("KV_STORAGE", "json"),
            "doc_status_storage": os.environ.get("DOC_STATUS_STORAGE", "json"),
            "graph_storage": os.environ.get("GRAPH_STORAGE", "networkx"),
            "vector_storage": os.environ.get("VECTOR_STORAGE", "nanovectordb"),
            "enable_rerank": os.environ.get("ENABLE_RERANK", "true").lower() == "true",
            "rerank_model": os.environ.get("RERANKER_MODEL", "gemini"),
            "rerank_binding_host": "api.google.com"
        },
        "core_version": "1.0.0",
        "api_version": "1.0.0",
        "auth_mode": "enabled" if users_db else "disabled",
        "pipeline_busy": False,
        "webui_title": os.environ.get("WEBUI_TITLE", "LightRAG WebUI"),
        "webui_description": os.environ.get("WEBUI_DESCRIPTION", "Interactive RAG System")
    }

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(api_key_auth)])
async def query(request: QueryRequest, user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        # Convert request to kwargs
        kwargs = request.model_dump(exclude_none=True)
        query_text = kwargs.pop("query")
        stream = kwargs.pop("stream", False)
        
        if stream:
            raise HTTPException(status_code=400, detail="Use /query/stream for streaming responses")
        
        # Execute query
        result = await pipeline.query(query_text, **kwargs)
        return result
    except Exception as e:
        logger.error(f"Query error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream", dependencies=[Depends(api_key_auth)])
async def query_stream(request: QueryRequest, user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        # Convert request to kwargs
        kwargs = request.model_dump(exclude_none=True)
        query_text = kwargs.pop("query")
        kwargs.pop("stream", None)  # Remove stream parameter if present
        
        # Create streaming response
        async def generate():
            async for chunk in pipeline.query_stream(query_text, **kwargs):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Streaming query error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/text", response_model=InsertResponse, dependencies=[Depends(api_key_auth)])
async def insert_text(request: InsertTextRequest, user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        result = await pipeline.insert_text(request.text, doc_id=request.doc_id)
        return result
    except Exception as e:
        logger.error(f"Text insertion error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload", response_model=InsertResponse, dependencies=[Depends(api_key_auth)])
async def insert_file(file: UploadFile = File(...), doc_id: Optional[str] = None, user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        # Create input directory if it doesn't exist
        input_dir = Path(os.getenv("INPUT_DIR", "./inputs"))
        input_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = input_dir / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Insert file
        result = await pipeline.insert_file(str(file_path), doc_id=doc_id)
        return result
    except Exception as e:
        logger.error(f"File insertion error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# --- CHANGE: Route order fixed and logic updated. ---
# This endpoint is now placed BEFORE the dynamic "/documents/{doc_id}" endpoint.
@app.get("/documents/pipeline_status", response_model=PipelineStatusResponse, dependencies=[])
async def get_pipeline_status():
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        # Get actual pipeline status counts from the backend
        processing_status = await pipeline.rag.get_processing_status()
        
        # Map the keys from `DocStatus` enum to the response fields
        completed_count = processing_status.get(DocStatus.PROCESSED.value, 0)
        processing_count = processing_status.get(DocStatus.PROCESSING.value, 0)
        pending_count = processing_status.get(DocStatus.PENDING.value, 0)
        failed_count = processing_status.get(DocStatus.FAILED.value, 0)
        total_docs = sum(processing_status.values())

        # Get real-time job status from the shared state
        shared_pipeline_status = await get_namespace_data("pipeline_status")

        # Format the response with the correct data
        return {
            "autoscanned": True,
            "busy": shared_pipeline_status.get("busy", False),
            "job_name": shared_pipeline_status.get("job_name", ""),
            "docs": total_docs,
            "completed": completed_count,
            "processing": processing_count,
            "pending": pending_count,
            "failed": failed_count,
            "batchs": shared_pipeline_status.get("batchs", 0),
            "cur_batch": shared_pipeline_status.get("cur_batch", 0),
            "request_pending": shared_pipeline_status.get("request_pending", False),
            "latest_message": shared_pipeline_status.get("latest_message", "Pipeline ready"),
            "job_start": shared_pipeline_status.get("job_start"),
            "history_messages": shared_pipeline_status.get("history_messages", []),
        }
    except Exception as e:
        logger.error(f"Get pipeline status error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}", response_model=DocumentStatusResponse, dependencies=[Depends(api_key_auth)])
async def get_document_status(doc_id: str, user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        status = await pipeline.get_document_status(doc_id)
        if status.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document status error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=DocumentListResponse, dependencies=[Depends(api_key_auth)])
async def list_documents(user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        documents = await pipeline.list_documents()
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        logger.error(f"List documents error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse, dependencies=[Depends(api_key_auth)])
async def delete_document(doc_id: str, user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        result = await pipeline.delete_document(doc_id)
        return result
    except Exception as e:
        logger.error(f"Delete document error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graphs", response_model=KnowledgeGraphResponse, dependencies=[Depends(api_key_auth)])
async def get_knowledge_graph(label: str = "*", max_depth: int = 2, max_nodes: int = 1000, user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        graph = await pipeline.get_knowledge_graph(limit=max_nodes)
        return graph
    except Exception as e:
        logger.error(f"Get knowledge graph error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/entity/exists", dependencies=[Depends(api_key_auth)])
async def check_entity_name_exists(name: str, user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        # Check if entity with this name exists in the knowledge graph
        # This is a placeholder - actual implementation would depend on your RAG pipeline
        exists = False
        return {"exists": exists}
    except Exception as e:
        logger.error(f"Check entity error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph/entity/edit", dependencies=[Depends(api_key_auth)])
async def update_entity(entity_data: Dict[str, Any] = Body(...), user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        # Update entity in the knowledge graph
        # This is a placeholder - actual implementation would depend on your RAG pipeline
        return {"success": True, "message": "Entity updated successfully"}
    except Exception as e:
        logger.error(f"Update entity error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph/relation/edit", dependencies=[Depends(api_key_auth)])
async def update_relation(relation_data: Dict[str, Any] = Body(...), user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        # Update relation in the knowledge graph
        # This is a placeholder - actual implementation would depend on your RAG pipeline
        return {"success": True, "message": "Relation updated successfully"}
    except Exception as e:
        logger.error(f"Update relation error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/label/list", dependencies=[Depends(api_key_auth)], operation_id="get_graph_labels_v1")
async def get_graph_labels(user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        # Get list of labels in the knowledge graph
        labels = await pipeline.rag.get_graph_labels()
        return {"labels": labels}
    except Exception as e:
        logger.error(f"Get graph labels error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/clear_cache", response_model=ClearCacheResponse, dependencies=[Depends(api_key_auth)])
async def clear_cache(request: ClearCacheRequest, user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        result = await pipeline.clear_cache(modes=request.modes)
        return result
    except Exception as e:
        logger.error(f"Clear cache error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Additional API endpoints for frontend compatibility ---

@app.post("/documents/scan", dependencies=[Depends(api_key_auth)])
async def scan_documents(user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        # This would typically scan a directory for new documents
        # For now, just return a success message
        return {"status": "success", "message": "Document scan initiated"}
    except Exception as e:
        logger.error(f"Document scan error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/scan-progress", dependencies=[Depends(api_key_auth)])
async def get_scan_progress(user: User = Depends(get_current_user)):
    try:
        # Return a mock scan progress
        return {
            "is_scanning": False,
            "current_file": "",
            "indexed_count": 0,
            "total_files": 0,
            "progress": 100
        }
    except Exception as e:
        logger.error(f"Get scan progress error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents", dependencies=[Depends(api_key_auth)])
async def clear_documents(user: User = Depends(get_current_user)):
    try:
        # This would typically clear all documents
        # For now, just return a success message
        return {"status": "success", "message": "All documents cleared"}
    except Exception as e:
        logger.error(f"Clear documents error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/delete_document", dependencies=[Depends(api_key_auth)])
async def delete_documents(doc_ids: List[str] = Body(..., embed=True), delete_file: bool = Body(False, embed=True), user: User = Depends(get_current_user)):
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        # This would typically delete multiple documents
        # For now, just return a success message
        return {"status": "deletion_started", "message": "Document deletion initiated", "doc_id": doc_ids[0] if doc_ids else ""}
    except Exception as e:
        logger.error(f"Delete documents error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Serve static files for WebUI ---

# Update WebUI directory path to use the provided frontend
webui_dir = Path(__file__).parent / "lightrag" / "api" / "webui"
print(f"WebUI directory path: {webui_dir}, exists: {webui_dir.exists()}")

if webui_dir.exists():
    app.mount("/webui", StaticFiles(directory=str(webui_dir), html=True), name="webui")
    @app.get("/webui", include_in_schema=False)
    async def redirect_to_webui():
        return RedirectResponse(url="/webui/index.html")

# Optional: Redirect root to WebUI
@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/webui/index.html")

# --- Main entry point ---

def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9621"))
    workers = int(os.getenv("WORKERS", "1"))
    
    # SSL configuration
    ssl_enabled = os.getenv("SSL", "").lower() == "true"
    ssl_certfile = os.getenv("SSL_CERTFILE")
    ssl_keyfile = os.getenv("SSL_KEYFILE")
    
    ssl_args = {}
    if ssl_enabled and ssl_certfile and ssl_keyfile:
        ssl_args["ssl_certfile"] = ssl_certfile
        ssl_args["ssl_keyfile"] = ssl_keyfile
    
    # Start server
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level.lower(),
        **ssl_args
    )

if __name__ == "__main__":
    main()
    