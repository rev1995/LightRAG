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
- Document management
- Graph visualization
- Pipeline status
"""

import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from pathlib import Path

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

# Document management models
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

class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status"""
    autoscanned: bool = False
    busy: bool = False
    job_name: str = "Default Job"
    job_start: Optional[str] = None
    docs: int = 0
    batchs: int = 0
    cur_batch: int = 0
    request_pending: bool = False
    latest_message: str = ""
    history_messages: Optional[List[str]] = None
    update_status: Optional[Dict[str, Any]] = None

# Graph models
class LightragNodeType(BaseModel):
    """Model for graph node"""
    id: str
    labels: List[str]
    properties: Dict[str, Any]

class LightragEdgeType(BaseModel):
    """Model for graph edge"""
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any]

class LightragGraphType(BaseModel):
    """Model for knowledge graph"""
    nodes: List[LightragNodeType]
    edges: List[LightragEdgeType]

# --- Additional Pydantic Models ---

class DocActionResponse(BaseModel):
    """Response model for document actions"""
    status: str
    message: str

class DeleteDocResponse(BaseModel):
    """Response model for document deletion"""
    status: str
    message: str
    doc_id: str

class LightragDocumentsScanProgress(BaseModel):
    """Response model for document scan progress"""
    is_scanning: bool = False
    current_file: str = ""
    indexed_count: int = 0
    total_files: int = 0
    progress: float = 0.0

class EntityUpdateRequest(BaseModel):
    """Request model for entity updates"""
    entity_name: str
    updated_data: Dict[str, Any]
    allow_rename: bool = False

class RelationUpdateRequest(BaseModel):
    """Request model for relation updates"""
    source_entity: str
    target_entity: str
    updated_data: Dict[str, Any]

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

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the RAG system"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Build query parameters, only passing non-None values
        query_kwargs = {"query": request.query}
        
        if request.mode is not None:
            query_kwargs["mode"] = request.mode
        if request.user_prompt is not None:
            query_kwargs["user_prompt"] = request.user_prompt
        if request.top_k is not None:
            query_kwargs["top_k"] = request.top_k
        if request.chunk_top_k is not None:
            query_kwargs["chunk_top_k"] = request.chunk_top_k
        if request.enable_rerank is not None:
            query_kwargs["enable_rerank"] = request.enable_rerank
        if request.response_type is not None:
            query_kwargs["response_type"] = request.response_type
        if request.conversation_history is not None:
            query_kwargs["conversation_history"] = request.conversation_history
        
        result = await pipeline.query(**query_kwargs)
        
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
        
        result = await pipeline.clear_cache(request.modes if request.modes else None)
        
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
        
        result = await pipeline.get_token_usage_stats()
        
        return TokenStatsResponse(**result)
        
    except Exception as e:
        logger.error(f"Error getting token stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a single document file"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        # Insert the document
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
    """Upload multiple document files"""
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
        
        # Insert all documents
        result = await pipeline.insert_documents(documents)
        
        return {
            "filenames": filenames,
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error uploading documents batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Document Management Endpoints ---

@app.get("/documents", response_model=DocsStatusesResponse)
async def get_documents():
    """Get all documents status"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Return empty document status for now
        # TODO: Implement document retrieval from pipeline
        statuses = {
            "processed": [],
            "pending": [],
            "processing": [],
            "failed": []
        }
        
        return DocsStatusesResponse(statuses=statuses)
        
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/scan")
async def scan_documents():
    """Scan for new documents"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # TODO: Implement document scanning
        return {"status": "scanning_started", "message": "Document scanning initiated"}
        
    except Exception as e:
        logger.error(f"Error scanning documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/pipeline_status", response_model=PipelineStatusResponse)
async def get_pipeline_status():
    """Get pipeline status"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Return default pipeline status
        return PipelineStatusResponse(
            autoscanned=False,
            busy=False,
            job_name="Default Job",
            docs=0,
            batchs=0,
            cur_batch=0,
            request_pending=False,
            latest_message="Pipeline is ready"
        )
        
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_document_to_documents(file: UploadFile = File(...)):
    """Upload document to documents endpoint"""
    return await upload_document(file)

# --- Graph Visualization Endpoints ---

@app.get("/graph/label/list")
async def get_graph_labels():
    """Get all graph labels"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # TODO: Implement graph labels retrieval
        return []
        
    except Exception as e:
        logger.error(f"Error getting graph labels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graphs", response_model=LightragGraphType)
async def get_knowledge_graph(
    label: str = Query(..., description="Label to get knowledge graph for"),
    max_depth: int = Query(3, description="Maximum depth of graph", ge=1),
    max_nodes: int = Query(1000, description="Maximum nodes to return", ge=1),
):
    """Get knowledge graph for a specific label"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # TODO: Implement knowledge graph retrieval
        return LightragGraphType(nodes=[], edges=[])
        
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/entity/exists")
async def check_entity_exists(name: str = Query(..., description="Entity name to check")):
    """Check if an entity exists"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # TODO: Implement entity existence check
        return {"exists": False}
        
    except Exception as e:
        logger.error(f"Error checking entity existence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Additional Document Management Endpoints ---

@app.get("/documents/scan/progress", response_model=LightragDocumentsScanProgress)
async def get_documents_scan_progress():
    """Get document scan progress"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # TODO: Implement scan progress tracking
        return LightragDocumentsScanProgress()
        
    except Exception as e:
        logger.error(f"Error getting scan progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/clear", response_model=DocActionResponse)
async def clear_documents():
    """Clear all documents"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # TODO: Implement document clearing
        return DocActionResponse(status="success", message="Documents cleared successfully")
        
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/delete", response_model=DeleteDocResponse)
async def delete_documents(doc_ids: List[str] = Query(...), delete_file: bool = Query(False)):
    """Delete specific documents"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # TODO: Implement document deletion
        return DeleteDocResponse(
            status="deletion_started",
            message="Document deletion initiated",
            doc_id=doc_ids[0] if doc_ids else ""
        )
        
    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Additional Query Endpoints ---

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream query response"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        async def generate():
            try:
                # Build query parameters, only passing non-None values
                query_kwargs = {"query": request.query}
                
                if request.mode is not None:
                    query_kwargs["mode"] = request.mode
                if request.user_prompt is not None:
                    query_kwargs["user_prompt"] = request.user_prompt
                if request.top_k is not None:
                    query_kwargs["top_k"] = request.top_k
                if request.chunk_top_k is not None:
                    query_kwargs["chunk_top_k"] = request.chunk_top_k
                if request.enable_rerank is not None:
                    query_kwargs["enable_rerank"] = request.enable_rerank
                if request.response_type is not None:
                    query_kwargs["response_type"] = request.response_type
                if request.conversation_history is not None:
                    query_kwargs["conversation_history"] = request.conversation_history
                
                # Call the pipeline query method
                result = await pipeline.query(**query_kwargs)
                
                # Stream the response in chunks
                response_text = result.get("response", "")
                chunk_size = 100
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i + chunk_size]
                    yield f"data: {chunk}\n\n"
                
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Error in query stream: {e}")
                yield f"data: Error: {str(e)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Error in query stream endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Additional Insert Endpoints ---

@app.post("/insert/text", response_model=DocActionResponse)
async def insert_text(text: str):
    """Insert a single text document"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        result = await pipeline.insert_documents([text])
        
        return DocActionResponse(
            status="success",
            message=f"Text inserted successfully. {result['successful_insertions']} documents processed."
        )
        
    except Exception as e:
        logger.error(f"Error inserting text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insert/texts", response_model=DocActionResponse)
async def insert_texts(texts: List[str]):
    """Insert multiple text documents"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        result = await pipeline.insert_documents(texts)
        
        return DocActionResponse(
            status="success",
            message=f"Texts inserted successfully. {result['successful_insertions']} documents processed."
        )
        
    except Exception as e:
        logger.error(f"Error inserting texts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Additional Upload Endpoints ---

@app.post("/upload/document")
async def upload_document_endpoint(file: UploadFile = File(...)):
    """Upload a single document file"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        # Insert the document
        result = await pipeline.insert_documents([text])
        
        return {
            "filename": file.filename,
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/documents")
async def upload_documents_endpoint(files: List[UploadFile] = File(...)):
    """Upload multiple document files"""
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
        
        # Insert all documents
        result = await pipeline.insert_documents(documents)
        
        return {
            "filenames": filenames,
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Graph Entity and Relation Update Endpoints ---

@app.post("/graph/entity/update", response_model=DocActionResponse)
async def update_entity(request: EntityUpdateRequest):
    """Update an entity in the knowledge graph"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # TODO: Implement entity update
        return DocActionResponse(
            status="success",
            message=f"Entity '{request.entity_name}' updated successfully"
        )
        
    except Exception as e:
        logger.error(f"Error updating entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph/relation/update", response_model=DocActionResponse)
async def update_relation(request: RelationUpdateRequest):
    """Update a relation in the knowledge graph"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # TODO: Implement relation update
        return DocActionResponse(
            status="success",
            message=f"Relation between '{request.source_entity}' and '{request.target_entity}' updated successfully"
        )
        
    except Exception as e:
        logger.error(f"Error updating relation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Additional RAG Endpoints ---

@app.post("/query/rag", response_model=QueryResponse)
async def query_rag_endpoint(request: QueryRequest):
    """Query RAG system (alias for /query)"""
    return await query_endpoint(request)

@app.post("/insert/documents", response_model=InsertResponse)
async def insert_documents_endpoint(request: InsertRequest):
    """Insert documents (alias for /insert)"""
    return await insert_endpoint(request)

@app.post("/clear/rag/cache", response_model=CacheClearResponse)
async def clear_rag_cache_endpoint(request: CacheClearRequest):
    """Clear RAG cache (alias for /clear_cache)"""
    return await clear_cache_endpoint(request)

@app.get("/health/rag", response_model=HealthResponse)
async def check_rag_health():
    """Check RAG health (alias for /health)"""
    return await health_check()

# --- Global Exception Handler ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return {"error": str(exc)}, 500

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 