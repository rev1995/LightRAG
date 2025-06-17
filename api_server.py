# FILE: api_server.py
import uvicorn
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import the setup and processing functions from your new setup file
from rag_setup import initialize_rag_system, process_markdown_files, log
from lightrag import LightRAG, QueryParam

# --- Configuration ---
# Define the data directory at the application level
DATA_DIR = "./data"

# --- Global RAG Instance ---
# This will be initialized on server startup
rag_system: Optional[LightRAG] = None

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the FastAPI application."""
    global rag_system
    log.info("Server startup: Initializing RAG system...")
    if not all(os.getenv(key) for key in ["GEMINI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]):
        log.error("FATAL: Missing one or more required environment variables. Server cannot start.")
        raise RuntimeError("Missing required environment variables. Please check your .env file.")
    
    rag_system = await initialize_rag_system()
    log.info("RAG system is live.")
    yield
    log.info("Server shutdown: Finalizing RAG system storage...")
    if rag_system:
        await rag_system.finalize_storages()
    log.info("RAG system finalized. Goodbye.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LightRAG Gemini API Server",
    description="An API for interacting with a custom LightRAG system powered by Gemini and Neo4j.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models for API ---
class APIQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The query text.")
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        default="hybrid", description="The query mode to use."
    )
    stream: bool = Field(default=False, description="Whether to stream the response.")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="A list of past messages to provide context. Format: [{'role': 'user'|'assistant', 'content': '...'}]."
    )
    user_prompt: Optional[str] = Field(
        default=None,
        description="An optional instruction to the LLM on how to format the final response. Does not affect retrieval."
    )

class IngestResponse(BaseModel):
    status: str
    message: str

# --- API Endpoints ---
@app.post("/ingest", response_model=IngestResponse)
async def ingest_data(background_tasks: BackgroundTasks):
    """Triggers a background task to scan the DATA_DIR and ingest all markdown files."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system is not initialized.")
    
    background_tasks.add_task(process_markdown_files, rag_system, DATA_DIR)
    
    return IngestResponse(
        status="success",
        message=f"Ingestion process started for directory '{DATA_DIR}'. Check server logs for progress."
    )

@app.post("/query")
async def query_rag(request: APIQueryRequest):
    """
    Queries the RAG system with the specified parameters.
    Supports different modes, streaming, and conversation history.
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system is not initialized.")

    try:
        query_param = QueryParam(
            mode=request.mode,
            stream=request.stream,
            conversation_history=request.conversation_history or [],
            user_prompt=request.user_prompt
        )

        response_generator = await rag_system.aquery(request.query, param=query_param)

        if request.stream:
            async def stream_wrapper():
                try:
                    full_response = ""
                    async for chunk in response_generator:
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    log.info(f"Streamed response sent for query: '{request.query[:50]}...'")
                except Exception as e:
                    log.error(f"Error during streaming: {e}")
                    yield f"data: {json.dumps({'error': 'An error occurred during streaming.'})}\n\n"
            
            return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
        else:
            return {"response": response_generator}

    except Exception as e:
        log.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# --- Main execution block to run the server ---
if __name__ == "__main__":
    log.info("Starting LightRAG API server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)