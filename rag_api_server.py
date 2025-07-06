# rag_api_server.py

# --- Standard Library Imports ---
import os
import asyncio
import logging
import logging.handlers
from pathlib import Path
import hashlib
import dataclasses
from datetime import datetime, timezone
from typing import List, Optional, Union, AsyncIterator, Dict, Any, Literal
import threading
import uuid
from collections import deque
import time

# --- API & Server Management ---
import uvicorn
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# --- Dependency Management ---
# This block ensures that all required packages are installed automatically upon first run.
# It uses pipmaster for dynamic installation, simplifying setup for new environments.
try:
    import pipmaster as pm
    # Core RAG dependencies
    if not pm.is_installed("google-generativeai"): pm.install("google-generativeai")
    if not pm.is_installed("sentence-transformers"): pm.install("sentence-transformers")
    if not pm.is_installed("sentencepiece"): pm.install("sentencepiece")
    if not pm.is_installed("PyMuPDF"): pm.install("PyMuPDF")
    if not pm.is_installed("python-dotenv"): pm.install("python-dotenv")
    if not pm.is_installed("aiolimiter"): pm.install("aiolimiter")
    # API Server dependencies
    if not pm.is_installed("fastapi"): pm.install("fastapi")
    if not pm.is_installed("uvicorn"): pm.install("uvicorn[standard]")
    if not pm.is_installed("python-multipart"): pm.install("python-multipart")
except ImportError:
    print("pipmaster is not installed. Please install dependencies manually.")
    exit(1)

# --- Core LightRAG and Dependency Imports ---
from lightrag.utils import setup_logger, EmbeddingFunc, Tokenizer
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.base import DocStatus

# Third-party library imports
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import sentencepiece as spm
import fitz  # PyMuPDF
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter

# --- Logging and Metrics Setup ---
# Load environment variables from a .env file into the environment
load_dotenv()

class ProcessLogFormatter(logging.Formatter):
    """A custom log formatter that includes the process ID for multi-process debugging."""
    def format(self, record):
        record.pid = os.getpid()
        return super().format(record)

# Setup a detailed formatter for log files to trace calls across processes
detailed_formatter = ProcessLogFormatter('%(asctime)s - PID:%(pid)s - %(name)s - %(levelname)s - %(message)s')
# Setup a simpler formatter for console output for better readability
simple_formatter = logging.Formatter('%(levelname)s: %(message)s')

# Configure the root logger for the 'lightrag' namespace
log = logging.getLogger("lightrag")
log.setLevel("INFO")
log.propagate = False
log.handlers = []

# Add the console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(simple_formatter)
log.addHandler(console_handler)

# Add a rotating file handler for persistent logging
try:
    log_file_path = os.getenv("LOG_FILE_PATH", "rag_server.log")
    file_handler = logging.handlers.RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(detailed_formatter)
    log.addHandler(file_handler)
    log.info(f"Logging to file: {log_file_path}")
except Exception as e:
    log.warning(f"Could not set up file logging: {e}")

# --- Global State for RAG and Metrics ---
# This will be initialized on server startup within the lifespan context manager
rag: Optional[LightRAG] = None
# Use a deque for an efficient, thread-safe, size-limited queue to store the latest metrics
llm_call_metrics = deque(maxlen=200)

# --- Configuration via Environment Variables ---
WORKING_DIR = "./rag_storage"
DATA_DIR = "./data" # Directory for uploaded files

# Concurrency and performance settings, loaded from .env with sensible defaults
MAX_GLEANING = int(os.getenv("MAX_GLEANING", 1))
MAX_PARALLEL_INSERT = int(os.getenv("MAX_PARALLEL_INSERT", 2))
MAX_ASYNC_LLM_CALLS = int(os.getenv("MAX_ASYNC_LLM_CALLS", 4))
MAX_ASYNC_EMBEDDING_CALLS = int(os.getenv("MAX_ASYNC_EMBEDDING_CALLS", 8))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", 3))

# Rate limiter for Gemini API to prevent 429 errors.
# The default limit for gemini-1.5-flash is 1500 queries per minute.
gemini_rate_limiter = AsyncLimiter(1500, 60)

# --- Singleton Model Loader ---
class SBERTModel:
    """
    Singleton class to ensure the SentenceTransformer model is loaded only once.
    This prevents race conditions and redundant memory usage in concurrent environments.
    """
    _instance = None
    _lock = threading.Lock()
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking to ensure thread safety
                if cls._instance is None:
                    log.info("Initializing SentenceTransformer model (this happens only once)...")
                    cls._instance = SentenceTransformer('all-MiniLM-L6-v2')
                    log.info("SentenceTransformer model initialized successfully.")
        return cls._instance

# --- Custom Tokenizer ---
class GemmaTokenizer(Tokenizer):
    """Custom tokenizer using Google's Gemma model."""
    @dataclasses.dataclass(frozen=True)
    class _TokenizerConfig:
        tokenizer_model_url: str
        tokenizer_model_hash: str
    _TOKENIZERS = { "google/gemma2": _TokenizerConfig(
            tokenizer_model_url="https://raw.githubusercontent.com/google/gemma_pytorch/33b652c465537c6158f9a472ea5700e5e770ad3f/tokenizer/tokenizer.model",
            tokenizer_model_hash="61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
        ) }
    def __init__(self, tokenizer_dir: str):
        # ... (implementation remains the same) ...
        tokenizer_name = "google/gemma2"
        file_url = self._TOKENIZERS[tokenizer_name].tokenizer_model_url
        tokenizer_model_name = file_url.rsplit("/", 1)[1]
        expected_hash = self._TOKENIZERS[tokenizer_name].tokenizer_model_hash
        file_path = Path(tokenizer_dir) / tokenizer_model_name
        model_data = self._maybe_load_from_cache(file_path=file_path, expected_hash=expected_hash)
        if not model_data:
            model_data = self._load_from_url(file_url=file_url, expected_hash=expected_hash)
            self._save_tokenizer_to_cache(cache_path=file_path, model_data=model_data)
        tokenizer_engine = spm.SentencePieceProcessor()
        tokenizer_engine.LoadFromSerializedProto(model_data)
        super().__init__(model_name="gemma2", tokenizer=tokenizer_engine)
    def _is_valid_model(self, model_data: bytes, expected_hash: str) -> bool: return hashlib.sha256(model_data).hexdigest() == expected_hash
    def _maybe_load_from_cache(self, file_path: Path, expected_hash: str) -> Optional[bytes]:
        if not file_path.is_file(): return None
        with open(file_path, "rb") as f: content = f.read()
        if self._is_valid_model(model_data=content, expected_hash=expected_hash): return content
        self._maybe_remove_file(file_path)
        return None
    def _load_from_url(self, file_url: str, expected_hash: str) -> bytes:
        resp = requests.get(file_url); resp.raise_for_status(); content = resp.content
        if not self._is_valid_model(model_data=content, expected_hash=expected_hash): raise ValueError("Downloaded tokenizer model file is corrupted.")
        return content
    @staticmethod
    def _save_tokenizer_to_cache(cache_path: Path, model_data: bytes) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f: f.write(model_data)
        except OSError: pass
    @staticmethod
    def _maybe_remove_file(file_path: Path) -> None:
        if file_path.is_file():
            try: file_path.unlink()
            except OSError: pass

# --- LLM and Embedding Functions with Metrics ---
async def gemini_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> Union[str, AsyncIterator[str]]:
    """LLM function that calls the Gemini API with detailed call tracking and rate limiting."""
    call_id = uuid.uuid4().hex[:8]
    purpose = kwargs.get('_purpose', 'User Query') # LightRAG provides a purpose
    start_time = time.time()
    
    metric_entry = {
        "call_id": call_id, "pid": os.getpid(), "purpose": purpose,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "duration_sec": -1, "status": "started", "error": None
    }
    llm_call_metrics.append(metric_entry)

    if history_messages is None: history_messages = []
    
    # Enforce the rate limit before making the API call
    async with gemini_rate_limiter:
        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-1.5-flash')
            gemini_history = [{"role": "user" if msg["role"] == "user" else "model", "parts": [{"text": msg["content"]}]} for msg in history_messages]
            full_prompt_parts = [system_prompt] if system_prompt else []
            full_prompt_parts.append(prompt)
            full_prompt = "\n\n".join(full_prompt_parts)
            chat_session_messages = gemini_history + [{"role": "user", "parts": [{"text": full_prompt}]}]
            is_stream = kwargs.get("stream", False)

            log.info(f"[LLM_CALL_START] ID:{call_id} | Purpose: {purpose} | Prompt Length: {len(full_prompt)}")
            metric_entry["status"] = "processing"

            if is_stream:
                response_stream = await model.generate_content_async(contents=chat_session_messages, stream=True, generation_config=genai.types.GenerationConfig(temperature=0.1))
                async def stream_generator():
                    response_text = ""
                    try:
                        async for chunk in response_stream:
                            if chunk.text:
                                response_text += chunk.text; yield chunk.text
                    finally:
                        duration = time.time() - start_time
                        metric_entry.update({"status": "completed", "duration_sec": round(duration, 2)})
                        log.info(f"[LLM_CALL_END] ID:{call_id} | Purpose: {purpose} | Duration: {duration:.2f}s | Response Length: {len(response_text)}")
                return stream_generator()
            else:
                response = await model.generate_content_async(contents=chat_session_messages, generation_config=genai.types.GenerationConfig(temperature=0.1))
                duration = time.time() - start_time
                metric_entry.update({"status": "completed", "duration_sec": round(duration, 2)})
                log.info(f"[LLM_CALL_END] ID:{call_id} | Purpose: {purpose} | Duration: {duration:.2f}s | Response Length: {len(response.text)}")
                return response.text
        except Exception as e:
            duration = time.time() - start_time
            metric_entry.update({"status": "failed", "error": str(e), "duration_sec": round(duration, 2)})
            log.error(f"[LLM_CALL_ERROR] ID:{call_id} | Purpose: {purpose} | Error: {e}", exc_info=True)
            raise

def sbert_embedding_func_sync(texts: List[str]) -> np.ndarray:
    """Synchronous embedding function using a singleton Sentence-Transformers model."""
    model = SBERTModel.get_instance()
    return model.encode(texts, convert_to_numpy=True)

async def sbert_embedding_func(texts: List[str]) -> np.ndarray:
    """Asynchronous wrapper for the synchronous embedding function."""
    return await asyncio.to_thread(sbert_embedding_func_sync, texts)

# --- Document Loading ---
def load_document(file_path: Path) -> Optional[str]:
    """Loads content from a file, supporting .txt, .md, and .pdf formats."""
    try:
        ext = file_path.suffix.lower()
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f: return f.read()
        elif ext == ".pdf":
            with fitz.open(file_path) as doc: return "".join(page.get_text() for page in doc)
        else:
            log.warning(f"Unsupported file format: {ext}. Skipping file: {file_path.name}")
            return None
    except Exception as e:
        log.error(f"Failed to load document {file_path.name}. Error: {e}", exc_info=True)
        return None

# --- RAG System and Ingestion Pipeline ---
async def initialize_rag_system() -> LightRAG:
    """Initializes and configures the LightRAG instance with all components."""
    log.info("Initializing LightRAG system...")
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    tokenizer_path = os.path.join(WORKING_DIR, "gemma_tokenizer")
    custom_tokenizer = GemmaTokenizer(tokenizer_dir=tokenizer_path)

    rag_instance = LightRAG(
        working_dir=WORKING_DIR,
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
        llm_model_func=gemini_llm_func,
        embedding_func=EmbeddingFunc(embedding_dim=384, max_token_size=512, func=sbert_embedding_func),
        tokenizer=custom_tokenizer,
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        llm_model_max_async=MAX_ASYNC_LLM_CALLS,
        embedding_func_max_async=MAX_ASYNC_EMBEDDING_CALLS, # Use env variable
        max_parallel_insert=MAX_PARALLEL_INSERT,
        entity_extract_max_gleaning=MAX_GLEANING,
    )
    await rag_instance.initialize_storages()
    await initialize_pipeline_status()
    log.info("LightRAG system initialized successfully.")
    return rag_instance

async def ingest_documents(data_directory: str):
    """Scans a directory for new/failed documents and ingests them into the RAG system."""
    log.info(f"Starting resumable document ingestion for directory: '{data_directory}'...")
    data_path = Path(data_directory)
    all_files = [p for p in data_path.rglob("*") if p.is_file() and p.suffix.lower() in [".txt", ".md", ".pdf"]]

    if not all_files:
        log.warning(f"No supported documents found in '{data_directory}'.")
        return

    files_to_process: List[Path] = []
    all_doc_ids = [str(file_path.resolve()) for file_path in all_files]
    existing_statuses = await rag.aget_docs_by_ids(all_doc_ids)

    for file_path in all_files:
        doc_id = str(file_path.resolve())
        status_obj = existing_statuses.get(doc_id)
        if not status_obj or status_obj.get('status') != DocStatus.PROCESSED.value:
            files_to_process.append(file_path)

    if not files_to_process:
        log.info("All documents are up-to-date. Ingestion finished.")
        return

    log.info(f"Found {len(files_to_process)} new or failed documents to process.")
    for file_path in files_to_process:
        content = load_document(file_path)
        if content:
            await rag.apipeline_enqueue_documents(input=content, ids=[str(file_path.resolve())], file_paths=[str(file_path.resolve())])
    await rag.apipeline_process_enqueue_documents()
    log.info("Document ingestion cycle has completed.")

# --- FastAPI Server Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    global rag
    log.info("Server starting up...")
    rag = await initialize_rag_system()
    yield
    log.info("Server shutting down...")

app = FastAPI(title="LightRAG API Server", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    query: str
    mode: Literal["local", "global_", "hybrid", "naive", "mix"] = Field("hybrid", alias="mode")
    history: Optional[List[Dict[str, str]]] = []
    response_type: Optional[str] = Field(
        "Multiple Paragraphs",
        description="Defines the desired response format, e.g., 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'."
    )

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accepts a file upload and triggers the ingestion pipeline in the background."""
    try:
        save_path = Path(DATA_DIR) / file.filename
        with open(save_path, "wb") as buffer: buffer.write(await file.read())
        log.info(f"File '{file.filename}' uploaded to {save_path}")
        background_tasks.add_task(ingest_documents, DATA_DIR)
        return {"filename": file.filename, "status": "processing_started"}
    except Exception as e:
        log.error(f"File upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Handles a streaming query to the RAG system."""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")
    try:
        actual_mode = "global" if request.mode == "global_" else request.mode
        
        query_params = QueryParam(
            mode=actual_mode,
            stream=True,
            conversation_history=request.history,
            top_k=5,
            history_turns=HISTORY_TURNS,
            response_type=request.response_type
        )
        response_stream = await rag.aquery(request.query, param=query_params)

        async def stream_generator():
            async for chunk in response_stream:
                yield chunk

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    except Exception as e:
        log.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Returns the most recent LLM call metrics."""
    return {"llm_calls": list(llm_call_metrics)}

@app.get("/status")
async def get_status():
    """Returns the operational status of the server."""
    return {"status": "online", "rag_initialized": rag is not None, "timestamp": datetime.now(timezone.utc).isoformat()}

# --- Main Execution ---
if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("FATAL: GEMINI_API_KEY environment variable is not set.")
    else:
        # To run the server: uvicorn rag_api_server:app --reload --port 8000
        uvicorn.run(app, host="0.0.0.0", port=8000)