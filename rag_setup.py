import os
import asyncio
import logging
from pathlib import Path
import hashlib
import dataclasses
import requests
from datetime import datetime, timezone
from typing import List, Optional, Union, AsyncIterator, Dict, Any

# --- Dependency Management ---
# Automatically install required packages if they are not present.
try:
    import pipmaster as pm
    if not pm.is_installed("google-generativeai"): pm.install("google-generativeai")
    if not pm.is_installed("sentence-transformers"): pm.install("sentence-transformers")
    if not pm.is_installed("sentencepiece"): pm.install("sentencepiece")
    if not pm.is_installed("PyMuPDF"): pm.install("PyMuPDF")
    if not pm.is_installed("python-dotenv"): pm.install("python-dotenv")
    if not pm.is_installed("aiolimiter"): pm.install("aiolimiter")
except ImportError:
    print("pipmaster is not installed. Please install dependencies manually:")
    print("pip install google-generativeai sentence-transformers sentencepiece PyMuPDF python-dotenv aiolimiter")
    exit(1)

# --- Core LightRAG and Dependency Imports ---
# Import from the local lightrag source directory
from lightrag.utils import setup_logger, EmbeddingFunc, Tokenizer, compute_args_hash
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.base import DocStatus
from lightrag import operate
from lightrag import prompt as prompt_template_module # Import the prompt module to access templates directly

# Third-party library imports
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import sentencepiece as spm
import fitz  # PyMuPDF
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter

# --- Initial Configuration ---
# Load environment variables from the .env file
load_dotenv()

# Set up a logger for the application
setup_logger("lightrag", level="INFO")
log = logging.getLogger("lightrag")

# Define working directories
WORKING_DIR = "./rag_storage"
DATA_DIR = "./data"

# --- Configuration via Environment Variables ---
# Number of times to attempt LLM gleaning for entity extraction
MAX_GLEANING = int(os.getenv("MAX_GLEANING", 1))
# Maximum number of concurrent file processing operations
MAX_PARALLEL_INSERT = int(os.getenv("MAX_PARALLEL_INSERT", 2))
# Maximum number of concurrent LLM API calls
MAX_ASYNC_LLM_CALLS = int(os.getenv("MAX_ASYNC_LLM_CALLS", 4))

# --- Rate Limiter for Gemini API ---
# Models like gemini-1.5-flash have a limit of 1500 queries per minute.
embedding_qpm_limiter = AsyncLimiter(1500, 60)

# --- Custom Tokenizer (avoids tiktoken dependency) ---
class GemmaTokenizer(Tokenizer):
    """
    A custom tokenizer that uses Google's Gemma tokenizer model.
    This class handles downloading the model, caching it locally, and providing
    the standard encode/decode interface required by LightRAG.
    """
    @dataclasses.dataclass(frozen=True)
    class _TokenizerConfig:
        tokenizer_model_url: str
        tokenizer_model_hash: str

    _TOKENIZERS = {
        "google/gemma2": _TokenizerConfig(
            tokenizer_model_url="https://raw.githubusercontent.com/google/gemma_pytorch/33b652c465537c6158f9a472ea5700e5e770ad3f/tokenizer/tokenizer.model",
            tokenizer_model_hash="61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
        ),
    }

    def __init__(self, tokenizer_dir: str):
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

    def _is_valid_model(self, model_data: bytes, expected_hash: str) -> bool:
        return hashlib.sha256(model_data).hexdigest() == expected_hash

    def _maybe_load_from_cache(self, file_path: Path, expected_hash: str) -> Optional[bytes]:
        if not file_path.is_file():
            return None
        with open(file_path, "rb") as f:
            content = f.read()
        if self._is_valid_model(model_data=content, expected_hash=expected_hash):
            return content
        self._maybe_remove_file(file_path)
        return None

    def _load_from_url(self, file_url: str, expected_hash: str) -> bytes:
        resp = requests.get(file_url)
        resp.raise_for_status()
        content = resp.content
        if not self._is_valid_model(model_data=content, expected_hash=expected_hash):
            raise ValueError("Downloaded tokenizer model file is corrupted.")
        return content

    @staticmethod
    def _save_tokenizer_to_cache(cache_path: Path, model_data: bytes) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(model_data)
        except OSError:
            pass

    @staticmethod
    def _maybe_remove_file(file_path: Path) -> None:
        if file_path.is_file():
            try:
                file_path.unlink()
            except OSError:
                pass

# --- LLM and Embedding Functions ---
async def gemini_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> Union[str, AsyncIterator[str]]:
    """
    LLM function that calls the Gemini API.
    Handles standard and streaming responses, and integrates with LightRAG's history and system prompts.
    """
    if history_messages is None:
        history_messages = []
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Format messages for Gemini API
        gemini_history = [{"role": "user" if msg["role"] == "user" else "model", "parts": [{"text": msg["content"]}]} for msg in history_messages]
        
        # Combine system prompt with user prompt
        full_prompt_parts = []
        if system_prompt:
            full_prompt_parts.append(system_prompt)
        full_prompt_parts.append(prompt)
        full_prompt = "\n\n".join(full_prompt_parts)

        chat_session_messages = gemini_history + [{"role": "user", "parts": [{"text": full_prompt}]}]
        
        is_stream = kwargs.get("stream", False)
        if is_stream:
            response_stream = await model.generate_content_async(
                contents=chat_session_messages,
                stream=True,
                generation_config=genai.types.GenerationConfig(temperature=0.1)
            )
            async def stream_generator():
                async for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
            return stream_generator()
        else:
            response = await model.generate_content_async(
                contents=chat_session_messages,
                generation_config=genai.types.GenerationConfig(temperature=0.1)
            )
            return response.text
            
    except Exception as e:
        log.error(f"FATAL ERROR in Gemini API call: {e}", exc_info=True)
        raise

def sbert_embedding_func_sync(texts: List[str]) -> np.ndarray:
    """
    Synchronous embedding function using Sentence-Transformers.
    This is wrapped in an async function for compatibility with LightRAG's async pipeline.
    """
    # Use SentenceTransformer for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts, convert_to_numpy=True)

async def sbert_embedding_func(texts: List[str]) -> np.ndarray:
    """
    Asynchronous wrapper for the synchronous Sentence-Transformer embedding function.
    Runs the synchronous function in a separate thread to avoid blocking the event loop.
    """
    # Use asyncio.to_thread to run the synchronous, CPU-bound function in a separate thread
    return await asyncio.to_thread(sbert_embedding_func_sync, texts)

# --- Document Loading ---
def load_document(file_path: Path) -> Optional[str]:
    """
    Loads content from a file, supporting .txt, .md, and .pdf formats.
    Handles potential file reading errors.
    """
    try:
        ext = file_path.suffix.lower()
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            with fitz.open(file_path) as doc:
                text = "".join(page.get_text() for page in doc)
                return text
        else:
            log.warning(f"Unsupported file format: {ext}. Skipping file: {file_path.name}")
            return None
    except FileNotFoundError:
        log.error(f"File not found: {file_path}. Skipping.")
        return None
    except Exception as e:
        log.error(f"Failed to load document {file_path.name}. Error: {e}", exc_info=True)
        return None

# --- RAG System Initialization ---
async def initialize_rag_system() -> LightRAG:
    """
    Initializes and configures the LightRAG instance with all necessary components.
    Sets up LLM, embeddings, tokenizer, caching, concurrency limits, and storage.
    """
    log.info("Initializing LightRAG system with Gemini, Sentence-Transformers, and file-based storage...")
    
    # Ensure working directories exist
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
        log.info(f"Created working directory: {WORKING_DIR}")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        log.warning(f"Data directory '{DATA_DIR}' not found. Please add your documents there.")
    
    # Initialize the custom Gemma tokenizer
    tokenizer_path = os.path.join(WORKING_DIR, "gemma_tokenizer")
    custom_tokenizer = GemmaTokenizer(tokenizer_dir=tokenizer_path)

    # Configure concurrency limits from environment variables
    max_async_calls = int(os.getenv("MAX_ASYNC_LLM_CALLS", 4)) # Default to 4 concurrent LLM calls
    max_parallel_files = int(os.getenv("MAX_PARALLEL_INSERT", 2)) # Default to 2 concurrent file ingestions

    log.info(f"Configuring LightRAG with: MAX_ASYNC_LLM_CALLS={max_async_calls}, MAX_PARALLEL_INSERT={max_parallel_files}")

    # Configure and create the LightRAG instance
    rag = LightRAG(
        working_dir=WORKING_DIR,
        # Default file-based storages for simplicity and local execution
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
        
        # LLM and Embedding configurations
        llm_model_func=gemini_llm_func, # Using Gemini for LLM
        embedding_func=EmbeddingFunc(
            embedding_dim=384,  # `all-MiniLM-L6-v2` has a dimension of 384
            max_token_size=512, # Model's context window, adjusted for embedding limits
            func=sbert_embedding_func, # Using Sentence-Transformer for embeddings
        ),
        # Use the custom Gemma tokenizer
        tokenizer=custom_tokenizer,
        
        # Enable caching to save on API calls and processing time
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        
        # Concurrency settings
        llm_model_max_async=max_async_calls,
        max_parallel_insert=max_parallel_files,
        
        # Entity extraction gleaning count (number of additional LLM calls per chunk for better extraction)
        entity_extract_max_gleaning=MAX_GLEANING,
    )

    # Initialize the configured storage components (e.g., create files, connect to DBs)
    await rag.initialize_storages()
    # Initialize pipeline status tracking (used for resumable processing)
    await initialize_pipeline_status()
    log.info("LightRAG system initialized successfully.")
    return rag

# --- Document Ingestion Pipeline ---
async def ingest_documents(rag: LightRAG, data_directory: str):
    """
    Scans a directory for documents (.txt, .md, .pdf), checks their processing status,
    and ingests any new or failed documents into the RAG system.
    This process is resumable: it skips already processed files and retries failed ones.
    """
    log.info(f"Starting resumable document ingestion for directory: '{data_directory}'...")
    
    data_path = Path(data_directory)
    # Recursively find all supported document files
    all_files = [p for p in data_path.rglob("*") if p.is_file() and p.suffix.lower() in [".txt", ".md", ".pdf"]]

    if not all_files:
        log.warning(f"No supported documents (.txt, .md, .pdf) found in '{data_directory}'. Ingestion finished.")
        return

    log.info(f"Found {len(all_files)} files to check.")
    
    # Track which files need processing
    files_to_process: List[Path] = []
    # Fetch existing statuses for all files in one batch operation
    try:
        # Use resolved paths to ensure consistency across different systems/mounts
        all_doc_ids = [str(file_path.resolve()) for file_path in all_files]
        existing_statuses = await rag.aget_docs_by_ids(all_doc_ids)
    except Exception as e:
        log.error(f"Failed to fetch existing document statuses: {e}. Will attempt to process all files.", exc_info=True)
        existing_statuses = {} # Treat as if no files are processed if status fetch fails

    # Determine which files need processing based on status
    for file_path in all_files:
        doc_id = str(file_path.resolve())
        status_obj = existing_statuses.get(doc_id)
        
        if status_obj and status_obj.get('status') == DocStatus.PROCESSED.value:
            log.info(f"SKIPPED: Document '{file_path.name}' is already processed.")
        else:
            # Include files that are PENDING, FAILED, or not found in status
            files_to_process.append(file_path)
            if status_obj and status_obj.get('status') == DocStatus.FAILED.value:
                log.warning(f"RETRYING: Document '{file_path.name}' failed previously.")

    if not files_to_process:
        log.info("All documents are up-to-date or skipped. Ingestion finished.")
        return

    log.info(f"Found {len(files_to_process)} documents to process.")
    
    # Process files in batches, respecting max_parallel_insert
    try:
        # Use LightRAG's internal pipeline for managed processing
        # The pipeline handles chunking, embedding, KG extraction, and status updates.
        # It also manages concurrency based on `max_parallel_insert`.
        # We need to iterate through files and enqueue them.
        for i, file_path in enumerate(files_to_process):
            log.info(f"--- Enqueuing file for processing: '{file_path.name}' ({i+1}/{len(files_to_process)}) ---")
            
            content = load_document(file_path)
            if not content:
                # If file content couldn't be loaded, mark as FAILED and skip
                await rag.doc_status.upsert({
                    str(file_path.resolve()): {
                        "status": DocStatus.FAILED,
                        "error": "Failed to load document content",
                        "content": "",
                        "content_summary": "",
                        "content_length": 0,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": str(file_path.resolve()),
                    }
                })
                continue

            # Enqueue the document with its file path for status tracking
            await rag.apipeline_enqueue_documents(
                input=content,
                ids=[str(file_path.resolve())],  # Use resolved path as unique ID
                file_paths=[str(file_path.resolve())] # Store file path for later reference
            )
            # Process the enqueued documents. This call might block if concurrency limits are hit.
            # LightRAG's internal pipeline manages the batching and LLM calls based on its config.
            await rag.apipeline_process_enqueue_documents()
            
            # After processing a file, log its status. LightRAG's pipeline handles
            # updating the status to PROCESSED or FAILED internally. We don't need to
            # explicitly set PROCESSED here unless we want to override the pipeline's final status.
            # However, for resumability, the pipeline's status updates are crucial.

        log.info("Document ingestion cycle has completed. Check logs for file-specific processing statuses.")

    except Exception as e:
        log.error(f"A critical error occurred during the ingestion pipeline. Reason: {e}", exc_info=True)
        # Mark all remaining files to be processed as FAILED if a critical error occurred
        # Note: This might be too broad; ideally, catch errors per file within the loop.
        # For simplicity here, we catch top-level errors.
        for file_path in files_to_process:
            doc_id = str(file_path.resolve())
            current_status = await rag.doc_status.get_by_id(doc_id)
            if current_status and current_status.get('status') != DocStatus.PROCESSED.value:
                await rag.doc_status.upsert({
                    doc_id: {
                        "status": DocStatus.FAILED,
                        "error": f"Pipeline failed: {str(e)}",
                        "content": current_status.get("content", ""),
                        "content_summary": current_status.get("content_summary", ""),
                        "content_length": current_status.get("content_length", 0),
                        "created_at": current_status.get("created_at"),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": current_status.get("file_path", str(file_path.resolve())),
                        "chunks_count": current_status.get("chunks_count", -1),
                    }
                })


# --- Interactive Query Loop ---
async def main_query_loop(rag: LightRAG):
    """
    An interactive command-line loop for querying the RAG system.
    It handles user input, query parameter configuration, and displays streaming or single-line responses.
    Manages conversation history for context.
    """
    print("\n\n===================================")
    print(" LightRAG Interactive Query Mode ")
    print("===================================")
    print("Enter your query below. Type 'exit' or 'quit' to end.")
    
    conversation_history = [] # Stores the history of user and assistant messages
    
    while True:
        try:
            query_text = input("\nYour Query: ")
            if query_text.lower() in ["exit", "quit"]:
                print("Exiting query mode. Goodbye!")
                break

            # Create QueryParam object to configure the query
            query_params = QueryParam(
                mode="hybrid",  # Use a mix of local (chunk) and global (KG) context
                stream=True,    # Enable streaming for a better user experience
                # Pass conversation history for context
                conversation_history=conversation_history, 
                # Set top_k for retrieval (adjust as needed)
                top_k=5 
            )
            
            print("\nLightRAG's Answer:")
            print("-------------------")

            # Execute the query asynchronously
            # `aquery` returns an async generator for streaming responses
            response_stream = await rag.aquery(query_text, param=query_params)
            
            full_response = ""
            # Iterate through the stream and print output
            async for chunk in response_stream:
                print(chunk, end="", flush=True)
                full_response += chunk
            
            # Update conversation history for context in next turns
            conversation_history.append({"role": "user", "content": query_text})
            conversation_history.append({"role": "assistant", "content": full_response})
            print("\n-------------------")

        except Exception as e:
            log.error(f"An error occurred during query: {e}", exc_info=True)
            print("\nSorry, an error occurred. Please try again.")

# --- Main Execution Function ---
async def main():
    """
    The main function that orchestrates the entire RAG pipeline:
    1. Initializes the LightRAG system.
    2. Ingests documents from the data directory, processing only new or failed ones.
    3. Starts an interactive query loop for user interaction.
    """
    # 1. Initialize the RAG system
    rag = await initialize_rag_system()

    # 2. Ingest documents from the data directory (resumable)
    await ingest_documents(rag, DATA_DIR)

    # 3. Start the interactive query loop
    await main_query_loop(rag)

# --- Script Entry Point ---
if __name__ == "__main__":
    # Check for Gemini API Key before starting
    if not os.getenv("GEMINI_API_KEY"):
        print("FATAL: GEMINI_API_KEY environment variable is not set.")
        print("Please create a .env file in the root directory and add:")
        print("GEMINI_API_KEY='YOUR_API_KEY_HERE'")
    else:
        try:
            # Run the main asynchronous pipeline
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Exiting.")
        except Exception as e:
            log.error(f"An unexpected error occurred in the main execution block: {e}", exc_info=True)