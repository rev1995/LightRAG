# FILE: rag_setup.py
import os
import asyncio
import logging
from pathlib import Path
import hashlib
import dataclasses
import requests

# Import load_dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Load environment variables from .env file at the beginning
load_dotenv()

# Ensure the logger is configured
from lightrag.utils import setup_logger, EmbeddingFunc, Tokenizer
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.base import DocStatus
from lightrag import operate

# --- Dependency Imports ---
import numpy as np
import google.generativeai as genai
import sentencepiece as spm
from aiolimiter import AsyncLimiter

# --- Configuration ---
setup_logger("lightrag", level="INFO")
log = logging.getLogger("lightrag")

# The working directory for LightRAG's cache and other files
WORKING_DIR = "./rag_gemini_neo4j_storage"

# The directory containing your files
DATA_DIR = "./data"

# --- Rate Limiting for Gemini API (Good practice, especially for embeddings) ---
# This limiter primarily acts as a safety for the embedding function, which can process in batches.
# The main LLM rate limiting is handled by the MAX_ASYNC=1 setting.
embedding_qpm_limiter = AsyncLimiter(1500, 60)

# --- Custom Tokenizer (avoids tiktoken dependency) ---
class GemmaTokenizer(Tokenizer):
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

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.LoadFromSerializedProto(model_data)
        super().__init__(model_name="gemma2", tokenizer=tokenizer)

    def _is_valid_model(self, model_data: bytes, expected_hash: str) -> bool:
        return hashlib.sha256(model_data).hexdigest() == expected_hash

    def _maybe_load_from_cache(self, file_path: Path, expected_hash: str) -> bytes | None:
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
            raise ValueError("Downloaded model file is corrupted.")
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
async def gemini_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    # The user's original limiters are removed as MAX_ASYNC=1 is the primary control now.
    if history_messages is None:
        history_messages = []
    # The try...except block now re-raises the exception to be caught by the main pipeline
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_history = [{"role": "user" if msg["role"] == "user" else "model", "parts": [{"text": msg["content"]}]} for msg in history_messages]

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
        # Re-raise the exception so the pipeline knows to fail
        raise

async def gemini_embedding_func(texts: list[str]) -> np.ndarray:
    async with embedding_qpm_limiter:
        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            task_type = "retrieval_query" if len(texts) == 1 else "retrieval_document"
            # Using embed_content for batching efficiency
            result = await genai.embed_content_async(
                model="models/embedding-001",
                content=texts,
                task_type=task_type
            )
            # The result for a batch is in 'embedding', which is a list of lists.
            return np.array(result['embedding'])
        except Exception as e:
            log.error(f"FATAL ERROR in Gemini Embedding API call: {e}", exc_info=True)
            # Re-raise the exception
            raise

# --- Patched RAG System for Correct Caching ---
class PatchedLightRAG(LightRAG):
    async def aquery(self, query: str, param: QueryParam = QueryParam(), system_prompt: str | None = None) -> str | asyncio.StreamReader:
        original_hasher = operate.compute_args_hash
        def patched_hasher(*args, **kwargs):
            return original_hasher(*args, param.stream, **kwargs)

        operate.compute_args_hash = patched_hasher
        try:
            result = await super().aquery(query, param, system_prompt)
        finally:
            operate.compute_args_hash = original_hasher
        return result

# --- RAG System Initialization ---
async def initialize_rag_system() -> LightRAG:
    log.info("Initializing LightRAG with Gemini, Neo4j, and cache patch...")
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    tokenizer_path = os.path.join(WORKING_DIR, "gemma_tokenizer")
    custom_tokenizer = GemmaTokenizer(tokenizer_dir=tokenizer_path)

    # Fetch concurrency settings from environment variables, with safe defaults.
    max_async_calls = int(os.getenv("MAX_ASYNC", 1))
    max_parallel_files = int(os.getenv("MAX_PARALLEL_INSERT", 1))

    log.info(f"Configuring LightRAG with MAX_ASYNC={max_async_calls} and MAX_PARALLEL_INSERT={max_parallel_files}")

    rag = PatchedLightRAG(
        working_dir=WORKING_DIR,
        graph_storage="Neo4JStorage",
        llm_model_func=gemini_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=2048, # The embedding model has a 2048 token limit
            func=gemini_embedding_func,
        ),
        tokenizer=custom_tokenizer,
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        # Set the concurrency parameters fetched from the environment
        llm_model_max_async=max_async_calls,
        max_parallel_insert=max_parallel_files
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    log.info("LightRAG system initialized successfully.")
    return rag

# --- Document Processing with Batching and Fail-Fast Error Handling ---
async def process_markdown_files(rag_instance: LightRAG, data_directory: str):
    """
    Finds and ingests markdown files in batches. Skips already processed files,
    retries failed files, and stops the entire process on any unrecoverable error.
    """
    log.info(f"Starting ingestion process for directory '{data_directory}'...")
    data_path = Path(data_directory)
    all_files = list(data_path.rglob("*.md"))
    
    if not all_files:
        log.warning(f"No markdown files found in '{data_directory}'. Ingestion finished.")
        return

    log.info(f"Found {len(all_files)} markdown files to check.")

    try:
        # 1. Determine which files actually need processing
        all_doc_ids = [str(file_path) for file_path in all_files]
        existing_statuses = await rag_instance.aget_docs_by_ids(all_doc_ids)

        files_to_process = []
        for file_path in all_files:
            doc_id = str(file_path)
            status_obj = existing_statuses.get(doc_id)
            if status_obj and status_obj.get('status') == DocStatus.PROCESSED.value:
                log.info(f"SKIPPED: Document '{file_path.name}' is already processed.")
            else:
                if status_obj and status_obj.get('status') == DocStatus.FAILED.value:
                    error_reason = status_obj.get('error', 'Unknown error')
                    log.warning(f"RETRYING: Document '{file_path.name}' previously failed. Reason: {error_reason}")
                files_to_process.append(file_path)

        if not files_to_process:
            log.info("All found documents are already processed. Ingestion finished.")
            return
        
        # 2. Process the necessary files in batches
        batch_size = rag_instance.max_parallel_insert
        log.info(f"Processing {len(files_to_process)} new/failed files in batches of {batch_size}.")

        for i in range(0, len(files_to_process), batch_size):
            current_batch_paths = files_to_process[i:i + batch_size]
            log.info(f"--- Starting Batch {i//batch_size + 1} ({len(current_batch_paths)} files) ---")

            contents_to_process = []
            file_paths_to_process = []
            doc_ids_to_process = []
            for file_path in current_batch_paths:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if content.strip():
                        contents_to_process.append(content)
                        file_paths_to_process.append(file_path.name)
                        doc_ids_to_process.append(str(file_path))
                    else:
                        log.warning(f"SKIPPED in batch: File '{file_path.name}' is empty.")
                except Exception as e:
                    log.error(f"FAILED TO READ in batch: Could not read file '{file_path.name}'. Reason: {e}")
            
            if not contents_to_process:
                log.warning(f"Batch {i//batch_size + 1} has no valid files to process. Skipping.")
                continue

            # Enqueue and process the current batch
            await rag_instance.apipeline_enqueue_documents(
                input=contents_to_process,
                file_paths=file_paths_to_process,
                ids=doc_ids_to_process
            )
            await rag_instance.apipeline_process_enqueue_documents()
            log.info(f"--- Finished Batch {i//batch_size + 1} ---")

        log.info("All batches processed successfully.")

    except Exception as e:
        # This is the "fail-fast" logic. Any unhandled exception from the pipeline will be caught here.
        log.error(f"A critical error occurred during the ingestion pipeline, and the process has been stopped. Reason: {e}", exc_info=True)
        # We explicitly stop here by returning. The background task will terminate.
        return
