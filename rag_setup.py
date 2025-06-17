# FILE: rag_setup.py
import os
import asyncio
import logging
from pathlib import Path
import hashlib
import dataclasses
import requests

# Ensure the logger is configured
from lightrag.utils import setup_logger, EmbeddingFunc, Tokenizer
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag import operate

# --- Dependency Imports ---
import numpy as np
import google.generativeai as genai
import sentencepiece as spm
from aiolimiter import AsyncLimiter  # <-- ADDED: Import for rate limiting

# --- Configuration ---
setup_logger("lightrag", level="INFO")
log = logging.getLogger("lightrag")

# The working directory for LightRAG's cache and other files
WORKING_DIR = "./rag_gemini_neo4j_storage"

# The directory containing your markdown files
DATA_DIR = "./data"

# --- Rate Limiting for Gemini API (NEW) ---
# Gemini 2.0 Flash limits: 15 RPM, 1500 RPD
# We create two limiters to handle these two separate constraints.
gemini_rpm_limiter = AsyncLimiter(15, 60)  # Max 15 requests per 60 seconds
gemini_rpd_limiter = AsyncLimiter(1500, 24 * 60 * 60) # Max 1500 requests per day

# The embedding-001 model has a much higher limit of 1500 QPM (Queries Per Minute)
embedding_qpm_limiter = AsyncLimiter(1500, 60) # Max 1500 requests per 60 seconds

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
    # UPDATED: Added rate limiting wrappers
    # The async with statements will pause execution if a rate limit is exceeded,
    # and resume only when a slot is available in both the RPM and RPD buckets.
    async with gemini_rpm_limiter, gemini_rpd_limiter:
        if history_messages is None:
            history_messages = []
        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-2.0-flash')
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
            log.error(f"Error calling Gemini API: {e}", exc_info=True)
            return "Error: Could not get a response from the LLM."

async def gemini_embedding_func(texts: list[str]) -> np.ndarray:
    # UPDATED: Added rate limiting for the embedding model as well
    async with embedding_qpm_limiter:
        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            task_type = "retrieval_query" if len(texts) == 1 else "retrieval_document"
            
            result = await genai.embed_content_async(
                model="models/embedding-001",
                content=texts,
                task_type=task_type
            )
            return np.array(result['embedding'])
        except Exception as e:
            log.error(f"Error calling Gemini Embedding API: {e}", exc_info=True)
            return np.zeros((len(texts), 768))


# --- Patched RAG System for Correct Caching ---
class PatchedLightRAG(LightRAG):
    """
    Overrides the aquery method to ensure the cache key is unique for
    streaming vs. non-streaming responses. This is a crucial fix for caching.
    """
    async def aquery(self, query: str, param: QueryParam = QueryParam(), system_prompt: str | None = None) -> str | asyncio.StreamReader:
        # Temporarily patch the hashing function to include the stream flag
        original_hasher = operate.compute_args_hash
        
        def patched_hasher(*args, **kwargs):
            # Add the stream flag to the list of arguments being hashed
            return original_hasher(*args, param.stream, **kwargs)

        operate.compute_args_hash = patched_hasher
        
        try:
            # Call the original parent method
            result = await super().aquery(query, param, system_prompt)
        finally:
            # IMPORTANT: Restore the original function to avoid side effects
            operate.compute_args_hash = original_hasher
            
        return result

# --- RAG System Initialization ---
async def initialize_rag_system() -> LightRAG:
    """Initializes the PatchedLightRAG instance."""
    log.info("Initializing LightRAG with Gemini, Neo4j, and cache patch...")
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    tokenizer_path = os.path.join(WORKING_DIR, "gemma_tokenizer")
    custom_tokenizer = GemmaTokenizer(tokenizer_dir=tokenizer_path)
    
    rag = PatchedLightRAG( # Use the patched class
        working_dir=WORKING_DIR,
        graph_storage="Neo4JStorage",
        llm_model_func=gemini_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=gemini_embedding_func,
        ),
        tokenizer=custom_tokenizer,
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    log.info("LightRAG system initialized successfully.")
    return rag

# --- Document Processing ---
async def process_markdown_files(rag_instance: LightRAG, data_directory: str):
    """Recursively finds and ingests all markdown files."""
    log.info(f"Scanning for markdown files in '{data_directory}'...")
    data_path = Path(data_directory)
    md_files = list(data_path.rglob("*.md"))

    if not md_files:
        log.warning(f"No markdown files found in '{data_directory}'.")
        return

    log.info(f"Found {len(md_files)} markdown files to process.")
    
    batch_size = 5
    for i in range(0, len(md_files), batch_size):
        batch_files = md_files[i:i+batch_size]
        contents = []
        ids = []
        for file_path in batch_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    contents.append(content)
                    ids.append(str(file_path))
            except Exception as e:
                log.error(f"Failed to read file {file_path}: {e}")
        
        if contents:
            log.info(f"Inserting batch of {len(contents)} documents...")
            await rag_instance.ainsert(contents, ids=ids)
    
    log.info("Document ingestion complete.")