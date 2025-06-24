# FILE: rag_setup.py
import os
import asyncio
import logging
from pathlib import Path
import hashlib
import dataclasses
import requests
from datetime import datetime, timezone

# Import load_dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Load environment variables from .env file at the beginning
load_dotenv()

# Ensure the logger is configured
from lightrag.utils import setup_logger, EmbeddingFunc, Tokenizer, compute_args_hash
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.base import DocStatus
from lightrag import operate
# Import the prompt module to access templates directly
from lightrag import prompt as prompt_template_module

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

# --- Rate Limiting for Gemini API ---
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
    if history_messages is None:
        history_messages = []
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
        raise

async def gemini_embedding_func(texts: list[str]) -> np.ndarray:
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
            log.error(f"FATAL ERROR in Gemini Embedding API call: {e}", exc_info=True)
            raise

# --- RAG System Initialization ---
async def initialize_rag_system() -> LightRAG:
    log.info("Initializing LightRAG with Gemini and Neo4j...")
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    tokenizer_path = os.path.join(WORKING_DIR, "gemma_tokenizer")
    custom_tokenizer = GemmaTokenizer(tokenizer_dir=tokenizer_path)

    max_async_calls = int(os.getenv("MAX_ASYNC", 1))
    max_parallel_files = int(os.getenv("MAX_PARALLEL_INSERT", 1))

    log.info(f"Configuring LightRAG with MAX_ASYNC={max_async_calls} and MAX_PARALLEL_INSERT={max_parallel_files}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        graph_storage="Neo4JStorage",
        llm_model_func=gemini_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=2048,
            func=gemini_embedding_func,
        ),
        tokenizer=custom_tokenizer,
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        llm_model_max_async=max_async_calls,
        max_parallel_insert=max_parallel_files
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    log.info("LightRAG system initialized successfully.")
    return rag

# --- Helper for Cache Key Generation ---
def _get_extraction_prompt_for_chunk(rag_instance: LightRAG, chunk_content: str) -> str:
    """
    Meticulously reconstructs the exact prompt string that LightRAG uses for entity extraction.
    This is critical for generating the correct cache key.
    This logic is directly adapted from `lightrag.operate.extract_entities`.
    """
    prompts = prompt_template_module.PROMPTS
    addon_params = rag_instance.addon_params
    language = addon_params.get("language", prompts["DEFAULT_LANGUAGE"])
    entity_types = addon_params.get("entity_types", prompts["DEFAULT_ENTITY_TYPES"])
    examples = "\n".join(prompts["entity_extraction_examples"])

    context_base = dict(
        tuple_delimiter=prompts["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=prompts["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=prompts["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    examples = examples.format(**context_base)

    final_context = dict(
        **context_base,
        examples=examples,
        input_text=chunk_content
    )
    
    return prompts["entity_extraction"].format(**final_context)

# --- CORRECTED: Resumable Document Processing ---
async def process_markdown_files(rag_instance: LightRAG, data_directory: str):
    """
    Finds and ingests markdown files with chunk-level resume capability.
    It pre-chunks files and checks the LLM cache to only process necessary chunks.
    """
    log.info(f"Starting resumable ingestion process for directory '{data_directory}'...")
    data_path = Path(data_directory)
    all_files = list(data_path.rglob("*.md"))
    
    if not all_files:
        log.warning(f"No markdown files found in '{data_directory}'. Ingestion finished.")
        return

    log.info(f"Found {len(all_files)} markdown files to check.")
    
    try:
        all_doc_ids = [str(file_path) for file_path in all_files]
        existing_statuses = await rag_instance.aget_docs_by_ids(all_doc_ids)

        files_to_process = []
        for file_path in all_files:
            doc_id = str(file_path)
            status_obj = existing_statuses.get(doc_id)
            if status_obj and status_obj.get('status') == DocStatus.PROCESSED.value:
                log.info(f"SKIPPED: Document '{file_path.name}' is already fully processed.")
            else:
                files_to_process.append(file_path)

        if not files_to_process:
            log.info("All documents are already processed. Ingestion finished.")
            return

        extraction_cache = await rag_instance.llm_response_cache.get_by_id("default") or {}

        for file_path in files_to_process:
            log.info(f"--- Analyzing file for processing: '{file_path.name}' ---")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    log.warning(f"File '{file_path.name}' is empty. Skipping.")
                    continue

                all_chunks = rag_instance.chunking_func(
                    rag_instance.tokenizer,
                    content,
                    None,
                    False,
                    rag_instance.chunk_overlap_token_size,
                    rag_instance.chunk_token_size,
                )
                
                total_chunks = len(all_chunks)
                log.info(f"File '{file_path.name}' has {total_chunks} total chunks.")

                chunks_to_process_content = []
                for i, chunk in enumerate(all_chunks):
                    chunk_content = chunk['content']
                    full_prompt_for_chunk = _get_extraction_prompt_for_chunk(rag_instance, chunk_content)
                    cache_key = compute_args_hash(full_prompt_for_chunk)
                    
                    if cache_key not in extraction_cache:
                        chunks_to_process_content.append(chunk_content)

                cached_chunk_count = total_chunks - len(chunks_to_process_content)
                log.info(f"Cache check for '{file_path.name}': {cached_chunk_count} of {total_chunks} chunks are already cached.")

                if chunks_to_process_content:
                    log.info(f"Enqueuing {len(chunks_to_process_content)} new chunks for processing...")
                    # THE FIX: By setting `ids=None`, we let LightRAG generate a unique
                    # hash ID for each chunk's content, preventing the "IDs must be unique" error.
                    await rag_instance.apipeline_enqueue_documents(
                        input=chunks_to_process_content,
                        file_paths=[file_path.name] * len(chunks_to_process_content),
                        ids=None # Let LightRAG auto-generate unique IDs for each chunk.
                    )
                    await rag_instance.apipeline_process_enqueue_documents()
                
                log.info(f"Marking master document '{file_path.name}' as PROCESSED.")
                await rag_instance.doc_status.upsert({
                    str(file_path): {
                        "status": DocStatus.PROCESSED,
                        "content": content[:5000], # Store a snippet to avoid huge DB records
                        "content_summary": content[:250] + "...",
                        "content_length": len(content),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": file_path.name,
                        "chunks_count": total_chunks,
                    }
                })
                log.info(f"--- Finished processing for file: '{file_path.name}' ---")

            except Exception as file_error:
                log.error(f"Failed to process file '{file_path.name}'. Error: {file_error}", exc_info=True)
                await rag_instance.doc_status.upsert({
                    str(file_path): { "status": DocStatus.FAILED, "error": str(file_error) }
                })

        log.info("All files have been analyzed and processed. Ingestion cycle complete.")

    except Exception as e:
        log.error(f"A critical error occurred during the ingestion pipeline. Reason: {e}", exc_info=True)
        return
    