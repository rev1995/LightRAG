#!/usr/bin/env python3
"""
Production-Ready LightRAG Pipeline with Gemini LLM

This is a comprehensive RAG pipeline that includes:
- Gemini LLM integration with token tracking
- Efficient text chunking with Gemini tokenizer
- Advanced caching with multiple modes
- Reranker integration with mix mode as default
- Query parameter controls
- Data isolation between instances
- Multimodal document processing (RAG-Anything)
- Environment-based configuration
- Production-ready logging and error handling
"""

import os
import asyncio
import logging
import logging.config
import numpy as np
import nest_asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import hashlib
import requests
import sentencepiece as spm

# LightRAG imports
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, setup_logger, TokenTracker, Tokenizer
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.rerank import custom_rerank, RerankModel
from lightrag.llm_rerank import LLMReranker, AdaptiveLLMReranker, create_llm_reranker
from llm_rerank_robust import create_robust_llm_reranker

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# Load environment variables
load_dotenv()


class GemmaTokenizer(Tokenizer):
    """Gemini-compatible tokenizer using Gemma models"""
    
    @dataclass(frozen=True)
    class _TokenizerConfig:
        tokenizer_model_url: str
        tokenizer_model_hash: str

    _TOKENIZERS = {
        "google/gemma2": _TokenizerConfig(
            tokenizer_model_url="https://raw.githubusercontent.com/google/gemma_pytorch/33b652c465537c6158f9a472ea5700e5e770ad3f/tokenizer/tokenizer.model",
            tokenizer_model_hash="61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
        ),
        "google/gemma3": _TokenizerConfig(
            tokenizer_model_url="https://raw.githubusercontent.com/google/gemma_pytorch/cb7c0152a369e43908e769eb09e1ce6043afe084/tokenizer/gemma3_cleaned_262144_v2.spiece.model",
            tokenizer_model_hash="1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        ),
    }

    def __init__(
        self, model_name: str = "gemini-2.0-flash", tokenizer_dir: Optional[str] = None
    ):
        # https://github.com/google/gemma_pytorch/tree/main/tokenizer
        if "1.5" in model_name or "1.0" in model_name:
            # up to gemini 1.5 gemma2 is a comparable local tokenizer
            # https://github.com/googleapis/python-aiplatform/blob/main/vertexai/tokenization/_tokenizer_loading.py
            tokenizer_name = "google/gemma2"
        else:
            # for gemini > 2.0 gemma3 was used
            tokenizer_name = "google/gemma3"

        file_url = self._TOKENIZERS[tokenizer_name].tokenizer_model_url
        tokenizer_model_name = file_url.rsplit("/", 1)[1]
        expected_hash = self._TOKENIZERS[tokenizer_name].tokenizer_model_hash

        tokenizer_dir_path = Path(tokenizer_dir) if tokenizer_dir else Path("./tokenizer_cache")
        if tokenizer_dir_path.is_dir():
            file_path = tokenizer_dir_path / tokenizer_model_name
            model_data = self._maybe_load_from_cache(
                file_path=file_path, expected_hash=expected_hash
            )
        else:
            model_data = None
        if not model_data:
            model_data = self._load_from_url(
                file_url=file_url, expected_hash=expected_hash
            )
            self.save_tokenizer_to_cache(cache_path=file_path, model_data=model_data)

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.LoadFromSerializedProto(model_data)
        super().__init__(model_name=model_name, tokenizer=tokenizer)

    def _is_valid_model(self, model_data: bytes, expected_hash: str) -> bool:
        """Returns true if the content is valid by checking the hash."""
        return hashlib.sha256(model_data).hexdigest() == expected_hash

    def _maybe_load_from_cache(self, file_path: Path, expected_hash: str) -> Optional[bytes]:
        """Loads the model data from the cache path."""
        if not file_path.is_file():
            return None
        with open(file_path, "rb") as f:
            content = f.read()
        if self._is_valid_model(model_data=content, expected_hash=expected_hash):
            return content

        # Cached file corrupted.
        self._maybe_remove_file(file_path)
        return None

    def _load_from_url(self, file_url: str, expected_hash: str) -> bytes:
        """Loads model bytes from the given file url."""
        resp = requests.get(file_url)
        resp.raise_for_status()
        content = resp.content

        if not self._is_valid_model(model_data=content, expected_hash=expected_hash):
            actual_hash = hashlib.sha256(content).hexdigest()
            raise ValueError(
                f"Downloaded model file is corrupted."
                f" Expected hash {expected_hash}. Got file hash {actual_hash}."
            )
        return content

    @staticmethod
    def save_tokenizer_to_cache(cache_path: Path, model_data: bytes) -> None:
        """Saves the model data to the cache path."""
        try:
            if not cache_path.is_file():
                cache_dir = cache_path.parent
                cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(model_data)
        except OSError:
            # Don't raise if we cannot write file.
            pass

    @staticmethod
    def _maybe_remove_file(file_path: Path) -> None:
        """Removes the file if exists."""
        if not file_path.is_file():
            return
        try:
            file_path.unlink()
        except OSError:
            # Don't raise if we cannot remove file.
            pass


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline"""
    
    # Environment variables - all required
    GEMINI_API_KEY: str = None
    WORKING_DIR: str = None
    WORKSPACE: str = None
    LOG_LEVEL: str = "INFO"
    VERBOSE: bool = False
    
    # LLM Configuration
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_MAX_OUTPUT_TOKENS: int = 5000
    LLM_TEMPERATURE: float = 0.1
    LLM_TOP_K: int = 10
    
    # Embedding Configuration (Gemini Native)
    EMBEDDING_MODEL: str = "text-embedding-004"
    EMBEDDING_DIM: int = 768
    EMBEDDING_MAX_TOKEN_SIZE: int = 512
    
    # Reranker Configuration
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_BINDING_HOST: str = None
    RERANK_BINDING_API_KEY: str = None
    ENABLE_RERANK: bool = True
    RERANK_STRATEGY: str = "semantic_scoring"  # semantic_scoring, relevance_ranking, hybrid
    RERANK_BATCH_SIZE: int = 5
    RERANK_MAX_CONCURRENT: int = 3
    RERANK_CACHE_ENABLED: bool = True
    
    # Chunking Configuration
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP_SIZE: int = 100
    
    # Cache Configuration
    ENABLE_LLM_CACHE: bool = True
    ENABLE_LLM_CACHE_FOR_ENTITY_EXTRACT: bool = True
    ENABLE_EMBEDDING_CACHE: bool = True
    EMBEDDING_CACHE_SIMILARITY_THRESHOLD: float = 0.90
    
    # Query Configuration
    DEFAULT_QUERY_MODE: str = "mix"  # Default to mix mode when reranker is enabled
    TOP_K: int = 40
    CHUNK_TOP_K: int = 10
    MAX_ENTITY_TOKENS: int = 10000
    MAX_RELATION_TOKENS: int = 10000
    MAX_TOTAL_TOKENS: int = 32000
    HISTORY_TURNS: int = 3
    
    # Processing Configuration
    MAX_ASYNC: int = 4
    MAX_PARALLEL_INSERT: int = 2
    ENTITY_EXTRACT_MAX_GLEANING: int = 1
    
    # Tokenizer Configuration
    TOKENIZER_DIR: str = "./tokenizer_cache"
    
    def __post_init__(self):
        """Validate and set default values from environment"""
        # Required environment variables
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Working directory and workspace
        self.WORKING_DIR = os.getenv("WORKING_DIR", "./rag_storage")
        self.WORKSPACE = os.getenv("WORKSPACE", "")
        
        # Logging configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
        
        # LLM configuration
        self.LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        self.LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "5000"))
        self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.LLM_TOP_K = int(os.getenv("LLM_TOP_K", "10"))
        
        # Embedding configuration (Gemini Native)
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
        self.EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
        self.EMBEDDING_MAX_TOKEN_SIZE = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "512"))
        
        # Reranker configuration
        self.RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        self.RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")
        self.RERANK_BINDING_API_KEY = os.getenv("RERANK_BINDING_API_KEY")
        self.ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
        self.RERANK_STRATEGY = os.getenv("RERANK_STRATEGY", "semantic_scoring")
        self.RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "5"))
        self.RERANK_MAX_CONCURRENT = int(os.getenv("RERANK_MAX_CONCURRENT", "3"))
        self.RERANK_CACHE_ENABLED = os.getenv("RERANK_CACHE_ENABLED", "true").lower() == "true"
        
        # Chunking configuration
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
        self.CHUNK_OVERLAP_SIZE = int(os.getenv("CHUNK_OVERLAP_SIZE", "100"))
        
        # Cache configuration
        self.ENABLE_LLM_CACHE = os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true"
        self.ENABLE_LLM_CACHE_FOR_ENTITY_EXTRACT = os.getenv("ENABLE_LLM_CACHE_FOR_ENTITY_EXTRACT", "true").lower() == "true"
        self.ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
        self.EMBEDDING_CACHE_SIMILARITY_THRESHOLD = float(os.getenv("EMBEDDING_CACHE_SIMILARITY_THRESHOLD", "0.90"))
        
        # Query configuration
        self.DEFAULT_QUERY_MODE = os.getenv("DEFAULT_QUERY_MODE", "mix" if self.ENABLE_RERANK else "hybrid")
        self.TOP_K = int(os.getenv("TOP_K", "40"))
        self.CHUNK_TOP_K = int(os.getenv("CHUNK_TOP_K", "10"))
        self.MAX_ENTITY_TOKENS = int(os.getenv("MAX_ENTITY_TOKENS", "10000"))
        self.MAX_RELATION_TOKENS = int(os.getenv("MAX_RELATION_TOKENS", "10000"))
        self.MAX_TOTAL_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS", "32000"))
        self.HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "3"))
        
        # Processing configuration
        self.MAX_ASYNC = int(os.getenv("MAX_ASYNC", "4"))
        self.MAX_PARALLEL_INSERT = int(os.getenv("MAX_PARALLEL_INSERT", "2"))
        self.ENTITY_EXTRACT_MAX_GLEANING = int(os.getenv("ENTITY_EXTRACT_MAX_GLEANING", "1"))
        
        # Tokenizer configuration
        self.TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "./tokenizer_cache")


class ProductionRAGPipeline:
    """Production-ready RAG pipeline with all advanced features"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.token_tracker = TokenTracker()
        self.rag = None
        self._setup_logging()
        self._setup_gemini_tokenizer()
        
    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        log_dir = os.getenv("LOG_DIR", os.getcwd())
        log_file_path = os.path.abspath(os.path.join(log_dir, "production_rag.log"))
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # Get log configuration from environment
        log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
        log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups
        
        logging.config.dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": self.config.LOG_LEVEL,
                    "propagate": False,
                },
                "production_rag": {
                    "handlers": ["console", "file"],
                    "level": self.config.LOG_LEVEL,
                    "propagate": False,
                },
            },
        })
        
        self.logger = logging.getLogger("production_rag")
        setup_logger("lightrag", level=self.config.LOG_LEVEL)
        
        if self.config.VERBOSE:
            from lightrag.utils import set_verbose_debug
            set_verbose_debug(True)
    
    def _setup_gemini_tokenizer(self):
        """Setup Gemini tokenizer for efficient chunking"""
        try:
            # Use GemmaTokenizer for Gemini tokenization
            self.tokenizer = GemmaTokenizer(
                model_name=self.config.LLM_MODEL,
                tokenizer_dir=self.config.TOKENIZER_DIR
            )
            self.logger.info("GemmaTokenizer initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize GemmaTokenizer: {e}")
            self.tokenizer = None
    
    async def _llm_model_func(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        history_messages: List[Dict[str, str]] = [], 
        keyword_extraction: bool = False, 
        **kwargs
    ) -> str:
        """Gemini LLM model function with token tracking"""
        try:
            # Initialize Gemini client
            client = genai.Client(api_key=self.config.GEMINI_API_KEY)
            
            # Combine prompts
            if history_messages is None:
                history_messages = []
            
            combined_prompt = ""
            if system_prompt:
                combined_prompt += f"{system_prompt}\n"
            
            for msg in history_messages:
                combined_prompt += f"{msg['role']}: {msg['content']}\n"
            
            combined_prompt += f"user: {prompt}"
            
            # Call Gemini model
            response = client.models.generate_content(
                model=self.config.LLM_MODEL,
                contents=[combined_prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=self.config.LLM_MAX_OUTPUT_TOKENS,
                    temperature=self.config.LLM_TEMPERATURE,
                    top_k=self.config.LLM_TOP_K,
                ),
            )
            
            # Track token usage
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)
            
            token_counts = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            
            self.token_tracker.add_usage(token_counts)
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error in LLM model function: {e}")
            raise
    
    async def _embedding_func(self, texts: List[str]) -> np.ndarray:
        """Gemini native embedding function"""
        try:
            client = genai.Client(api_key=self.config.GEMINI_API_KEY)
            
            embeddings = []
            for text in texts:
                response = client.models.embed_content(
                    model=self.config.EMBEDDING_MODEL,
                    contents=text
                )
                # Extract the actual embedding values from ContentEmbedding object
                embedding_values = response.embeddings[0].values
                embeddings.append(embedding_values)
            
            return np.array(embeddings)
        except Exception as e:
            self.logger.error(f"Error in Gemini embedding function: {e}")
            raise
    
    async def _rerank_func(self, query: str, documents: List[Dict], top_k: int = None, **kwargs):
        """Robust LLM-based rerank function with better error handling"""
        if not self.config.ENABLE_RERANK:
            return documents[:top_k] if top_k else documents
        
        try:
            # Use robust LLM-based reranking
            robust_reranker = create_robust_llm_reranker(
                llm_func=self._llm_model_func,
                max_retries=3,
                retry_delay=2.0,  # Increased delay for rate limiting
                max_concurrent=self.config.RERANK_MAX_CONCURRENT,
                batch_size=self.config.RERANK_BATCH_SIZE,
                timeout=45.0,  # Increased timeout
                enable_cache=self.config.RERANK_CACHE_ENABLED,
                strategy=self.config.RERANK_STRATEGY
            )
            
            return await robust_reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k or self.config.CHUNK_TOP_K
            )
        except Exception as e:
            self.logger.warning(f"Robust LLM rerank failed, returning original documents: {e}")
            return documents[:top_k] if top_k else documents
    
    async def initialize(self):
        """Initialize the RAG pipeline"""
        try:
            self.logger.info("Initializing production RAG pipeline...")
            
            # Create working directory if it doesn't exist
            os.makedirs(self.config.WORKING_DIR, exist_ok=True)
            
            # Initialize LightRAG with all configurations
            self.rag = LightRAG(
                working_dir=self.config.WORKING_DIR,
                workspace=self.config.WORKSPACE,
                entity_extract_max_gleaning=self.config.ENTITY_EXTRACT_MAX_GLEANING,
                enable_llm_cache=self.config.ENABLE_LLM_CACHE,
                enable_llm_cache_for_entity_extract=self.config.ENABLE_LLM_CACHE_FOR_ENTITY_EXTRACT,
                embedding_cache_config={
                    "enabled": self.config.ENABLE_EMBEDDING_CACHE,
                    "similarity_threshold": self.config.EMBEDDING_CACHE_SIMILARITY_THRESHOLD
                },
                tokenizer=GemmaTokenizer(
                    tokenizer_dir=self.config.TOKENIZER_DIR,
                    model_name=self.config.LLM_MODEL,
                ),
                llm_model_func=self._llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.config.EMBEDDING_DIM,
                    max_token_size=8192,
                    func=self._embedding_func,
                ),
                rerank_model_func=self._rerank_func,
            )
            
            # Initialize storages
            await self.rag.initialize_storages()
            await initialize_pipeline_status()
            
            self.logger.info("Production RAG pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    async def insert_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Insert documents into the RAG pipeline"""
        try:
            self.logger.info(f"Inserting {len(documents)} documents...")
            
            results = []
            for i, doc in enumerate(documents):
                try:
                    result = await self.rag.ainsert(doc)
                    results.append({"index": i, "status": "success", "result": result})
                    self.logger.debug(f"Document {i+1}/{len(documents)} inserted successfully")
                except Exception as e:
                    results.append({"index": i, "status": "error", "error": str(e)})
                    self.logger.error(f"Failed to insert document {i+1}: {e}")
            
            success_count = len([r for r in results if r["status"] == "success"])
            self.logger.info(f"Inserted {success_count}/{len(documents)} documents successfully")
            
            return {
                "total_documents": len(documents),
                "successful_insertions": success_count,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error inserting documents: {e}")
            raise
    
    async def query(
        self, 
        query: str, 
        mode: str = None,
        user_prompt: str = None,
        top_k: int = None,
        chunk_top_k: int = None,
        enable_rerank: bool = None,
        response_type: str = "Multiple Paragraphs",
        conversation_history: List[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Query the RAG pipeline with comprehensive parameters"""
        try:
            # Use default mode if reranker is enabled
            if mode is None:
                mode = self.config.DEFAULT_QUERY_MODE
            
            # Use default values if not provided
            if top_k is None:
                top_k = self.config.TOP_K
            if chunk_top_k is None:
                chunk_top_k = self.config.CHUNK_TOP_K
            if enable_rerank is None:
                enable_rerank = self.config.ENABLE_RERANK
            
            # Create query parameters
            query_param = QueryParam(
                mode=mode,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                enable_rerank=enable_rerank,
                response_type=response_type,
                conversation_history=conversation_history or [],
                history_turns=self.config.HISTORY_TURNS,
                max_entity_tokens=self.config.MAX_ENTITY_TOKENS,
                max_relation_tokens=self.config.MAX_RELATION_TOKENS,
                max_total_tokens=self.config.MAX_TOTAL_TOKENS,
                user_prompt=user_prompt,
                **kwargs
            )
            
            self.logger.info(f"Querying with mode: {mode}, top_k: {top_k}, enable_rerank: {enable_rerank}")
            
            # Execute query with token tracking
            with self.token_tracker:
                response = await self.rag.aquery(query, param=query_param)
            
            # Get token usage statistics
            token_stats = self.token_tracker.get_usage()
            
            return {
                "response": response,
                "query_mode": mode,
                "token_usage": token_stats,
                "query_params": {
                    "top_k": top_k,
                    "chunk_top_k": chunk_top_k,
                    "enable_rerank": enable_rerank,
                    "response_type": response_type,
                    "user_prompt": user_prompt
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in query: {e}")
            raise
    
    async def clear_cache(self, modes: List[str] = None) -> Dict[str, Any]:
        """Clear LLM response cache with different modes"""
        try:
            self.logger.info(f"Clearing cache for modes: {modes or 'all'}")
            
            # Convert custom mode names to LightRAG valid modes
            mode_mapping = {
                "query": "default",
                "entity_extract": "default", 
                "relation_extract": "default",
                "summary": "default"
            }
            
            if modes:
                # Map custom modes to LightRAG modes
                lightrag_modes = []
                for mode in modes:
                    if mode in mode_mapping:
                        lightrag_modes.append(mode_mapping[mode])
                    else:
                        # If it's already a valid LightRAG mode, use it directly
                        lightrag_modes.append(mode)
                
                # Remove duplicates
                lightrag_modes = list(set(lightrag_modes))
                await self.rag.aclear_cache(lightrag_modes)
            else:
                # Clear all caches
                await self.rag.aclear_cache()
            
            return {
                "status": "success",
                "message": f"Cache cleared for modes: {modes or 'all'}",
                "modes_cleared": modes or "all"
            }
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return {
                "status": "error",
                "message": str(e),
                "modes_cleared": []
            }
    
    async def get_token_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive token usage statistics"""
        try:
            stats = self.token_tracker.get_usage()
            total_requests = stats.get("call_count", 0)
            total_tokens = stats.get("total_tokens", 0)
            prompt_tokens = stats.get("prompt_tokens", 0)
            completion_tokens = stats.get("completion_tokens", 0)
            
            # Calculate average tokens per request
            average_tokens_per_request = (
                total_tokens / total_requests if total_requests > 0 else 0
            )
            
            # Simple cost estimation (you can adjust rates as needed)
            # Assuming $0.001 per 1K tokens for input and $0.002 per 1K tokens for output
            cost_estimation = (
                (prompt_tokens * 0.001 / 1000) + 
                (completion_tokens * 0.002 / 1000)
            )
            
            return {
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "average_tokens_per_request": average_tokens_per_request,
                "cost_estimation": cost_estimation
            }
        except Exception as e:
            self.logger.error(f"Error getting token usage stats: {e}")
            return {"error": str(e)}
    
    async def finalize(self):
        """Clean up resources"""
        try:
            if self.rag:
                await self.rag.finalize_storages()
            self.logger.info("Production RAG pipeline finalized successfully")
        except Exception as e:
            self.logger.error(f"Error finalizing RAG pipeline: {e}")


async def main():
    """Main function demonstrating the production RAG pipeline"""
    try:
        # Initialize configuration
        config = RAGConfig()
        
        # Create and initialize the pipeline
        pipeline = ProductionRAGPipeline(config)
        await pipeline.initialize()
        
        # Example usage
        print("🚀 Production RAG Pipeline Demo")
        print("=" * 50)
        
        # Show tokenizer information
        print(f"📝 Using GemmaTokenizer for model: {config.LLM_MODEL}")
        print(f"📁 Tokenizer cache directory: {config.TOKENIZER_DIR}")
        print()
        
        # Insert sample documents
        sample_docs = [
            "LightRAG is a powerful retrieval-augmented generation system that combines knowledge graphs with vector search.",
            "The system supports multiple query modes including naive, local, global, hybrid, mix, and bypass.",
            "Reranking improves retrieval quality by re-ordering documents based on relevance to the query.",
            "Token tracking helps monitor usage and costs across different LLM providers.",
            "Caching mechanisms improve performance by storing frequently used embeddings and LLM responses.",
            "GemmaTokenizer provides efficient text chunking optimized for Gemini models."
        ]
        
        print("\n📄 Inserting sample documents...")
        insert_result = await pipeline.insert_documents(sample_docs)
        print(f"Inserted {insert_result['successful_insertions']}/{insert_result['total_documents']} documents")
        
        # Example queries with different modes
        queries = [
            {
                "query": "What is LightRAG and how does it work?",
                "mode": "hybrid",
                "user_prompt": "Provide a comprehensive explanation with examples."
            },
            {
                "query": "Explain the different query modes available",
                "mode": "mix",
                "user_prompt": "Create a comparison table of the modes."
            },
            {
                "query": "How does reranking improve retrieval?",
                "mode": "local",
                "user_prompt": "Focus on practical benefits and implementation details."
            }
        ]
        
        for i, query_info in enumerate(queries, 1):
            print(f"\n🔍 Query {i}: {query_info['query']}")
            print(f"Mode: {query_info['mode']}")
            
            result = await pipeline.query(
                query=query_info['query'],
                mode=query_info['mode'],
                user_prompt=query_info['user_prompt']
            )
            
            print(f"Response: {result['response'][:200]}...")
            print(f"Token Usage: {result['token_usage']}")
        
        # Get token usage statistics
        print("\n📊 Token Usage Statistics:")
        token_stats = await pipeline.get_token_usage_stats()
        for key, value in token_stats.items():
            print(f"  {key}: {value}")
        
        # Clear cache example
        print("\n🧹 Clearing cache...")
        cache_result = await pipeline.clear_cache(["query", "entity_extract"])
        print(f"Cache clear result: {cache_result}")
        
        # Finalize
        await pipeline.finalize()
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        logging.error(f"Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 