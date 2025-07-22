#!/usr/bin/env python3
"""
Production RAG Pipeline for LightRAG

A comprehensive, production-ready RAG pipeline that integrates Gemini models,
token tracking, and advanced caching for optimal performance.
"""

import os
import asyncio
import logging
import json
import numpy as np
import hashlib
import dataclasses
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Callable, Tuple, AsyncIterator
from pathlib import Path
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from google import genai
from google.genai import types

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, TokenTracker, setup_logger, Tokenizer
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.types import KnowledgeGraph
from lightrag.base import DocStatus

try:
    import sentencepiece as spm
except ImportError:
    raise ImportError(
        "sentencepiece is not installed. Please install it with `pip install sentencepiece` to use GemmaTokenizer."
    )

# Load environment variables
load_dotenv()

# Configure logging
setup_logger("lightrag", level="INFO")
logger = logging.getLogger(__name__)


class GemmaTokenizer(Tokenizer):
    """A Tokenizer implementation using the SentencePiece library for Gemini models."""

    @dataclasses.dataclass(frozen=True)
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
        """Initialize the GemmaTokenizer with a specified model name and tokenizer directory.
        
        Args:
            model_name: The model name for the tokenizer. Defaults to "gemini-2.0-flash".
            tokenizer_dir: Directory to store the tokenizer model. If None, uses the current directory.
        """
        # Select appropriate tokenizer based on model version
        if "1.5" in model_name or "1.0" in model_name:
            # up to gemini 1.5 gemma2 is a comparable local tokenizer
            tokenizer_name = "google/gemma2"
        else:
            # for gemini > 2.0 gemma3 was used
            tokenizer_name = "google/gemma3"

        file_url = self._TOKENIZERS[tokenizer_name].tokenizer_model_url
        tokenizer_model_name = file_url.rsplit("/", 1)[1]
        expected_hash = self._TOKENIZERS[tokenizer_name].tokenizer_model_hash

        # Use working directory if tokenizer_dir is not provided
        if tokenizer_dir is None:
            tokenizer_dir = os.path.join(os.getcwd(), "tokenizer_cache")
        
        tokenizer_dir = Path(tokenizer_dir)
        file_path = tokenizer_dir / tokenizer_model_name
        
        # Try to load from cache first
        model_data = self._maybe_load_from_cache(
            file_path=file_path, expected_hash=expected_hash
        )
        
        # If not in cache, download from URL
        if not model_data:
            model_data = self._load_from_url(
                file_url=file_url, expected_hash=expected_hash
            )
            self.save_tokenizer_to_cache(cache_path=file_path, model_data=model_data)

        # Initialize the SentencePiece tokenizer
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.LoadFromSerializedProto(model_data)
        super().__init__(model_name=model_name, tokenizer=tokenizer)

    def _is_valid_model(self, model_data: bytes, expected_hash: str) -> bool:
        """Returns true if the content is valid by checking the hash."""
        return hashlib.sha256(model_data).hexdigest() == expected_hash

    def _maybe_load_from_cache(self, file_path: Path, expected_hash: str) -> bytes:
        """Loads the model data from the cache path."""
        if not file_path.is_file():
            return None
        with open(file_path, "rb") as f:
            content = f.read()
        if self._is_valid_model(model_data=content, expected_hash=expected_hash):
            return content

        # Cached file corrupted
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
                f"Downloaded model file is corrupted. "
                f"Expected hash {expected_hash}. Got file hash {actual_hash}."
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
            # Don't raise if we cannot write file
            pass

    @staticmethod
    def _maybe_remove_file(file_path: Path) -> None:
        """Removes the file if exists."""
        if not file_path.is_file():
            return
        try:
            file_path.unlink()
        except OSError:
            # Don't raise if we cannot remove file
            pass

@dataclass
class RAGConfig:
    """Configuration for the Production RAG Pipeline"""
    # Directory settings
    working_dir: str = field(default_factory=lambda: os.getenv("WORKING_DIR", "./rag_storage"))
    input_dir: str = field(default_factory=lambda: os.getenv("INPUT_DIR", "./inputs"))
    
    # LLM settings
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    gemini_max_output_tokens: int = field(default_factory=lambda: int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "5000")))
    gemini_temperature: float = field(default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", "0.1")))
    gemini_top_k: int = field(default_factory=lambda: int(os.getenv("GEMINI_TOP_K", "10")))
    
    # Tokenizer settings
    use_gemma_tokenizer: bool = field(default_factory=lambda: os.getenv("USE_GEMMA_TOKENIZER", "True").lower() == "true")
    tokenizer_dir: str = field(default_factory=lambda: os.getenv("TOKENIZER_DIR", "./tokenizer_cache"))
    
    # Embedding settings
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "models/embedding-001"))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768")))
    embedding_max_token_size: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "8192")))
    
    # Reranker settings
    enable_rerank: bool = field(default_factory=lambda: os.getenv("ENABLE_RERANK", "True").lower() == "true")
    
    # Cache settings
    enable_llm_cache: bool = field(default_factory=lambda: os.getenv("ENABLE_LLM_CACHE", "True").lower() == "true")
    enable_llm_cache_for_entity_extract: bool = field(default_factory=lambda: os.getenv("ENABLE_LLM_CACHE_FOR_ENTITY_EXTRACT", "True").lower() == "true")
    embedding_cache_enabled: bool = field(default_factory=lambda: os.getenv("EMBEDDING_CACHE_ENABLED", "True").lower() == "true")
    embedding_cache_similarity_threshold: float = field(default_factory=lambda: float(os.getenv("EMBEDDING_CACHE_SIMILARITY_THRESHOLD", "0.95")))
    
    # Query parameters
    default_mode: str = field(default_factory=lambda: os.getenv("DEFAULT_MODE", "mix"))
    default_top_k: int = field(default_factory=lambda: int(os.getenv("DEFAULT_TOP_K", "5")))
    default_chunk_top_k: int = field(default_factory=lambda: int(os.getenv("DEFAULT_CHUNK_TOP_K", "10")))
    default_max_entity_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_ENTITY_TOKENS", "2000")))
    default_max_relation_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_RELATION_TOKENS", "2000")))
    default_max_total_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_TOTAL_TOKENS", "6000")))
    history_turns: int = field(default_factory=lambda: int(os.getenv("HISTORY_TURNS", "3")))
    
    # Entity extraction settings
    entity_extract_max_gleaning: int = field(default_factory=lambda: int(os.getenv("MAX_GLEANING", "1")))
    
    # Storage settings
    kv_storage: str = field(default_factory=lambda: os.getenv("KV_STORAGE", "JsonKVStorage"))
    vector_storage: str = field(default_factory=lambda: os.getenv("VECTOR_STORAGE", "NanoVectorDBStorage"))
    graph_storage: str = field(default_factory=lambda: os.getenv("GRAPH_STORAGE", "NetworkXStorage"))
    doc_status_storage: str = field(default_factory=lambda: os.getenv("DOC_STATUS_STORAGE", "JsonDocStatusStorage"))
    
    # Workspace for data isolation
    workspace: str = field(default_factory=lambda: os.getenv("WORKSPACE", ""))
    
    # Concurrency settings
    max_async_llm_calls: int = field(default_factory=lambda: int(os.getenv("MAX_ASYNC", "4")))
    max_async_embedding_calls: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "8")))
    max_parallel_insert: int = field(default_factory=lambda: int(os.getenv("MAX_PARALLEL_INSERT", "2")))
    
    # Logging settings
    log_file_path: str = field(default_factory=lambda: os.getenv("LOG_FILE_PATH", "./rag_server.log"))


class ProductionRAGPipeline:
    """Production-ready RAG Pipeline with Gemini integration"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the Production RAG Pipeline
        
        Args:
            config: Configuration for the pipeline. If None, default config will be used.
        """
        self.config = config or RAGConfig()
        self.token_tracker = TokenTracker()
        self.rag = None
        self.initialized = False
        
        # Create directories if they don't exist
        os.makedirs(self.config.working_dir, exist_ok=True)
        os.makedirs(self.config.input_dir, exist_ok=True)
        
    async def initialize(self):
        """Initialize the RAG pipeline"""
        if self.initialized:
            return
        
        logger.info("Initializing Production RAG Pipeline...")
        
        # Configure logging to file if specified
        if self.config.log_file_path:
            log_handler = logging.FileHandler(self.config.log_file_path)
            log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(log_handler)
        
        # Initialize LightRAG instance
        
        # Create tokenizer based on configuration
        tokenizer = None
        if self.config.use_gemma_tokenizer:
            logger.info("Using GemmaTokenizer for tokenization")
            tokenizer = GemmaTokenizer(
                model_name=self.config.gemini_model,
                tokenizer_dir=self.config.tokenizer_dir
            )
        
        self.rag = LightRAG(
            working_dir=self.config.working_dir,
            kv_storage=self.config.kv_storage,
            vector_storage=self.config.vector_storage,
            graph_storage=self.config.graph_storage,
            doc_status_storage=self.config.doc_status_storage,
            workspace=self.config.workspace,
            tokenizer=tokenizer,  # Use our custom tokenizer
            llm_model_func=self._llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.config.embedding_dim,
                max_token_size=self.config.embedding_max_token_size,
                func=self._embedding_func,
                max_async=self.config.max_async_embedding_calls,
            ),
            rerank_model_func=self._rerank_model_func if self.config.enable_rerank else None,
            entity_extract_max_gleaning=self.config.entity_extract_max_gleaning,
            enable_llm_cache=self.config.enable_llm_cache,
            enable_llm_cache_for_entity_extract=self.config.enable_llm_cache_for_entity_extract,
            embedding_cache_config={
                "enabled": self.config.embedding_cache_enabled,
                "similarity_threshold": self.config.embedding_cache_similarity_threshold,
            },
            llm_model_max_async=self.config.max_async_llm_calls,
            max_parallel_insert=self.config.max_parallel_insert,
        )
        
        # Initialize storages and pipeline status
        await self.rag.initialize_storages()
        await initialize_pipeline_status()
        
        self.initialized = True
        logger.info("Production RAG Pipeline initialized successfully")
    
    async def _llm_model_func(self, prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
        """LLM model function for Gemini
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            history_messages: Optional conversation history
            keyword_extraction: Whether this is for keyword extraction
            **kwargs: Additional arguments
            
        Returns:
            str: The model response
        """
        if not self.config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set. Cannot use LLM.")
        
        client = genai.Client(api_key=self.config.gemini_api_key)
        
        # Combine prompts
        if history_messages is None:
            history_messages = []
        
        combined_prompt = ""
        if system_prompt:
            combined_prompt += f"{system_prompt}\n"
        
        for msg in history_messages:
            combined_prompt += f"{msg['role']}: {msg['content']}\n"
        
        combined_prompt += f"user: {prompt}"
        
        # Call the Gemini model
        response = client.models.generate_content(
            model=self.config.gemini_model,
            contents=[combined_prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=self.config.gemini_max_output_tokens,
                temperature=self.config.gemini_temperature,
                top_k=self.config.gemini_top_k,
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
    
    async def _embedding_func(self, texts: List[str]) -> np.ndarray:
        """Embedding function for Gemini
        
        Args:
            texts: List of texts to embed
            
        Returns:
            np.ndarray: The embeddings
        """
        if not self.config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set. Cannot use embeddings.")
        
        client = genai.Client(api_key=self.config.gemini_api_key)
        
        # Process texts in batches to avoid rate limits
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                try:
                    # Using the gemini-embedding-001 model
                    result = client.models.embed_content(
                        model=self.config.embedding_model,
                        text=text,
                        task_type="RETRIEVAL_DOCUMENT",  # Optimize for document retrieval
                    )
                    
                    # Extract the embedding values from the response
                    embedding_values = result.embeddings[0].values
                    embedding = np.array(embedding_values)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Error embedding text with Gemini: {e}")
                    # Create a zero embedding as fallback
                    embedding = np.zeros(self.config.embedding_dim)
                    batch_embeddings.append(embedding)
            
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    async def _rerank_model_func(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """Reranker function using Gemini LLM
        
        Args:
            query: The query text
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List[Tuple[int, float]]: List of (document_index, score) tuples
        """
        try:
            from lightrag.rerank import gemini_llm_rerank
            
            # Convert string documents to dict format for reranking
            doc_dicts = [{"content": doc} for doc in documents]
            
            # Use Gemini LLM for reranking
            reranked_docs = await gemini_llm_rerank(
                query=query,
                documents=doc_dicts,
                top_k=top_k,
                api_key=self.config.gemini_api_key,
                model=self.config.gemini_model,
                temperature=self.config.gemini_temperature,
            )
            
            # Extract indices and scores
            results = []
            for doc in reranked_docs:
                # Find the original index by matching content
                content = doc.get("content")
                if content in documents:
                    idx = documents.index(content)
                    score = doc.get("rerank_score", 5.0)
                    results.append((idx, float(score)))
            
            return results
        except Exception as e:
            logger.error(f"Error in Gemini LLM reranking: {e}")
            # Return original order with dummy scores if reranking fails
            return [(i, 1.0 - i/len(documents)) for i in range(min(top_k, len(documents)))]
    
    async def query(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """Query the RAG system
        
        Args:
            query_text: The query text
            **kwargs: Additional query parameters
            
        Returns:
            Dict[str, Any]: Query response with token usage
        """
        if not self.initialized:
            await self.initialize()
        
        # Prepare query parameters
        mode = kwargs.get("mode", self.config.default_mode)
        top_k = kwargs.get("top_k", self.config.default_top_k)
        chunk_top_k = kwargs.get("chunk_top_k", self.config.default_chunk_top_k)
        max_entity_tokens = kwargs.get("max_entity_tokens", self.config.default_max_entity_tokens)
        max_relation_tokens = kwargs.get("max_relation_tokens", self.config.default_max_relation_tokens)
        max_total_tokens = kwargs.get("max_total_tokens", self.config.default_max_total_tokens)
        enable_rerank = kwargs.get("enable_rerank", self.config.enable_rerank)
        history_turns = kwargs.get("history_turns", self.config.history_turns)
        
        # Create query parameters
        param = QueryParam(
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            max_entity_tokens=max_entity_tokens,
            max_relation_tokens=max_relation_tokens,
            max_total_tokens=max_total_tokens,
            response_type=kwargs.get("response_type"),
            conversation_history=kwargs.get("conversation_history"),
            only_need_context=kwargs.get("only_need_context", False),
            only_need_prompt=kwargs.get("only_need_prompt", False),
            user_prompt=kwargs.get("user_prompt"),
            enable_rerank=enable_rerank,
            history_turns=history_turns,
        )
        
        # Reset token tracker for this query
        self.token_tracker.reset()
        
        # Execute query with token tracking
        with self.token_tracker:
            response = await self.rag.query(query_text, param=param)
        
        # Prepare response
        result = {
            "response": response,
            "query_mode": mode,
            "token_usage": self.token_tracker.get_usage(),
            "query_params": {
                "mode": mode,
                "top_k": top_k,
                "chunk_top_k": chunk_top_k,
                "max_entity_tokens": max_entity_tokens,
                "max_relation_tokens": max_relation_tokens,
                "max_total_tokens": max_total_tokens,
                "enable_rerank": enable_rerank,
            }
        }
        
        return result
    
    async def query_stream(self, query_text: str, **kwargs) -> AsyncIterator[str]:
        """Stream query results from the RAG system
        
        Args:
            query_text: The query text
            **kwargs: Additional query parameters
            
        Yields:
            str: Chunks of the response
        """
        if not self.initialized:
            await self.initialize()
        
        # Prepare query parameters (same as query method)
        mode = kwargs.get("mode", self.config.default_mode)
        top_k = kwargs.get("top_k", self.config.default_top_k)
        chunk_top_k = kwargs.get("chunk_top_k", self.config.default_chunk_top_k)
        max_entity_tokens = kwargs.get("max_entity_tokens", self.config.default_max_entity_tokens)
        max_relation_tokens = kwargs.get("max_relation_tokens", self.config.default_max_relation_tokens)
        max_total_tokens = kwargs.get("max_total_tokens", self.config.default_max_total_tokens)
        enable_rerank = kwargs.get("enable_rerank", self.config.enable_rerank)
        history_turns = kwargs.get("history_turns", self.config.history_turns)
        
        param = QueryParam(
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            max_entity_tokens=max_entity_tokens,
            max_relation_tokens=max_relation_tokens,
            max_total_tokens=max_total_tokens,
            response_type=kwargs.get("response_type"),
            conversation_history=kwargs.get("conversation_history"),
            only_need_context=kwargs.get("only_need_context", False),
            only_need_prompt=kwargs.get("only_need_prompt", False),
            user_prompt=kwargs.get("user_prompt"),
            enable_rerank=enable_rerank,
            history_turns=history_turns,
        )
        
        # Reset token tracker
        self.token_tracker.reset()
        
        # Execute streaming query
        async for chunk in self.rag.query_stream(query_text, param=param):
            yield chunk
    
    async def insert_text(self, text: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Insert text into the RAG system
        
        Args:
            text: The text to insert
            doc_id: Optional document ID
            
        Returns:
            Dict[str, Any]: Insertion result
        """
        if not self.initialized:
            await self.initialize()
        
        result = self.rag.insert(text, ids=doc_id)
        return {"status": "success", "doc_id": result}
    
    async def insert_file(self, file_path: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Insert a file into the RAG system
        
        Args:
            file_path: Path to the file
            doc_id: Optional document ID
            
        Returns:
            Dict[str, Any]: Insertion result
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Properly await the async insert operation
            result = await self.rag.ainsert(text, ids=doc_id, file_paths=file_path)
            return {"status": "success", "doc_id": result}
        except Exception as e:
            logger.error(f"Error inserting file {file_path}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_document_status(self, doc_id: str) -> Dict[str, Any]:
        """Get document processing status
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dict[str, Any]: Document status
        """
        if not self.initialized:
            await self.initialize()
        
        status = await self.rag.get_doc_status(doc_id)
        if status:
            # Convert DocStatus to dict and format datetime
            status_dict = asdict(status)
            if status_dict.get("created_at"):
                status_dict["created_at"] = status_dict["created_at"].isoformat()
            if status_dict.get("updated_at"):
                status_dict["updated_at"] = status_dict["updated_at"].isoformat()
            return status_dict
        return {"status": "not_found"}
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the system
        
        Returns:
            List[Dict[str, Any]]: List of document statuses
        """
        if not self.initialized:
            await self.initialize()
        
        statuses = await self.rag.list_doc_statuses()
        
        # Convert DocStatus objects to dicts and format datetime if needed
        result = []
        for status in statuses:
            status_dict = asdict(status)
            if status_dict.get("created_at") and not isinstance(status_dict["created_at"], str):
                status_dict["created_at"] = status_dict["created_at"].isoformat()
            if status_dict.get("updated_at") and not isinstance(status_dict["updated_at"], str):
                status_dict["updated_at"] = status_dict["updated_at"].isoformat()
            result.append(status_dict)
        
        return result
    
    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document from the system
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dict[str, Any]: Deletion result
        """
        if not self.initialized:
            await self.initialize()
        
        result = await self.rag.delete_doc(doc_id)
        return {"status": "success" if result.success else "error", "message": result.message}
    
    async def get_knowledge_graph(self, limit: int = 1000) -> Dict[str, Any]:
        """Get the knowledge graph
        
        Args:
            limit: Maximum number of nodes to return
            
        Returns:
            Dict[str, Any]: Knowledge graph data
        """
        if not self.initialized:
            await self.initialize()
        
        graph = await self.rag.get_knowledge_graph(node_label="*", max_depth=3, max_nodes=limit)
        return {"nodes": graph.nodes, "edges": graph.edges}
    
    async def clear_cache(self, modes: List[str] = None) -> Dict[str, Any]:
        """Clear caches
        
        Args:
            modes: List of cache modes to clear
            
        Returns:
            Dict[str, Any]: Result
        """
        if not self.initialized:
            await self.initialize()
        
        if modes is None:
            modes = ["default", "naive", "local", "global", "hybrid", "mix"]
        
        self.rag.clear_cache(modes=modes)
        return {"status": "success", "cleared_modes": modes}
    
    async def finalize(self):
        """Finalize the pipeline and release resources"""
        if self.initialized and self.rag:
            await self.rag.finalize_storages()
            self.initialized = False
            logger.info("Production RAG Pipeline finalized")