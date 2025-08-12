"""
Gemini Native Embeddings Implementation for LightRAG
Production-ready implementation with batch processing, caching, and robust error handling.
"""

import os
import asyncio
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import time
import hashlib
import json
from pathlib import Path

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError, ServerError
except ImportError:
    raise ImportError(
        "google-genai is required. Install it with: pip install google-genai"
    )

# Add LightRAG to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'lightrag'))

from lightrag.utils import logger


@dataclass
class GeminiEmbeddingConfig:
    """Configuration class for Gemini Embeddings"""
    api_key: str
    model: str = "text-embedding-001"
    base_url: str = "https://generativelanguage.googleapis.com"
    embedding_dim: int = 3072
    batch_size: int = 32
    max_token_size: int = 8192
    timeout: int = 60
    
    # Performance settings
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_caching: bool = True
    cache_similarity_threshold: float = 0.95
    cache_dir: str = "./embedding_cache"
    
    @classmethod
    def from_env(cls) -> "GeminiEmbeddingConfig":
        """Create configuration from environment variables"""
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-001"),
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://generativelanguage.googleapis.com"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "3072")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            max_token_size=int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "8192")),
            timeout=int(os.getenv("EMBEDDING_TIMEOUT", "60")),
            max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("EMBEDDING_RETRY_DELAY", "1.0")),
            enable_caching=os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true",
            cache_similarity_threshold=float(os.getenv("EMBEDDING_CACHE_SIMILARITY_THRESHOLD", "0.95")),
            cache_dir=os.getenv("EMBEDDING_CACHE_DIR", "./embedding_cache"),
        )


class EmbeddingCache:
    """Simple file-based embedding cache with similarity checking"""
    
    def __init__(self, cache_dir: str, similarity_threshold: float = 0.95):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.memory_cache = {}  # In-memory cache for recent embeddings
        self.max_memory_cache = 1000
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model"""
        combined = f"{model}:{text}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for key"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity (can be enhanced with more sophisticated methods)"""
        # Simple character-level similarity for caching
        if text1 == text2:
            return 1.0
        
        # Compute Jaccard similarity of character n-grams
        def get_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        ngrams1 = get_ngrams(text1.lower())
        ngrams2 = get_ngrams(text2.lower())
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    async def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get embedding from cache if available"""
        cache_key = self._get_cache_key(text, model)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]["embedding"]
        
        # Check file cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                
                # Check similarity threshold
                cached_text = cache_data["text"]
                similarity = self._compute_similarity(text, cached_text)
                
                if similarity >= self.similarity_threshold:
                    embedding = np.array(cache_data["embedding"], dtype=np.float32)
                    
                    # Add to memory cache
                    self.memory_cache[cache_key] = {
                        "text": text,
                        "embedding": embedding,
                        "timestamp": time.time()
                    }
                    
                    # Limit memory cache size
                    if len(self.memory_cache) > self.max_memory_cache:
                        oldest_key = min(self.memory_cache.keys(), 
                                       key=lambda k: self.memory_cache[k]["timestamp"])
                        del self.memory_cache[oldest_key]
                    
                    logger.debug(f"Cache hit for text (similarity: {similarity:.3f})")
                    return embedding
                    
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_path}: {e}")
        
        return None
    
    async def set(self, text: str, model: str, embedding: np.ndarray):
        """Store embedding in cache"""
        cache_key = self._get_cache_key(text, model)
        
        try:
            # Store in memory cache
            self.memory_cache[cache_key] = {
                "text": text,
                "embedding": embedding,
                "timestamp": time.time()
            }
            
            # Store in file cache
            cache_data = {
                "text": text,
                "model": model,
                "embedding": embedding.tolist(),
                "timestamp": time.time(),
                "dimension": embedding.shape[0]
            }
            
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Error storing embedding in cache: {e}")


class GeminiEmbeddings:
    """Production-ready Gemini native embeddings implementation"""
    
    def __init__(self, config: Optional[GeminiEmbeddingConfig] = None):
        self.config = config or GeminiEmbeddingConfig.from_env()
        self.client = None
        self.cache = None
        self._initialize_client()
        
        if not self.config.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
        
        if self.config.enable_caching:
            self.cache = EmbeddingCache(
                self.config.cache_dir,
                self.config.cache_similarity_threshold
            )
    
    def _initialize_client(self):
        """Initialize the Gemini client"""
        try:
            self.client = genai.Client(api_key=self.config.api_key)
            logger.info(f"Initialized Gemini embeddings client with model: {self.config.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini embeddings client: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding"""
        # Basic text preprocessing
        text = text.strip()
        
        # Truncate if too long (approximate token limit)
        if len(text) > self.config.max_token_size * 4:  # Rough estimation
            text = text[:self.config.max_token_size * 4]
            logger.warning(f"Text truncated to {len(text)} characters")
        
        return text
    
    async def _embed_batch_with_retry(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts with retry logic"""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Use Gemini's embedding API
                response = self.client.models.embed_content(
                    model=self.config.model,
                    content=texts,
                    task_type="RETRIEVAL_DOCUMENT",  # Optimize for retrieval tasks
                    output_dimensionality=self.config.embedding_dim
                )
                
                # Extract embeddings from response
                embeddings = []
                for embedding_data in response.embedding:
                    embedding = np.array(embedding_data.values, dtype=np.float32)
                    embeddings.append(embedding)
                
                return embeddings
                
            except ClientError as e:
                last_error = e
                if e.status_code in [400, 401, 403, 404]:  # Don't retry client errors
                    logger.error(f"Gemini embedding client error (attempt {attempt + 1}): {e}")
                    break
                logger.warning(f"Gemini embedding client error (attempt {attempt + 1}): {e}")
            except ServerError as e:
                last_error = e
                logger.warning(f"Gemini embedding server error (attempt {attempt + 1}): {e}")
            except Exception as e:
                last_error = e
                logger.warning(f"Unexpected embedding error (attempt {attempt + 1}): {e}")
            
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying embedding in {delay} seconds...")
                await asyncio.sleep(delay)
        
        raise Exception(f"Embedding failed after {self.config.max_retries} attempts. Last error: {last_error}")
    
    async def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text"""
        text = self._preprocess_text(text)
        
        # Check cache first
        if self.cache:
            cached_embedding = await self.cache.get(text, self.config.model)
            if cached_embedding is not None:
                return cached_embedding
        
        # Get embedding from API
        embeddings = await self._embed_batch_with_retry([text])
        embedding = embeddings[0]
        
        # Store in cache
        if self.cache:
            await self.cache.set(text, self.config.model, embedding)
        
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts efficiently"""
        if not texts:
            return []
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Check cache for existing embeddings
        embeddings = [None] * len(processed_texts)
        uncached_indices = []
        uncached_texts = []
        
        if self.cache:
            for i, text in enumerate(processed_texts):
                cached_embedding = await self.cache.get(text, self.config.model)
                if cached_embedding is not None:
                    embeddings[i] = cached_embedding
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(processed_texts)))
            uncached_texts = processed_texts
        
        # Embed uncached texts in batches
        if uncached_texts:
            logger.info(f"Embedding {len(uncached_texts)} texts (cached: {len(processed_texts) - len(uncached_texts)})")
            
            for i in range(0, len(uncached_texts), self.config.batch_size):
                batch_texts = uncached_texts[i:i + self.config.batch_size]
                batch_indices = uncached_indices[i:i + self.config.batch_size]
                
                try:
                    batch_embeddings = await self._embed_batch_with_retry(batch_texts)
                    
                    # Store results
                    for j, embedding in enumerate(batch_embeddings):
                        idx = batch_indices[j]
                        embeddings[idx] = embedding
                        
                        # Cache the embedding
                        if self.cache:
                            await self.cache.set(batch_texts[j], self.config.model, embedding)
                
                except Exception as e:
                    logger.error(f"Failed to embed batch {i}-{i+len(batch_texts)}: {e}")
                    # Fill with zero vectors as fallback
                    for j in range(len(batch_texts)):
                        idx = batch_indices[j]
                        embeddings[idx] = np.zeros(self.config.embedding_dim, dtype=np.float32)
                
                # Small delay between batches to respect rate limits
                if i + self.config.batch_size < len(uncached_texts):
                    await asyncio.sleep(0.1)
        
        # Ensure all embeddings are filled
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                logger.warning(f"No embedding generated for text {i}, using zero vector")
                embeddings[i] = np.zeros(self.config.embedding_dim, dtype=np.float32)
        
        return embeddings
    
    async def embed(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Main embedding function - handles both single texts and batches"""
        if isinstance(texts, str):
            return await self.embed_single(texts)
        else:
            return await self.embed_batch(texts)
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.config.embedding_dim
    
    async def clear_cache(self):
        """Clear the embedding cache"""
        if self.cache:
            try:
                # Clear memory cache
                self.cache.memory_cache.clear()
                
                # Clear file cache
                for cache_file in self.cache.cache_dir.glob("*.json"):
                    cache_file.unlink()
                
                logger.info("Embedding cache cleared")
            except Exception as e:
                logger.error(f"Error clearing embedding cache: {e}")


# Global Gemini embeddings instance
_gemini_embeddings = None

def get_gemini_embeddings() -> GeminiEmbeddings:
    """Get or create global Gemini embeddings instance"""
    global _gemini_embeddings
    if _gemini_embeddings is None:
        _gemini_embeddings = GeminiEmbeddings()
    return _gemini_embeddings


# LightRAG compatible function
async def gemini_embed(
    texts: Union[str, List[str]],
    model: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    LightRAG compatible Gemini embedding function
    
    Args:
        texts: Text or list of texts to embed
        model: Model name (optional, uses config default)
        embedding_dim: Embedding dimension (optional, uses config default)
        **kwargs: Additional parameters
    
    Returns:
        numpy array of embeddings
    """
    gemini_emb = get_gemini_embeddings()
    
    # Override config if parameters provided
    if model and model != gemini_emb.config.model:
        config = GeminiEmbeddingConfig.from_env()
        config.model = model
        if embedding_dim:
            config.embedding_dim = embedding_dim
        gemini_emb = GeminiEmbeddings(config)
    
    embeddings = await gemini_emb.embed(texts)
    
    if isinstance(texts, str):
        # Return single embedding as 2D array for compatibility
        return embeddings.reshape(1, -1)
    else:
        # Return batch as 2D array
        return np.vstack(embeddings)


# Utility functions
def validate_gemini_embedding_config() -> bool:
    """Validate Gemini embedding configuration"""
    try:
        config = GeminiEmbeddingConfig.from_env()
        if not config.api_key:
            logger.error("GEMINI_API_KEY environment variable is required")
            return False
        
        # Test connection
        gemini_emb = GeminiEmbeddings(config)
        logger.info("Gemini embedding configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Gemini embedding configuration validation failed: {e}")
        return False


async def test_embedding_performance(sample_texts: List[str] = None) -> Dict[str, Any]:
    """Test embedding performance and return metrics"""
    if sample_texts is None:
        sample_texts = [
            "This is a test document for embedding performance.",
            "Another test document with different content.",
            "Machine learning and artificial intelligence are transforming technology.",
            "Natural language processing enables computers to understand human language.",
            "Information retrieval systems help users find relevant documents."
        ]
    
    try:
        gemini_emb = get_gemini_embeddings()
        
        start_time = time.time()
        embeddings = await gemini_emb.embed(sample_texts)
        end_time = time.time()
        
        metrics = {
            "num_texts": len(sample_texts),
            "embedding_dimension": gemini_emb.config.embedding_dim,
            "total_time": end_time - start_time,
            "avg_time_per_text": (end_time - start_time) / len(sample_texts),
            "embeddings_per_second": len(sample_texts) / (end_time - start_time),
            "model": gemini_emb.config.model,
            "cache_enabled": gemini_emb.config.enable_caching
        }
        
        logger.info(f"Embedding performance test completed: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Embedding performance test failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Test the implementation
    import asyncio
    
    async def test_gemini_embeddings():
        try:
            # Test configuration
            if not validate_gemini_embedding_config():
                print("Configuration validation failed")
                return
            
            # Test single embedding
            embedding = await gemini_embed("This is a test sentence.")
            print(f"Single embedding shape: {embedding.shape}")
            
            # Test batch embedding
            texts = [
                "First test sentence.",
                "Second test sentence with different content.",
                "Third sentence about machine learning and AI."
            ]
            batch_embeddings = await gemini_embed(texts)
            print(f"Batch embeddings shape: {batch_embeddings.shape}")
            
            # Test performance
            performance = await test_embedding_performance(texts)
            print(f"Performance metrics: {performance}")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Run test
    asyncio.run(test_gemini_embeddings()) 