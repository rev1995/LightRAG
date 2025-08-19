"""
Gemini Embedding Integration with Token Tracking
Compatible with local LightRAG source code
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import google.generativeai as genai

# Import from local LightRAG
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LightRAG"))

from lightrag.utils import TokenTracker, logger, EmbeddingFunc


class GeminiEmbeddingWithTracking:
    """Enhanced Gemini Embedding with comprehensive token tracking"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
        embedding_dim: int = 768,
        max_token_size: int = 8192,
        enable_tracking: bool = True
    ):
        self.api_key = api_key
        self.model = model
        self.embedding_dim = embedding_dim
        self.max_token_size = max_token_size
        self.enable_tracking = enable_tracking
        
        # Initialize Gemini client
        genai.configure(api_key=api_key)
        
        # Token tracking
        self.token_tracker = TokenTracker() if enable_tracking else None
        self.session_stats = {
            "total_requests": 0,
            "total_texts": 0,
            "total_tokens": 0
        }
        
        logger.info(f"✅ Initialized Gemini Embeddings: {model} (dim={embedding_dim})")
    
    async def embed_documents(
        self, 
        texts: List[str],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Generate embeddings for multiple documents with tracking
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            Dict containing:
            - embeddings: numpy array of embeddings
            - token_usage: Token usage statistics
            - success: Whether embedding was successful
            - error: Error message if any
        """
        
        try:
            if not texts:
                return {
                    "embeddings": np.array([]),
                    "token_usage": {"total_tokens": 0},
                    "success": True,
                    "error": None
                }
            
            all_embeddings = []
            total_token_usage = {"total_tokens": 0}
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                batch_result = await self._embed_batch(batch_texts)
                
                if not batch_result["success"]:
                    return batch_result
                
                all_embeddings.extend(batch_result["embeddings"])
                total_token_usage["total_tokens"] += batch_result["token_usage"]["total_tokens"]
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            
            # Track usage
            if self.enable_tracking and self.token_tracker:
                self.token_tracker.add_usage(total_token_usage)
                self._update_session_stats(total_token_usage, len(texts))
            
            return {
                "embeddings": embeddings_array,
                "token_usage": total_token_usage,
                "success": True,
                "error": None,
                "model": self.model,
                "embedding_dim": self.embedding_dim
            }
            
        except Exception as e:
            error_msg = f"Gemini embedding generation failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "embeddings": np.array([]),
                "token_usage": {"total_tokens": 0},
                "success": False,
                "error": error_msg,
                "model": self.model,
                "embedding_dim": self.embedding_dim
            }
    
    async def _embed_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Embed a batch of texts"""
        
        try:
            # Truncate texts to max token size
            processed_texts = [self._truncate_text(text) for text in texts]
            
            # Generate embeddings using Gemini
            embeddings = []
            total_tokens = 0
            
            for text in processed_texts:
                # Use genai.embed_content for text embedding
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda t=text: genai.embed_content(
                        model=f"models/{self.model}",
                        content=t,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
                
                embeddings.append(result['embedding'])
                
                # Estimate tokens (Gemini doesn't provide exact token count for embeddings)
                estimated_tokens = len(text.split()) * 1.3  # Rough estimate
                total_tokens += int(estimated_tokens)
            
            return {
                "embeddings": embeddings,
                "token_usage": {"total_tokens": total_tokens},
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "embeddings": [],
                "token_usage": {"total_tokens": 0},
                "success": False,
                "error": str(e)
            }
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to maximum token size"""
        # Simple word-based truncation (should be improved with proper tokenizer)
        words = text.split()
        estimated_tokens = len(words) * 1.3
        
        if estimated_tokens > self.max_token_size:
            max_words = int(self.max_token_size / 1.3)
            return " ".join(words[:max_words])
        
        return text
    
    def _update_session_stats(self, token_usage: Dict[str, int], num_texts: int):
        """Update session statistics"""
        self.session_stats["total_requests"] += 1
        self.session_stats["total_texts"] += num_texts
        self.session_stats["total_tokens"] += token_usage.get("total_tokens", 0)
        
        # Remove cost calculations as requested
    

    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return {
            **self.session_stats,
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "tracking_enabled": self.enable_tracking
        }
    
    def reset_session_stats(self):
        """Reset session statistics"""
        self.session_stats = {
            "total_requests": 0,
            "total_texts": 0,
            "total_tokens": 0
        }
        
        if self.token_tracker:
            self.token_tracker.reset()


# Create LightRAG-compatible embedding function
async def gemini_embedding_func(texts: List[str]) -> np.ndarray:
    """
    LightRAG-compatible Gemini embedding function
    
    This function signature matches what LightRAG expects
    """
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    # Get model configuration from environment
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    max_token_size = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "8192"))
    
    # Create embedding instance
    embedder = GeminiEmbeddingWithTracking(
        api_key=api_key,
        model=model,
        embedding_dim=embedding_dim,
        max_token_size=max_token_size
    )
    
    # Generate embeddings
    result = await embedder.embed_documents(texts)
    
    # Return numpy array (as expected by LightRAG)
    if result["success"]:
        return result["embeddings"]
    else:
        raise Exception(result["error"])


def create_gemini_embedding_func() -> EmbeddingFunc:
    """
    Create a LightRAG EmbeddingFunc wrapper for Gemini embeddings
    """
    
    # Get configuration from environment
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    max_token_size = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "8192"))
    
    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=gemini_embedding_func
    )


# Global instance management
_global_embedding_instance: Optional[GeminiEmbeddingWithTracking] = None

def get_global_embedding_instance() -> Optional[GeminiEmbeddingWithTracking]:
    """Get or create global embedding instance for tracking"""
    global _global_embedding_instance
    
    if _global_embedding_instance is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            model = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
            embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
            max_token_size = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "8192"))
            
            _global_embedding_instance = GeminiEmbeddingWithTracking(
                api_key=api_key,
                model=model,
                embedding_dim=embedding_dim,
                max_token_size=max_token_size
            )
    
    return _global_embedding_instance

def reset_global_embedding_instance():
    """Reset global embedding instance"""
    global _global_embedding_instance
    _global_embedding_instance = None 