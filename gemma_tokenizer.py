"""
Gemma Tokenizer Implementation for LightRAG
Accurate token counting for Gemini models with caching and production-ready features.
"""

import os
import hashlib
import requests
import dataclasses
from pathlib import Path
from typing import List, Optional, Union
import logging

try:
    import sentencepiece as spm
except ImportError:
    raise ImportError(
        "sentencepiece is required. Install it with: pip install sentencepiece"
    )

# Add LightRAG to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lightrag'))

from lightrag.utils import Tokenizer, logger


@dataclasses.dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for tokenizer model download and caching"""
    tokenizer_model_url: str
    tokenizer_model_hash: str


class GemmaTokenizer(Tokenizer):
    """
    Gemma tokenizer implementation for accurate token counting with Gemini models.
    
    This tokenizer provides accurate token counting for Gemini models by using
    the appropriate Gemma tokenizer model based on the Gemini model version.
    """
    
    # Available tokenizer configurations
    _TOKENIZERS = {
        "google/gemma2": TokenizerConfig(
            tokenizer_model_url="https://raw.githubusercontent.com/google/gemma_pytorch/33b652c465537c6158f9a472ea5700e5e770ad3f/tokenizer/tokenizer.model",
            tokenizer_model_hash="61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
        ),
        "google/gemma3": TokenizerConfig(
            tokenizer_model_url="https://raw.githubusercontent.com/google/gemma_pytorch/cb7c0152a369e43908e769eb09e1ce6043afe084/tokenizer/gemma3_cleaned_262144_v2.spiece.model", 
            tokenizer_model_hash="1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        ),
    }
    
    def __init__(
        self, 
        model_name: str = "gemini-2.0-flash", 
        tokenizer_dir: Optional[str] = None,
        download_timeout: int = 300
    ):
        """
        Initialize Gemma tokenizer for the specified Gemini model.
        
        Args:
            model_name: The Gemini model name (e.g., "gemini-2.0-flash")
            tokenizer_dir: Directory to cache tokenizer models (defaults to ./tokenizer_cache)
            download_timeout: Timeout for downloading tokenizer models in seconds
        """
        self.model_name = model_name
        self.download_timeout = download_timeout
        
        # Set default tokenizer directory
        if tokenizer_dir is None:
            tokenizer_dir = os.getenv("TOKENIZER_CACHE_DIR", "./tokenizer_cache")
        
        self.tokenizer_dir = Path(tokenizer_dir)
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
        
        # Select appropriate tokenizer based on model
        tokenizer_name = self._select_tokenizer_for_model(model_name)
        logger.info(f"Using {tokenizer_name} tokenizer for model {model_name}")
        
        # Get tokenizer configuration
        tokenizer_config = self._TOKENIZERS[tokenizer_name]
        file_url = tokenizer_config.tokenizer_model_url
        tokenizer_model_name = file_url.rsplit("/", 1)[1]
        expected_hash = tokenizer_config.tokenizer_model_hash
        
        # Set up file path
        file_path = self.tokenizer_dir / tokenizer_model_name
        
        # Load or download tokenizer model
        model_data = self._load_tokenizer_model(file_path, file_url, expected_hash)
        
        # Initialize SentencePiece tokenizer
        try:
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.LoadFromSerializedProto(model_data)
            logger.info(f"Successfully loaded tokenizer for {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SentencePiece tokenizer: {e}")
            raise
        
        # Initialize parent class
        super().__init__(model_name=model_name, tokenizer=tokenizer)
    
    def _select_tokenizer_for_model(self, model_name: str) -> str:
        """Select the appropriate tokenizer based on the Gemini model version."""
        model_lower = model_name.lower()
        
        # For Gemini 1.5 and earlier, use gemma2 tokenizer
        if any(version in model_lower for version in ["1.0", "1.5"]):
            return "google/gemma2"
        
        # For Gemini 2.0 and later, use gemma3 tokenizer
        if any(version in model_lower for version in ["2.0", "2.1", "flash"]):
            return "google/gemma3"
        
        # Default to gemma3 for newer/unknown models
        logger.warning(f"Unknown model version for {model_name}, defaulting to gemma3 tokenizer")
        return "google/gemma3"
    
    def _load_tokenizer_model(self, file_path: Path, file_url: str, expected_hash: str) -> bytes:
        """Load tokenizer model from cache or download if needed."""
        # Try to load from cache first
        model_data = self._maybe_load_from_cache(file_path, expected_hash)
        
        if model_data is None:
            logger.info(f"Downloading tokenizer model from {file_url}")
            model_data = self._download_from_url(file_url, expected_hash)
            self._save_to_cache(file_path, model_data)
        else:
            logger.info(f"Loaded tokenizer model from cache: {file_path}")
        
        return model_data
    
    def _maybe_load_from_cache(self, file_path: Path, expected_hash: str) -> Optional[bytes]:
        """Load tokenizer model from cache if it exists and is valid."""
        if not file_path.is_file():
            return None
        
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            
            if self._is_valid_model(content, expected_hash):
                return content
            else:
                logger.warning(f"Cached tokenizer model {file_path} is invalid, will re-download")
                self._maybe_remove_file(file_path)
                return None
                
        except Exception as e:
            logger.warning(f"Error reading cached tokenizer model {file_path}: {e}")
            self._maybe_remove_file(file_path)
            return None
    
    def _download_from_url(self, file_url: str, expected_hash: str) -> bytes:
        """Download tokenizer model from URL with validation."""
        try:
            response = requests.get(file_url, timeout=self.download_timeout)
            response.raise_for_status()
            content = response.content
            
            if not self._is_valid_model(content, expected_hash):
                actual_hash = hashlib.sha256(content).hexdigest()
                raise ValueError(
                    f"Downloaded tokenizer model is corrupted. "
                    f"Expected hash {expected_hash}, got {actual_hash}"
                )
            
            logger.info(f"Successfully downloaded tokenizer model ({len(content)} bytes)")
            return content
            
        except requests.RequestException as e:
            logger.error(f"Failed to download tokenizer model from {file_url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing downloaded tokenizer model: {e}")
            raise
    
    def _is_valid_model(self, model_data: bytes, expected_hash: str) -> bool:
        """Validate tokenizer model by checking SHA256 hash."""
        actual_hash = hashlib.sha256(model_data).hexdigest()
        return actual_hash == expected_hash
    
    def _save_to_cache(self, file_path: Path, model_data: bytes):
        """Save tokenizer model to cache."""
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first, then rename for atomic operation
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, "wb") as f:
                f.write(model_data)
            
            # Atomic rename
            temp_path.rename(file_path)
            logger.info(f"Cached tokenizer model to {file_path}")
            
        except Exception as e:
            logger.warning(f"Failed to cache tokenizer model to {file_path}: {e}")
            self._maybe_remove_file(temp_path)
    
    @staticmethod
    def _maybe_remove_file(file_path: Path):
        """Safely remove file if it exists."""
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove file {file_path}: {e}")
    
    def encode(self, content: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            content: Text to encode
            
        Returns:
            List of token IDs
        """
        try:
            return self.tokenizer.encode(content)
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            # Fallback to rough estimation
            return list(range(len(content.split())))
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        try:
            return self.tokenizer.decode(tokens)
        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            return " ".join([f"<token_{t}>" for t in tokens])
    
    def count_tokens(self, content: str) -> int:
        """
        Count the number of tokens in the given content.
        
        Args:
            content: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encode(content))
    
    def truncate_to_token_limit(self, content: str, max_tokens: int) -> str:
        """
        Truncate content to fit within token limit.
        
        Args:
            content: Text to truncate
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Truncated text
        """
        if max_tokens <= 0:
            return ""
        
        tokens = self.encode(content)
        
        if len(tokens) <= max_tokens:
            return content
        
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        return self.decode(truncated_tokens)
    
    def get_token_limits(self) -> dict:
        """Get token limits for the model."""
        # Token limits based on Gemini model specifications
        model_limits = {
            "gemini-2.0-flash": {
                "input_limit": 1048576,  # 1M tokens
                "output_limit": 8192
            },
            "gemini-1.5-pro": {
                "input_limit": 2097152,  # 2M tokens
                "output_limit": 8192
            },
            "gemini-1.5-flash": {
                "input_limit": 1048576,  # 1M tokens
                "output_limit": 8192
            },
            "gemini-1.0-pro": {
                "input_limit": 32768,    # 32K tokens
                "output_limit": 8192
            }
        }
        
        # Find matching model or use default
        for model_key, limits in model_limits.items():
            if model_key in self.model_name.lower():
                return limits
        
        # Default limits for unknown models
        return {
            "input_limit": 1048576,  # 1M tokens
            "output_limit": 8192
        }
    
    def estimate_cost(self, input_tokens: int, output_tokens: int = 0) -> dict:
        """
        Estimate the cost for token usage (approximate pricing).
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary with cost estimation
        """
        # Approximate pricing (actual pricing may vary)
        pricing = {
            "gemini-2.0-flash": {
                "input_cost_per_1k": 0.000075,  # $0.000075 per 1K tokens
                "output_cost_per_1k": 0.0003    # $0.0003 per 1K tokens
            },
            "gemini-1.5-pro": {
                "input_cost_per_1k": 0.00125,   # $0.00125 per 1K tokens
                "output_cost_per_1k": 0.005     # $0.005 per 1K tokens
            },
            "gemini-1.5-flash": {
                "input_cost_per_1k": 0.000075,  # $0.000075 per 1K tokens
                "output_cost_per_1k": 0.0003    # $0.0003 per 1K tokens
            },
            "gemini-1.0-pro": {
                "input_cost_per_1k": 0.0005,    # $0.0005 per 1K tokens
                "output_cost_per_1k": 0.0015    # $0.0015 per 1K tokens
            }
        }
        
        # Find matching model pricing
        model_pricing = None
        for model_key, costs in pricing.items():
            if model_key in self.model_name.lower():
                model_pricing = costs
                break
        
        if model_pricing is None:
            # Default to Flash pricing
            model_pricing = pricing["gemini-2.0-flash"]
        
        # Calculate costs
        input_cost = (input_tokens / 1000) * model_pricing["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * model_pricing["output_cost_per_1k"]
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "currency": "USD",
            "model": self.model_name,
            "pricing_date": "2024-12",  # Approximate pricing date
            "note": "Pricing is approximate and may vary"
        }


# Global tokenizer instance
_gemma_tokenizer = None

def get_gemma_tokenizer(model_name: str = None) -> GemmaTokenizer:
    """Get or create global Gemma tokenizer instance."""
    global _gemma_tokenizer
    
    if model_name is None:
        model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    
    if _gemma_tokenizer is None or _gemma_tokenizer.model_name != model_name:
        _gemma_tokenizer = GemmaTokenizer(model_name)
    
    return _gemma_tokenizer


# Utility functions for LightRAG integration
def create_gemma_tokenizer_for_model(model_name: str) -> GemmaTokenizer:
    """Create a Gemma tokenizer for the specified model."""
    return GemmaTokenizer(model_name)


def validate_tokenizer_setup() -> bool:
    """Validate that the tokenizer can be initialized properly."""
    try:
        model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        tokenizer = GemmaTokenizer(model_name)
        
        # Test basic functionality
        test_text = "This is a test sentence for tokenizer validation."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        logger.info(f"Tokenizer validation successful for {model_name}")
        logger.info(f"Test text tokens: {len(tokens)}")
        return True
        
    except Exception as e:
        logger.error(f"Tokenizer validation failed: {e}")
        return False


def clear_tokenizer_cache(tokenizer_dir: str = None):
    """Clear the tokenizer model cache."""
    if tokenizer_dir is None:
        tokenizer_dir = os.getenv("TOKENIZER_CACHE_DIR", "./tokenizer_cache")
    
    cache_path = Path(tokenizer_dir)
    
    try:
        if cache_path.exists():
            for file in cache_path.iterdir():
                if file.is_file():
                    file.unlink()
            logger.info(f"Cleared tokenizer cache at {cache_path}")
        else:
            logger.info("No tokenizer cache found to clear")
    except Exception as e:
        logger.error(f"Error clearing tokenizer cache: {e}")


if __name__ == "__main__":
    # Test the tokenizer implementation
    import time
    
    def test_tokenizer():
        try:
            print("Testing Gemma tokenizer...")
            
            # Test tokenizer initialization
            if not validate_tokenizer_setup():
                print("Tokenizer validation failed")
                return
            
            # Create tokenizer
            tokenizer = get_gemma_tokenizer("gemini-2.0-flash")
            
            # Test text
            test_texts = [
                "This is a simple test sentence.",
                "Machine learning and artificial intelligence are transforming technology.",
                "The quick brown fox jumps over the lazy dog.",
                "Natural language processing enables computers to understand human language better.",
            ]
            
            print(f"\nTokenizer model: {tokenizer.model_name}")
            print(f"Token limits: {tokenizer.get_token_limits()}")
            
            total_tokens = 0
            start_time = time.time()
            
            for i, text in enumerate(test_texts, 1):
                tokens = tokenizer.encode(text)
                decoded = tokenizer.decode(tokens)
                token_count = tokenizer.count_tokens(text)
                
                print(f"\nTest {i}:")
                print(f"  Original: {text}")
                print(f"  Tokens: {len(tokens)} (count method: {token_count})")
                print(f"  Decoded: {decoded}")
                print(f"  Matches: {text == decoded}")
                
                total_tokens += len(tokens)
            
            end_time = time.time()
            
            # Test cost estimation
            cost_estimate = tokenizer.estimate_cost(total_tokens, total_tokens // 2)
            print(f"\nCost estimation: {cost_estimate}")
            
            # Test truncation
            long_text = " ".join(test_texts) * 10
            truncated = tokenizer.truncate_to_token_limit(long_text, 100)
            print(f"\nTruncation test:")
            print(f"  Original length: {len(tokenizer.encode(long_text))} tokens")
            print(f"  Truncated length: {len(tokenizer.encode(truncated))} tokens")
            
            print(f"\nTokenization completed in {end_time - start_time:.2f} seconds")
            print(f"Total tokens processed: {total_tokens}")
            
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    test_tokenizer() 