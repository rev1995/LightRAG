"""
Gemini 2.0 Flash LLM Implementation for LightRAG
Production-ready implementation with token tracking, streaming support, and robust error handling.
"""

import os
import asyncio
import json
import logging
from typing import Dict, List, Optional, Union, Any, AsyncIterator
from dataclasses import dataclass
import time
from enum import Enum

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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lightrag'))

from lightrag.utils import TokenTracker, logger
from lightrag.types import GPTKeywordExtractionFormat


class SafetySetting(Enum):
    """Gemini safety settings enum"""
    BLOCK_NONE = "BLOCK_NONE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"


@dataclass
class GeminiConfig:
    """Configuration class for Gemini 2.0 Flash"""
    api_key: str
    model: str = "gemini-2.0-flash"
    base_url: str = "https://generativelanguage.googleapis.com"
    max_output_tokens: int = 8192
    temperature: float = 0.1
    top_k: int = 40
    top_p: float = 0.9
    candidate_count: int = 1
    stop_sequences: Optional[List[str]] = None
    timeout: int = 300
    
    # Safety settings
    safety_harassment: SafetySetting = SafetySetting.BLOCK_MEDIUM_AND_ABOVE
    safety_hate_speech: SafetySetting = SafetySetting.BLOCK_MEDIUM_AND_ABOVE
    safety_sexually_explicit: SafetySetting = SafetySetting.BLOCK_MEDIUM_AND_ABOVE
    safety_dangerous_content: SafetySetting = SafetySetting.BLOCK_MEDIUM_AND_ABOVE
    
    # Performance settings
    enable_token_tracking: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    
    @classmethod
    def from_env(cls) -> "GeminiConfig":
        """Create configuration from environment variables"""
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
            base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
            max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            top_k=int(os.getenv("TOP_K_SAMPLING", "40")),
            top_p=float(os.getenv("TOP_P_SAMPLING", "0.9")),
            candidate_count=int(os.getenv("GEMINI_CANDIDATE_COUNT", "1")),
            stop_sequences=os.getenv("GEMINI_STOP_SEQUENCES", "").split(",") if os.getenv("GEMINI_STOP_SEQUENCES") else None,
            timeout=int(os.getenv("TIMEOUT", "300")),
            safety_harassment=SafetySetting(os.getenv("GEMINI_SAFETY_HARASSMENT", "BLOCK_MEDIUM_AND_ABOVE")),
            safety_hate_speech=SafetySetting(os.getenv("GEMINI_SAFETY_HATE_SPEECH", "BLOCK_MEDIUM_AND_ABOVE")),
            safety_sexually_explicit=SafetySetting(os.getenv("GEMINI_SAFETY_SEXUALLY_EXPLICIT", "BLOCK_MEDIUM_AND_ABOVE")),
            safety_dangerous_content=SafetySetting(os.getenv("GEMINI_SAFETY_DANGEROUS_CONTENT", "BLOCK_MEDIUM_AND_ABOVE")),
            enable_token_tracking=os.getenv("ENABLE_TOKEN_TRACKING", "true").lower() == "true",
            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("GEMINI_RETRY_DELAY", "1.0")),
        )


class GeminiLLM:
    """Production-ready Gemini 2.0 Flash LLM implementation"""
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig.from_env()
        self.client = None
        self.token_tracker = TokenTracker() if self.config.enable_token_tracking else None
        self._initialize_client()
        
        if not self.config.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
    
    def _initialize_client(self):
        """Initialize the Gemini client"""
        try:
            self.client = genai.Client(api_key=self.config.api_key)
            logger.info(f"Initialized Gemini client with model: {self.config.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def _build_safety_settings(self) -> List[Dict[str, str]]:
        """Build safety settings for Gemini"""
        return [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": self.config.safety_harassment.value
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": self.config.safety_hate_speech.value
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": self.config.safety_sexually_explicit.value
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": self.config.safety_dangerous_content.value
            }
        ]
    
    def _format_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Format prompt with system instructions and conversation history"""
        combined_prompt = ""
        
        if system_prompt:
            combined_prompt += f"System: {system_prompt}\n\n"
        
        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                combined_prompt += f"{role}: {content}\n"
        
        combined_prompt += f"user: {prompt}"
        return combined_prompt
    
    def _extract_token_usage(self, response) -> Dict[str, int]:
        """Extract token usage from Gemini response"""
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    
    async def _make_request_with_retry(
        self,
        formatted_prompt: str,
        keyword_extraction: bool = False,
        stream: bool = False,
        **kwargs
    ):
        """Make request to Gemini with retry logic"""
        generation_config = types.GenerateContentConfig(
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            candidate_count=self.config.candidate_count,
            stop_sequences=self.config.stop_sequences or [],
        )
        
        # Handle keyword extraction response format
        if keyword_extraction:
            generation_config.response_mime_type = "application/json"
            generation_config.response_schema = GPTKeywordExtractionFormat.model_json_schema()
        
        request_params = {
            "model": self.config.model,
            "contents": [formatted_prompt],
            "config": generation_config,
            "safety_settings": self._build_safety_settings(),
        }
        
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                if stream:
                    return self.client.models.generate_content_stream(**request_params)
                else:
                    response = self.client.models.generate_content(**request_params)
                    return response
                    
            except ClientError as e:
                last_error = e
                if e.status_code in [400, 401, 403, 404]:  # Don't retry client errors
                    logger.error(f"Gemini client error (attempt {attempt + 1}): {e}")
                    break
                logger.warning(f"Gemini client error (attempt {attempt + 1}): {e}")
            except ServerError as e:
                last_error = e
                logger.warning(f"Gemini server error (attempt {attempt + 1}): {e}")
            except Exception as e:
                last_error = e
                logger.warning(f"Unexpected error (attempt {attempt + 1}): {e}")
            
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
        
        raise Exception(f"Failed after {self.config.max_retries} attempts. Last error: {last_error}")
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        keyword_extraction: bool = False,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """
        Complete a prompt using Gemini 2.0 Flash
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            history_messages: Conversation history
            keyword_extraction: Whether to extract keywords in JSON format
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            String response or async iterator for streaming
        """
        start_time = time.time()
        
        try:
            # Format the prompt
            formatted_prompt = self._format_prompt(prompt, system_prompt, history_messages)
            
            # Make the request
            response = await self._make_request_with_retry(
                formatted_prompt, keyword_extraction, stream, **kwargs
            )
            
            if stream:
                return self._handle_streaming_response(response, start_time)
            else:
                return self._handle_single_response(response, start_time)
                
        except Exception as e:
            logger.error(f"Gemini completion failed: {e}")
            raise
    
    def _handle_single_response(self, response, start_time: float) -> str:
        """Handle non-streaming response"""
        try:
            response_text = response.text
            
            # Track token usage
            if self.token_tracker:
                token_usage = self._extract_token_usage(response)
                token_usage["request_time"] = time.time() - start_time
                self.token_tracker.add_usage(token_usage)
                
                logger.debug(f"Gemini completion: {token_usage['total_tokens']} tokens in {token_usage['request_time']:.2f}s")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error handling Gemini response: {e}")
            raise
    
    async def _handle_streaming_response(self, response_stream, start_time: float) -> AsyncIterator[str]:
        """Handle streaming response"""
        total_tokens = 0
        completion_tokens = 0
        
        try:
            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                    
                # Track tokens for streaming (approximation)
                if self.token_tracker and hasattr(chunk, 'usage_metadata'):
                    usage = self._extract_token_usage(chunk)
                    total_tokens = max(total_tokens, usage.get("total_tokens", 0))
                    completion_tokens = max(completion_tokens, usage.get("completion_tokens", 0))
            
            # Log final token usage for streaming
            if self.token_tracker and total_tokens > 0:
                token_usage = {
                    "prompt_tokens": total_tokens - completion_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "request_time": time.time() - start_time,
                    "streaming": True
                }
                self.token_tracker.add_usage(token_usage)
                logger.debug(f"Gemini streaming completion: {total_tokens} tokens in {token_usage['request_time']:.2f}s")
                
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            raise
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        if not self.token_tracker:
            return {"tracking_disabled": True}
        return self.token_tracker.get_summary()
    
    def reset_token_tracker(self):
        """Reset token usage tracking"""
        if self.token_tracker:
            self.token_tracker = TokenTracker()


# Global Gemini LLM instance
_gemini_llm = None

def get_gemini_llm() -> GeminiLLM:
    """Get or create global Gemini LLM instance"""
    global _gemini_llm
    if _gemini_llm is None:
        _gemini_llm = GeminiLLM()
    return _gemini_llm


# LightRAG compatible functions
async def gemini_complete_if_cache(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    keyword_extraction: bool = False,
    stream: bool = False,
    **kwargs
) -> Union[str, AsyncIterator[str]]:
    """
    LightRAG compatible Gemini completion function with caching support
    
    This function provides the same interface as other LLM implementations in LightRAG
    """
    gemini = get_gemini_llm()
    return await gemini.complete(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        stream=stream,
        **kwargs
    )


async def gemini_model_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    keyword_extraction: bool = False,
    stream: bool = False,
    **kwargs
) -> Union[str, AsyncIterator[str]]:
    """
    Main Gemini model completion function for LightRAG
    
    This is the primary function that should be used as llm_model_func in LightRAG configuration
    """
    return await gemini_complete_if_cache(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        stream=stream,
        **kwargs
    )


# Utility functions for configuration
def validate_gemini_config() -> bool:
    """Validate Gemini configuration"""
    try:
        config = GeminiConfig.from_env()
        if not config.api_key:
            logger.error("GEMINI_API_KEY environment variable is required")
            return False
        
        # Test connection
        gemini = GeminiLLM(config)
        logger.info("Gemini configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Gemini configuration validation failed: {e}")
        return False


def get_gemini_models() -> List[str]:
    """Get list of available Gemini models"""
    try:
        config = GeminiConfig.from_env()
        client = genai.Client(api_key=config.api_key)
        
        # Note: This would need to be updated based on actual Gemini API
        # for listing available models
        available_models = [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro"
        ]
        
        return available_models
        
    except Exception as e:
        logger.error(f"Failed to get Gemini models: {e}")
        return ["gemini-2.0-flash"]  # Default fallback


if __name__ == "__main__":
    # Test the implementation
    import asyncio
    
    async def test_gemini():
        try:
            # Test configuration
            if not validate_gemini_config():
                print("Configuration validation failed")
                return
            
            # Test completion
            response = await gemini_model_complete(
                "Explain the concept of RAG (Retrieval-Augmented Generation) in simple terms.",
                system_prompt="You are a helpful AI assistant specialized in explaining AI concepts."
            )
            
            print(f"Response: {response}")
            
            # Test token tracking
            gemini = get_gemini_llm()
            usage = gemini.get_token_usage()
            print(f"Token usage: {usage}")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Run test
    asyncio.run(test_gemini()) 