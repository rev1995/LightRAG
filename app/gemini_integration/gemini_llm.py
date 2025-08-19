"""
Enhanced Gemini LLM Integration with Token Tracking
Compatible with local LightRAG source code
"""

import os
import asyncio
from typing import Dict, List, Any, Optional
import json

# Import with proper error handling
try:
    import google.generativeai as genai
    from google.generativeai import types
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Google Generative AI not available: {e}")
    GEMINI_AVAILABLE = False
    genai = None
    types = None

# Import from local LightRAG
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LightRAG"))

from lightrag.utils import TokenTracker, logger


class GeminiLLMWithTracking:
    """Enhanced Gemini LLM integration with comprehensive tracking"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", temperature: float = 0.0, max_output_tokens: int = 5000, top_k: int = 10, enable_tracking: bool = True):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI package is not installed. Please install with: pip install google-generativeai")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required but not provided")
        
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_k = top_k
        self.enable_tracking = enable_tracking
        
        try:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Gemini client: {e}")
        
        # Token tracking
        self.token_tracker = TokenTracker() if enable_tracking else None
        self.session_stats = {
            "total_calls": 0,
            "total_tokens": 0
        }
        
        logger.info(f"✅ Initialized Gemini LLM: {model}")
    
    async def generate_content(self, prompt: str, system_prompt: Optional[str] = None, history_messages: Optional[List[Dict[str, str]]] = None, stream: bool = False, **kwargs) -> Dict[str, Any]:
        """Generate content with comprehensive error handling"""
        
        if not GEMINI_AVAILABLE:
            return {
                "text": "",
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "success": False,
                "error": "Gemini API not available",
                "model": self.model,
                "response_metadata": {}
            }
        
        try:
            # Build prompt with proper handling
            full_prompt = self._build_prompt(prompt, system_prompt, history_messages)
            
            # Generation configuration with validation
            generation_config = types.GenerationConfig(
                temperature=max(0.0, min(2.0, self.temperature)),  # Clamp between 0-2
                max_output_tokens=max(1, min(8192, self.max_output_tokens)),  # Reasonable limits
                top_k=max(1, min(40, self.top_k)),
                **kwargs
            )
            
            # Generate content with timeout
            if stream:
                response = await self._generate_streaming(full_prompt, generation_config)
            else:
                response = await self._generate_non_streaming(full_prompt, generation_config)
            
            # Validate response
            if not response or not hasattr(response, 'text'):
                raise ValueError("Invalid response from Gemini API")
            
            # Extract token usage with error handling
            token_usage = self._extract_token_usage(response)
            
            # Track usage
            if self.enable_tracking and self.token_tracker:
                self.token_tracker.add_usage(token_usage)
                self._update_session_stats(token_usage)
            
            return {
                "text": response.text or "",
                "token_usage": token_usage,
                "success": True,
                "error": None,
                "model": self.model,
                "response_metadata": self._extract_metadata(response)
            }
            
        except Exception as e:
            error_msg = f"Gemini LLM generation failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "text": "",
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "success": False,
                "error": error_msg,
                "model": self.model,
                "response_metadata": {}
            }
    
    async def _generate_non_streaming(self, prompt: str, config: types.GenerationConfig):
        """Generate non-streaming response with timeout"""
        
        try:
            # Use asyncio timeout to prevent hanging
            response = await asyncio.wait_for(
                asyncio.to_thread(self.client.generate_content, prompt, generation_config=config),
                timeout=120.0  # 2 minute timeout
            )
            return response
        except asyncio.TimeoutError:
            raise TimeoutError("Gemini API request timed out after 2 minutes")
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")
    
    async def _generate_streaming(self, prompt: str, config: types.GenerationConfig):
        """Generate content with streaming (returns final response)"""
        # For streaming, we collect all chunks and return the final response
        full_text = ""
        final_response = None
        
        async def stream_generator():
            nonlocal full_text, final_response
            stream = self.client.generate_content(
                prompt,
                generation_config=config,
                stream=True
            )
            
            for chunk in stream:
                if chunk.text:
                    full_text += chunk.text
                final_response = chunk
        
        await asyncio.get_event_loop().run_in_executor(None, lambda: list(stream_generator()))
        
        # Create a response-like object with the full text
        class StreamResponse:
            def __init__(self, text, usage_metadata):
                self.text = text
                self.usage_metadata = usage_metadata
        
        return StreamResponse(full_text, getattr(final_response, 'usage_metadata', None))
    
    def _build_prompt(self, prompt: str, system_prompt: Optional[str] = None, history_messages: Optional[List[Dict[str, str]]] = None) -> str:
        """Build final prompt with proper validation"""
        
        parts = []
        
        # Add system prompt
        if system_prompt and system_prompt.strip():
            parts.append(f"System: {system_prompt.strip()}")
        
        # Add conversation history
        if history_messages:
            for msg in history_messages[-10:]:  # Limit history to prevent token overflow
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content.strip():
                    if role == "user":
                        parts.append(f"Human: {content.strip()}")
                    elif role == "assistant":
                        parts.append(f"Assistant: {content.strip()}")
        
        # Add current prompt
        if prompt and prompt.strip():
            parts.append(f"Human: {prompt.strip()}")
        
        # Add assistant prefix
        parts.append("Assistant:")
        
        return "\n\n".join(parts)
    
    def _extract_token_usage(self, response) -> Dict[str, int]:
        """Extract token usage with proper error handling"""
        
        try:
            # Try to get usage metadata from response
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                return {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }
            else:
                # Fallback: estimate tokens from text length
                prompt_est = len(response.prompt_feedback.block_reason if hasattr(response, 'prompt_feedback') else "") // 4
                completion_est = len(response.text or "") // 4
                return {
                    "prompt_tokens": max(0, prompt_est),
                    "completion_tokens": max(0, completion_est),
                    "total_tokens": max(0, prompt_est + completion_est)
                }
        except Exception:
            # Ultimate fallback
            text_length = len(getattr(response, 'text', ''))
            estimated_tokens = max(1, text_length // 4)
            return {
                "prompt_tokens": estimated_tokens // 2,
                "completion_tokens": estimated_tokens // 2,
                "total_tokens": estimated_tokens
            }
    
    def _extract_metadata(self, response) -> Dict[str, Any]:
        """Extract additional metadata from response"""
        metadata = {}
        
        # Safety ratings
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'safety_ratings'):
                metadata['safety_ratings'] = [
                    {
                        'category': rating.category.name,
                        'probability': rating.probability.name
                    }
                    for rating in candidate.safety_ratings
                ]
            
            if hasattr(candidate, 'finish_reason'):
                metadata['finish_reason'] = candidate.finish_reason.name
        
        return metadata
    
    def _update_session_stats(self, token_usage: Dict[str, int]):
        """Update session statistics with error handling"""
        
        try:
            self.session_stats["total_calls"] += 1
            self.session_stats["total_tokens"] += token_usage.get("total_tokens", 0)
        except Exception as e:
            logger.warning(f"Failed to update session stats: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return {
            **self.session_stats,
            "model": self.model,
            "tracking_enabled": self.enable_tracking,
            "api_available": GEMINI_AVAILABLE
        }
    
    def reset_session_stats(self):
        """Reset session statistics"""
        self.session_stats = {
            "total_calls": 0,
            "total_tokens": 0
        }
        
        if self.token_tracker:
            self.token_tracker.reset()


# Create LightRAG-compatible function
async def gemini_llm_model_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    keyword_extraction: bool = False,
    **kwargs
) -> str:
    """
    LightRAG-compatible Gemini LLM function
    
    This function signature matches what LightRAG expects
    """
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    # Get model configuration from environment
    model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    temperature = float(os.getenv("TEMPERATURE", "0.0"))
    max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "5000"))
    
    # Create LLM instance
    llm = GeminiLLMWithTracking(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens
    )
    
    # Generate content
    result = await llm.generate_content(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        **kwargs
    )
    
    # Return just the text (as expected by LightRAG)
    if result["success"]:
        return result["text"]
    else:
        raise Exception(result["error"])


# Global instance management with better error handling
_global_llm_instance: Optional[GeminiLLMWithTracking] = None

def get_global_llm_instance() -> Optional[GeminiLLMWithTracking]:
    """Get or create global LLM instance for tracking"""
    global _global_llm_instance
    
    if not GEMINI_AVAILABLE:
        return None
    
    if _global_llm_instance is None:
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not found in environment variables")
                return None
            
            model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
            temperature = float(os.getenv("TEMPERATURE", "0.0"))
            max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "5000"))
            
            _global_llm_instance = GeminiLLMWithTracking(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        except Exception as e:
            logger.error(f"Failed to create global LLM instance: {e}")
            return None
    
    return _global_llm_instance

def reset_global_llm_instance():
    """Reset global LLM instance"""
    global _global_llm_instance
    _global_llm_instance = None 