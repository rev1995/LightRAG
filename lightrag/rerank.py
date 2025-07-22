from __future__ import annotations

import os
import aiohttp
from typing import Callable, Any, List, Dict, Optional
from pydantic import BaseModel, Field

from .utils import logger


class RerankModel(BaseModel):
    """
    Wrapper for rerank functions that can be used with LightRAG.

    Example usage:
    ```python
    from lightrag.rerank import RerankModel, jina_rerank

    # Create rerank model
    rerank_model = RerankModel(
        rerank_func=jina_rerank,
        kwargs={
            "model": "BAAI/bge-reranker-v2-m3",
            "api_key": "your_api_key_here",
            "base_url": "https://api.jina.ai/v1/rerank"
        }
    )

    # Use in LightRAG
    rag = LightRAG(
        rerank_model_func=rerank_model.rerank,
        # ... other configurations
    )

    # Query with rerank enabled (default)
    result = await rag.aquery(
        "your query",
        param=QueryParam(enable_rerank=True)
    )
    ```

    Or define a custom function directly:
    ```python
    async def my_rerank_func(query: str, documents: list, top_k: int = None, **kwargs):
        return await jina_rerank(
            query=query,
            documents=documents,
            model="BAAI/bge-reranker-v2-m3",
            api_key="your_api_key_here",
            top_k=top_k or 10,
            **kwargs
        )

    rag = LightRAG(
        rerank_model_func=my_rerank_func,
        # ... other configurations
    )

    # Control rerank per query
    result = await rag.aquery(
        "your query",
        param=QueryParam(enable_rerank=True)  # Enable rerank for this query
    )
    ```
    """

    rerank_func: Callable[[Any], List[Dict]]
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        **extra_kwargs,
    ) -> List[Dict[str, Any]]:
        """Rerank documents using the configured model function."""
        # Merge extra kwargs with model kwargs
        kwargs = {**self.kwargs, **extra_kwargs}
        return await self.rerank_func(
            query=query, documents=documents, top_k=top_k, **kwargs
        )


class MultiRerankModel(BaseModel):
    """Multiple rerank models for different modes/scenarios."""

    # Primary rerank model (used if mode-specific models are not defined)
    rerank_model: Optional[RerankModel] = None

    # Mode-specific rerank models
    entity_rerank_model: Optional[RerankModel] = None
    relation_rerank_model: Optional[RerankModel] = None
    chunk_rerank_model: Optional[RerankModel] = None

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        mode: str = "default",
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Rerank using the appropriate model based on mode."""

        # Select model based on mode
        if mode == "entity" and self.entity_rerank_model:
            model = self.entity_rerank_model
        elif mode == "relation" and self.relation_rerank_model:
            model = self.relation_rerank_model
        elif mode == "chunk" and self.chunk_rerank_model:
            model = self.chunk_rerank_model
        elif self.rerank_model:
            model = self.rerank_model
        else:
            logger.warning(f"No rerank model available for mode: {mode}")
            return documents

        return await model.rerank(query, documents, top_k, **kwargs)


async def generic_rerank_api(
    query: str,
    documents: List[Dict[str, Any]],
    model: str,
    base_url: str,
    api_key: str,
    top_k: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Generic rerank function that works with Jina/Cohere compatible APIs.

    Args:
        query: The search query
        documents: List of documents to rerank
        model: Model identifier
        base_url: API endpoint URL
        api_key: API authentication key
        top_k: Number of top results to return
        **kwargs: Additional API-specific parameters

    Returns:
        List of reranked documents with relevance scores
    """
    if not api_key:
        logger.warning("No API key provided for rerank service")
        return documents

    if not documents:
        return documents

    # Prepare documents for reranking - handle both text and dict formats
    prepared_docs = []
    for doc in documents:
        if isinstance(doc, dict):
            # Use 'content' field if available, otherwise use 'text' or convert to string
            text = doc.get("content") or doc.get("text") or str(doc)
        else:
            text = str(doc)
        prepared_docs.append(text)

    # Prepare request
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    data = {"model": model, "query": query, "documents": prepared_docs, **kwargs}

    if top_k is not None:
        data["top_k"] = min(top_k, len(prepared_docs))

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(base_url, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Rerank API error {response.status}: {error_text}")
                    return documents

                result = await response.json()

                # Extract reranked results
                if "results" in result:
                    # Standard format: results contain index and relevance_score
                    reranked_docs = []
                    for item in result["results"]:
                        if "index" in item:
                            doc_idx = item["index"]
                            if 0 <= doc_idx < len(documents):
                                reranked_doc = documents[doc_idx].copy()
                                if "relevance_score" in item:
                                    reranked_doc["rerank_score"] = item[
                                        "relevance_score"
                                    ]
                                reranked_docs.append(reranked_doc)
                    return reranked_docs
                else:
                    logger.warning("Unexpected rerank API response format")
                    return documents

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        return documents


async def jina_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    model: str = "BAAI/bge-reranker-v2-m3",
    top_k: Optional[int] = None,
    base_url: str = "https://api.jina.ai/v1/rerank",
    api_key: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Jina AI API.

    Args:
        query: The search query
        documents: List of documents to rerank
        model: Jina rerank model name
        top_k: Number of top results to return
        base_url: Jina API endpoint
        api_key: Jina API key
        **kwargs: Additional parameters

    Returns:
        List of reranked documents with relevance scores
    """
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY") or os.getenv("RERANK_API_KEY")

    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_k=top_k,
        **kwargs,
    )


async def cohere_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    model: str = "rerank-english-v2.0",
    top_k: Optional[int] = None,
    base_url: str = "https://api.cohere.ai/v1/rerank",
    api_key: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Cohere API.

    Args:
        query: The search query
        documents: List of documents to rerank
        model: Cohere rerank model name
        top_k: Number of top results to return
        base_url: Cohere API endpoint
        api_key: Cohere API key
        **kwargs: Additional parameters

    Returns:
        List of reranked documents with relevance scores
    """
    if api_key is None:
        api_key = os.getenv("COHERE_API_KEY") or os.getenv("RERANK_API_KEY")

    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_k=top_k,
        **kwargs,
    )


# Convenience function for custom API endpoints
async def custom_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    model: str,
    base_url: str,
    api_key: str,
    top_k: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using a custom API endpoint.
    This is useful for self-hosted or custom rerank services.
    """
    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_k=top_k,
        **kwargs,
    )


async def gemini_llm_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.1,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Gemini LLM.
    
    This function uses Gemini's reasoning capabilities to evaluate the relevance
    of each document to the query and assign a relevance score.
    
    Args:
        query: The search query
        documents: List of documents to rerank
        top_k: Number of top results to return
        api_key: Gemini API key
        model: Gemini model name
        temperature: Temperature for generation
        **kwargs: Additional parameters
        
    Returns:
        List of reranked documents with relevance scores
    """
    if not documents:
        return documents
        
    # Use provided API key or get from environment
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
        
    if not api_key:
        logger.warning("No API key provided for Gemini LLM reranking")
        return documents
        
    try:
        from google import genai
        from google.genai import types
        
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
        # Prepare documents for reranking
        doc_texts = []
        for doc in documents:
            if isinstance(doc, dict):
                # Use 'content' field if available, otherwise use 'text' or convert to string
                text = doc.get("content") or doc.get("text") or str(doc)
            else:
                text = str(doc)
            doc_texts.append(text)
            
        # Create prompt for reranking
        prompt = f"""You are a document reranking system. Your task is to evaluate the relevance of each document to the given query and assign a relevance score from 0 to 10, where 10 is most relevant.

Query: {query}

Documents to rank:
"""
        
        for i, text in enumerate(doc_texts):
            prompt += f"\nDocument {i+1}:\n{text}\n"
            
        prompt += "\nFor each document, provide a relevance score from 0 to 10 based on how well it answers the query. Return your response in the following format:\nDocument 1: [score]\nDocument 2: [score]\n..."
        
        # Call Gemini model
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=temperature,
            ),
        )
        
        response_text = response.text
        
        # Parse scores from response
        scores = []
        for i, _ in enumerate(doc_texts):
            pattern = rf"Document {i+1}:\s*([0-9]{{1,2}}(?:\.[0-9]+)?)"
            import re
            match = re.search(pattern, response_text)
            if match:
                score = float(match.group(1))
                scores.append((i, score))
            else:
                # Default score if parsing fails
                scores.append((i, 5.0))
        
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top_k if specified
        if top_k is not None:
            scores = scores[:top_k]
        
        # Create reranked document list
        reranked_docs = []
        for idx, score in scores:
            if 0 <= idx < len(documents):
                reranked_doc = documents[idx].copy() if isinstance(documents[idx], dict) else {"content": documents[idx]}
                reranked_doc["rerank_score"] = score
                reranked_docs.append(reranked_doc)
        
        return reranked_docs
        
    except Exception as e:
        logger.error(f"Error during Gemini LLM reranking: {e}")
        return documents


if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        docs = [
            {"content": "The capital of France is Paris."},
            {"content": "Tokyo is the capital of Japan."},
            {"content": "London is the capital of England."},
        ]

        query = "What is the capital of France?"

        # Example with Jina rerank
        result_jina = await jina_rerank(
            query=query, documents=docs, top_k=2, api_key="your-api-key-here"
        )
        print("Jina rerank result:", result_jina)
        
        # Example with Gemini LLM rerank
        result_gemini = await gemini_llm_rerank(
            query=query, documents=docs, top_k=2, api_key="your-gemini-api-key-here"
        )
        print("Gemini LLM rerank result:", result_gemini)

    asyncio.run(main())
