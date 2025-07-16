"""
LLM-Based Reranking Solution for LightRAG

This module provides an efficient LLM-based reranking approach that:
1. Uses the same LLM for both generation and reranking (cost-effective)
2. Provides semantic relevance scoring
3. Supports batch processing for efficiency
4. Includes caching for repeated queries
5. Offers multiple reranking strategies
"""

import asyncio
import json
import hashlib
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .utils import logger


@dataclass
class RerankResult:
    """Result of LLM-based reranking"""
    document: Dict[str, Any]
    relevance_score: float
    reasoning: str
    confidence: float


class LLMReranker:
    """
    Efficient LLM-based reranking using the same LLM for generation and reranking.
    
    This approach is more cost-effective and reliable than external reranking services
    because it uses the same LLM that's already configured for the RAG system.
    """
    
    def __init__(
        self,
        llm_func: Callable,
        batch_size: int = 5,
        max_concurrent: int = 3,
        cache_enabled: bool = True,
        strategy: str = "semantic_scoring"
    ):
        """
        Initialize LLM-based reranker.
        
        Args:
            llm_func: The LLM function to use for reranking
            batch_size: Number of documents to process in each batch
            max_concurrent: Maximum concurrent reranking operations
            cache_enabled: Enable caching for reranking results
            strategy: Reranking strategy ('semantic_scoring', 'relevance_ranking', 'hybrid')
        """
        self.llm_func = llm_func
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.cache_enabled = cache_enabled
        self.strategy = strategy
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
    def _generate_cache_key(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Generate cache key for reranking results"""
        content_hash = hashlib.md5(
            json.dumps([doc.get('content', str(doc)) for doc in documents], sort_keys=True).encode()
        ).hexdigest()
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"rerank_{query_hash}_{content_hash}_{self.strategy}"
    
    def _create_rerank_prompt(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        strategy: str = None
    ) -> str:
        """Create reranking prompt based on strategy"""
        strategy = strategy or self.strategy
        
        if strategy == "semantic_scoring":
            return self._create_semantic_scoring_prompt(query, documents)
        elif strategy == "relevance_ranking":
            return self._create_relevance_ranking_prompt(query, documents)
        elif strategy == "hybrid":
            return self._create_hybrid_prompt(query, documents)
        else:
            return self._create_semantic_scoring_prompt(query, documents)
    
    def _create_semantic_scoring_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create prompt for semantic relevance scoring"""
        docs_text = "\n".join([
            f"{i+1}. {doc.get('content', str(doc))[:500]}..."
            for i, doc in enumerate(documents)
        ])
        
        return f"""You are an expert at evaluating document relevance to search queries.

Query: "{query}"

Documents to evaluate:
{docs_text}

For each document, provide a relevance score from 0.0 to 1.0 and a brief reasoning.
Format your response as a JSON array with objects containing:
- "index": document number (1-based)
- "score": relevance score (0.0-1.0)
- "reasoning": brief explanation of relevance
- "confidence": confidence in your assessment (0.0-1.0)

Example response format:
[
  {{"index": 1, "score": 0.9, "reasoning": "Highly relevant", "confidence": 0.95}},
  {{"index": 2, "score": 0.3, "reasoning": "Somewhat related", "confidence": 0.8}}
]

Respond only with valid JSON:"""

    def _create_relevance_ranking_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create prompt for relevance ranking"""
        docs_text = "\n".join([
            f"{i+1}. {doc.get('content', str(doc))[:500]}..."
            for i, doc in enumerate(documents)
        ])
        
        return f"""Rank these documents by relevance to the query.

Query: "{query}"

Documents:
{docs_text}

Rank the documents from most relevant (1) to least relevant ({len(documents)}).
Provide a JSON array with objects containing:
- "index": document number (1-based)
- "rank": ranking position (1 = most relevant)
- "reasoning": why this ranking
- "confidence": confidence in ranking (0.0-1.0)

Respond only with valid JSON:"""

    def _create_hybrid_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create hybrid prompt combining scoring and ranking"""
        docs_text = "\n".join([
            f"{i+1}. {doc.get('content', str(doc))[:500]}..."
            for i, doc in enumerate(documents)
        ])
        
        return f"""Evaluate and rank these documents for relevance to the query.

Query: "{query}"

Documents:
{docs_text}

For each document, provide:
1. A relevance score (0.0-1.0)
2. A ranking position (1 = most relevant)
3. Brief reasoning
4. Confidence level (0.0-1.0)

Format as JSON array with objects containing:
- "index": document number (1-based)
- "score": relevance score (0.0-1.0)
- "rank": ranking position
- "reasoning": explanation
- "confidence": confidence level

Respond only with valid JSON:"""

    async def _process_batch(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        strategy: str = None
    ) -> List[RerankResult]:
        """Process a batch of documents for reranking"""
        try:
            prompt = self._create_rerank_prompt(query, documents, strategy)
            
            # Get LLM response
            response = await self.llm_func(prompt)
            
            # Parse JSON response
            try:
                results = json.loads(response.strip())
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM rerank response as JSON")
                # Fallback: return documents with default scores
                return [
                    RerankResult(
                        document=doc,
                        relevance_score=0.5,
                        reasoning="Fallback scoring",
                        confidence=0.5
                    )
                    for doc in documents
                ]
            
            # Convert results to RerankResult objects
            rerank_results = []
            for result in results:
                if isinstance(result, dict) and "index" in result:
                    doc_idx = result["index"] - 1  # Convert to 0-based index
                    if 0 <= doc_idx < len(documents):
                        rerank_results.append(RerankResult(
                            document=documents[doc_idx],
                            relevance_score=float(result.get("score", 0.5)),
                            reasoning=result.get("reasoning", ""),
                            confidence=float(result.get("confidence", 0.5))
                        ))
            
            # If parsing failed, return fallback results
            if not rerank_results:
                return [
                    RerankResult(
                        document=doc,
                        relevance_score=0.5,
                        reasoning="Fallback scoring",
                        confidence=0.5
                    )
                    for doc in documents
                ]
            
            return rerank_results
            
        except Exception as e:
            logger.error(f"Error in batch reranking: {e}")
            # Return fallback results
            return [
                RerankResult(
                    document=doc,
                    relevance_score=0.5,
                    reasoning=f"Error in reranking: {str(e)}",
                    confidence=0.3
                )
                for doc in documents
            ]

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        strategy: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using LLM-based approach.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top results to return
            strategy: Reranking strategy to use
            **kwargs: Additional arguments
            
        Returns:
            List of reranked documents with scores
        """
        if not documents:
            return documents
        
        # Check cache
        if self.cache_enabled:
            cache_key = self._generate_cache_key(query, documents)
            if cache_key in self.cache:
                logger.debug("Using cached rerank results")
                cached_results = self.cache[cache_key]
                if top_k:
                    cached_results = cached_results[:top_k]
                return cached_results
        
        # Process documents in batches
        all_results = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            
            # Process batch
            batch_results = await self._process_batch(query, batch, strategy)
            all_results.extend(batch_results)
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Convert to document format with scores
        reranked_docs = []
        for result in all_results:
            doc = result.document.copy()
            doc["rerank_score"] = result.relevance_score
            doc["rerank_reasoning"] = result.reasoning
            doc["rerank_confidence"] = result.confidence
            reranked_docs.append(doc)
        
        # Apply top_k if specified
        if top_k:
            reranked_docs = reranked_docs[:top_k]
        
        # Cache results
        if self.cache_enabled:
            cache_key = self._generate_cache_key(query, documents)
            self.cache[cache_key] = reranked_docs
        
        return reranked_docs

    async def rerank_with_explanation(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        strategy: str = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Rerank documents and provide explanation of the reranking process.
        
        Returns:
            Tuple of (reranked_documents, explanation)
        """
        # Create explanation prompt
        explanation_prompt = f"""Explain why these documents were reranked for the query: "{query}"

Consider factors like:
- Semantic relevance
- Information completeness
- Query intent matching
- Document quality

Provide a brief explanation of the reranking logic:"""
        
        # Get reranked documents
        reranked_docs = await self.rerank(query, documents, top_k, strategy, **kwargs)
        
        # Get explanation
        try:
            explanation = await self.llm_func(explanation_prompt)
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")
            explanation = "Reranking completed based on semantic relevance to the query."
        
        return reranked_docs, explanation

    def clear_cache(self):
        """Clear the reranking cache"""
        self.cache.clear()
        logger.info("Reranking cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "cache_enabled": self.cache_enabled,
            "strategy": self.strategy,
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent
        }


class AdaptiveLLMReranker(LLMReranker):
    """
    Adaptive LLM-based reranker that automatically selects the best strategy
    based on query type and document characteristics.
    """
    
    def __init__(self, llm_func: Callable, **kwargs):
        super().__init__(llm_func, **kwargs)
        self.strategy_selector = self._create_strategy_selector()
    
    def _create_strategy_selector(self) -> Callable:
        """Create strategy selection function"""
        def select_strategy(query: str, documents: List[Dict[str, Any]]) -> str:
            # Simple heuristics for strategy selection
            query_lower = query.lower()
            
            # Check for specific query types
            if any(word in query_lower for word in ["how", "why", "explain", "describe"]):
                return "semantic_scoring"  # Better for explanatory queries
            
            if any(word in query_lower for word in ["compare", "difference", "similar"]):
                return "hybrid"  # Better for comparative queries
            
            if len(documents) > 10:
                return "relevance_ranking"  # More efficient for large document sets
            
            return "semantic_scoring"  # Default strategy
        
        return select_strategy
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        strategy: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Rerank with adaptive strategy selection"""
        if strategy is None:
            strategy = self.strategy_selector(query, documents)
            logger.debug(f"Selected reranking strategy: {strategy}")
        
        return await super().rerank(query, documents, top_k, strategy, **kwargs)


# Convenience functions for easy integration
async def create_llm_reranker(
    llm_func: Callable,
    adaptive: bool = True,
    **kwargs
) -> LLMReranker:
    """Create an LLM-based reranker"""
    if adaptive:
        return AdaptiveLLMReranker(llm_func, **kwargs)
    else:
        return LLMReranker(llm_func, **kwargs)


async def llm_rerank_func(
    query: str,
    documents: List[Dict[str, Any]],
    llm_func: Callable,
    top_k: Optional[int] = None,
    strategy: str = "semantic_scoring",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function for LLM-based reranking.
    
    Args:
        query: Search query
        documents: Documents to rerank
        llm_func: LLM function to use
        top_k: Number of top results
        strategy: Reranking strategy
        **kwargs: Additional arguments
        
    Returns:
        Reranked documents
    """
    reranker = LLMReranker(llm_func, strategy=strategy)
    return await reranker.rerank(query, documents, top_k, **kwargs) 