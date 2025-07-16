#!/usr/bin/env python3
"""
Robust LLM-based Reranking Implementation

This module provides a more robust implementation of LLM-based reranking
with better error handling, rate limiting, and fallback mechanisms.
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RerankStrategy(Enum):
    """Reranking strategies"""
    SEMANTIC_SCORING = "semantic_scoring"
    RELEVANCE_RANKING = "relevance_ranking"
    HYBRID = "hybrid"


@dataclass
class RerankConfig:
    """Configuration for LLM reranking"""
    max_retries: int = 3
    retry_delay: float = 1.0
    max_concurrent: int = 3
    batch_size: int = 5
    timeout: float = 30.0
    enable_cache: bool = True
    strategy: RerankStrategy = RerankStrategy.SEMANTIC_SCORING


class RobustLLMReranker:
    """Robust LLM-based reranker with error handling and rate limiting"""
    
    def __init__(self, llm_func, config: RerankConfig):
        self.llm_func = llm_func
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.cache = {} if config.enable_cache else None
        
    async def rerank(
        self, 
        query: str, 
        documents: List[Dict], 
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Rerank documents using LLM with robust error handling"""
        
        if not documents:
            return []
        
        # Limit to top_k if specified
        if top_k:
            documents = documents[:top_k]
        
        try:
            # Try to get cached results first
            cache_key = self._get_cache_key(query, documents)
            if self.cache and cache_key in self.cache:
                logger.debug("Using cached rerank results")
                return self.cache[cache_key]
            
            # Process documents in batches
            reranked_docs = await self._process_batches(query, documents)
            
            # Cache results if enabled
            if self.cache:
                self.cache[cache_key] = reranked_docs
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed, returning original documents: {e}")
            return documents[:top_k] if top_k else documents
    
    async def _process_batches(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Process documents in batches with error handling"""
        
        batches = [
            documents[i:i + self.config.batch_size] 
            for i in range(0, len(documents), self.config.batch_size)
        ]
        
        all_results = []
        
        for batch_idx, batch in enumerate(batches):
            try:
                logger.debug(f"Processing rerank batch {batch_idx + 1}/{len(batches)}")
                
                # Process batch with semaphore for rate limiting
                async with self.semaphore:
                    batch_results = await self._process_single_batch(query, batch)
                    all_results.extend(batch_results)
                    
                    # Add delay between batches to avoid rate limiting
                    if batch_idx < len(batches) - 1:
                        await asyncio.sleep(0.5)
                        
            except Exception as e:
                logger.warning(f"Batch {batch_idx + 1} failed, using original order: {e}")
                all_results.extend(batch)
        
        return all_results
    
    async def _process_single_batch(self, query: str, batch: List[Dict]) -> List[Dict]:
        """Process a single batch of documents"""
        
        for attempt in range(self.config.max_retries):
            try:
                # Create prompt based on strategy
                prompt = self._create_rerank_prompt(query, batch)
                
                # Call LLM with timeout
                response = await asyncio.wait_for(
                    self.llm_func(prompt),
                    timeout=self.config.timeout
                )
                
                # Parse response
                parsed_results = self._parse_rerank_response(response, batch)
                
                if parsed_results:
                    return parsed_results
                else:
                    logger.warning(f"Failed to parse rerank response on attempt {attempt + 1}")
                    
            except asyncio.TimeoutError:
                logger.warning(f"Rerank timeout on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Rerank error on attempt {attempt + 1}: {e}")
            
            # Wait before retry
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        # If all attempts failed, return original batch
        logger.warning("All rerank attempts failed, returning original order")
        return batch
    
    def _create_rerank_prompt(self, query: str, documents: List[Dict]) -> str:
        """Create rerank prompt based on strategy"""
        
        docs_text = "\n\n".join([
            f"Document {i+1}: {doc.get('content', '')[:500]}..."
            for i, doc in enumerate(documents)
        ])
        
        if self.config.strategy == RerankStrategy.SEMANTIC_SCORING:
            return f"""Please rerank the following documents based on their relevance to the query.

Query: {query}

Documents:
{docs_text}

Please provide your response as a JSON array with the following format:
[
  {{"index": 0, "score": 0.95, "reasoning": "This document is highly relevant because..."}},
  {{"index": 1, "score": 0.75, "reasoning": "This document is moderately relevant because..."}},
  ...
]

Score each document from 0.0 to 1.0 based on relevance to the query. Higher scores indicate better relevance."""

        elif self.config.strategy == RerankStrategy.RELEVANCE_RANKING:
            return f"""Please rank the following documents by relevance to the query.

Query: {query}

Documents:
{docs_text}

Please provide your response as a JSON array with the following format:
[
  {{"index": 0, "rank": 1, "reasoning": "Most relevant because..."}},
  {{"index": 1, "rank": 2, "reasoning": "Second most relevant because..."}},
  ...
]

Rank documents from 1 (most relevant) to {len(documents)} (least relevant)."""

        else:  # HYBRID
            return f"""Please analyze and rerank the following documents based on their relevance to the query.

Query: {query}

Documents:
{docs_text}

Please provide your response as a JSON array with the following format:
[
  {{"index": 0, "score": 0.95, "rank": 1, "reasoning": "Most relevant with high confidence because..."}},
  {{"index": 1, "score": 0.75, "rank": 2, "reasoning": "Moderately relevant because..."}},
  ...
]

Provide both a relevance score (0.0-1.0) and a rank (1-{len(documents)}) for each document."""
    
    def _parse_rerank_response(self, response: str, documents: List[Dict]) -> List[Dict]:
        """Parse LLM rerank response with robust error handling"""
        
        try:
            # Try to extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON array found in response")
                return []
            
            json_str = response[json_start:json_end]
            parsed_data = json.loads(json_str)
            
            if not isinstance(parsed_data, list):
                logger.warning("Response is not a JSON array")
                return []
            
            # Apply reranking based on parsed data
            reranked_docs = []
            for item in parsed_data:
                if isinstance(item, dict) and 'index' in item:
                    doc_index = item.get('index')
                    if 0 <= doc_index < len(documents):
                        doc = documents[doc_index].copy()
                        
                        # Add rerank metadata
                        doc['rerank_score'] = item.get('score', 0.0)
                        doc['rerank_rank'] = item.get('rank', doc_index + 1)
                        doc['rerank_reasoning'] = item.get('reasoning', '')
                        doc['rerank_confidence'] = min(item.get('score', 0.0), 1.0)
                        
                        reranked_docs.append(doc)
            
            # Sort by rank or score
            if self.config.strategy == RerankStrategy.RELEVANCE_RANKING:
                reranked_docs.sort(key=lambda x: x.get('rerank_rank', float('inf')))
            else:
                reranked_docs.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
            
            return reranked_docs
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return []
        except Exception as e:
            logger.warning(f"Failed to parse rerank response: {e}")
            return []
    
    def _get_cache_key(self, query: str, documents: List[Dict]) -> str:
        """Generate cache key for rerank results"""
        import hashlib
        
        # Create a hash of query and document contents
        content = query + "".join([doc.get('content', '')[:100] for doc in documents])
        return hashlib.md5(content.encode()).hexdigest()


# Convenience function for creating a robust reranker
def create_robust_llm_reranker(
    llm_func,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    max_concurrent: int = 3,
    batch_size: int = 5,
    timeout: float = 30.0,
    enable_cache: bool = True,
    strategy: str = "semantic_scoring"
) -> RobustLLMReranker:
    """Create a robust LLM reranker with the specified configuration"""
    
    config = RerankConfig(
        max_retries=max_retries,
        retry_delay=retry_delay,
        max_concurrent=max_concurrent,
        batch_size=batch_size,
        timeout=timeout,
        enable_cache=enable_cache,
        strategy=RerankStrategy(strategy)
    )
    
    return RobustLLMReranker(llm_func, config) 