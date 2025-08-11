"""
LLM-Based Reranking Implementation for LightRAG
Uses Gemini LLM for intelligent document reranking instead of external models.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

# Add LightRAG to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lightrag'))

from lightrag.utils import logger


@dataclass
class LLMRerankerConfig:
    """Configuration for LLM-based reranking"""
    model: str
    max_docs: int
    max_tokens_per_doc: int
    temperature: float
    enable_explanation: bool
    
    @classmethod
    def from_env(cls) -> "LLMRerankerConfig":
        """Create configuration from environment variables"""
        return cls(
            model=os.getenv("RERANK_LLM_MODEL", "gemini-2.0-flash"),
            max_docs=int(os.getenv("RERANK_MAX_DOCS", "20")),
            max_tokens_per_doc=int(os.getenv("RERANK_MAX_TOKENS_PER_DOC", "500")),
            temperature=float(os.getenv("RERANK_TEMPERATURE", "0.1")),
            enable_explanation=os.getenv("RERANK_ENABLE_EXPLANATION", "false").lower() == "true",
        )


class LLMReranker:
    """LLM-based document reranker using Gemini"""
    
    def __init__(self, config: Optional[LLMRerankerConfig] = None):
        self.config = config or LLMRerankerConfig.from_env()
        self.llm_func = None
        
    def set_llm_function(self, llm_func):
        """Set the LLM function to use for reranking"""
        self.llm_func = llm_func
    
    def _truncate_document(self, doc_text: str, max_tokens: int = 500) -> str:
        """Truncate document to fit within token limit (rough estimation)"""
        # Rough estimation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(doc_text) <= max_chars:
            return doc_text
        
        # Try to truncate at sentence boundary
        truncated = doc_text[:max_chars]
        last_sentence = truncated.rfind('.')
        
        if last_sentence > max_chars * 0.7:  # If we can keep at least 70% of content
            return truncated[:last_sentence + 1]
        else:
            return truncated + "..."
    
    def _create_reranking_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create a prompt for LLM-based reranking"""
        
        # Prepare documents for the prompt
        doc_texts = []
        for i, doc in enumerate(documents):
            content = doc.get('content', '') or doc.get('text', '') or str(doc)
            truncated_content = self._truncate_document(content, self.config.max_tokens_per_doc)
            doc_texts.append(f"Document {i+1}:\n{truncated_content}")
        
        documents_text = "\n\n".join(doc_texts)
        
        prompt = f"""You are a document reranking expert. Given a query and a list of documents, rank the documents by their relevance to the query from most relevant to least relevant.

Query: "{query}"

Documents:
{documents_text}

Instructions:
1. Analyze each document's relevance to the query
2. Consider semantic similarity, topical relevance, and information completeness
3. Rank documents from most relevant (1) to least relevant ({len(documents)})
4. Return ONLY a JSON array with the rankings

Expected output format:
[1, 3, 2, 4, 5] (where numbers represent the original document order, ranked by relevance)

Ranking:"""

        return prompt
    
    def _create_batch_reranking_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create a more efficient batch reranking prompt for larger document sets"""
        
        doc_summaries = []
        for i, doc in enumerate(documents):
            content = doc.get('content', '') or doc.get('text', '') or str(doc)
            # Create short summary for each document
            summary = self._truncate_document(content, 100)  # Very short summaries
            doc_summaries.append(f"{i+1}. {summary}")
        
        documents_text = "\n".join(doc_summaries)
        
        prompt = f"""Rank these document summaries by relevance to the query. Return only the ranking numbers.

Query: "{query}"

Document Summaries:
{documents_text}

Return format: [1, 3, 2, 4, 5, ...]
Ranking:"""

        return prompt
    
    async def _parse_ranking_response(self, response: str, num_docs: int) -> List[int]:
        """Parse the LLM response to extract ranking"""
        try:
            # Try to extract JSON array from response
            response = response.strip()
            
            # Look for JSON array pattern
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx + 1]
                ranking = json.loads(json_str)
                
                # Validate ranking
                if (isinstance(ranking, list) and 
                    len(ranking) == num_docs and 
                    set(ranking) == set(range(1, num_docs + 1))):
                    return [r - 1 for r in ranking]  # Convert to 0-based indexing
            
            # Fallback: try to extract numbers
            import re
            numbers = re.findall(r'\d+', response)
            if len(numbers) >= num_docs:
                ranking = [int(n) for n in numbers[:num_docs]]
                # Ensure all positions are represented
                if set(ranking) == set(range(1, num_docs + 1)):
                    return [r - 1 for r in ranking]
            
        except Exception as e:
            logger.warning(f"Failed to parse ranking response: {e}")
        
        # Fallback: return original order
        logger.warning("Using original document order as fallback")
        return list(range(num_docs))
    
    async def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using LLM
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of top documents to return (optional)
            
        Returns:
            Reranked list of documents
        """
        if not self.llm_func:
            logger.warning("LLM function not set for reranking, returning original order")
            return documents[:top_n] if top_n else documents
        
        if not documents:
            return documents
        
        # Limit documents to max_docs for efficiency
        if len(documents) > self.config.max_docs:
            logger.info(f"Limiting reranking to top {self.config.max_docs} documents")
            documents = documents[:self.config.max_docs]
        
        try:
            # Choose prompt based on document count
            if len(documents) <= 10:
                prompt = self._create_reranking_prompt(query, documents)
            else:
                prompt = self._create_batch_reranking_prompt(query, documents)
            
            # Get ranking from LLM
            logger.debug(f"Reranking {len(documents)} documents with LLM")
            
            response = await self.llm_func(
                prompt,
                temperature=self.config.temperature,
                max_output_tokens=2000,
            )
            
            if isinstance(response, str):
                # Parse ranking from response
                ranking_indices = await self._parse_ranking_response(response, len(documents))
                
                # Apply ranking
                reranked_docs = []
                for idx in ranking_indices:
                    if 0 <= idx < len(documents):
                        doc = documents[idx].copy()
                        # Add rerank score (higher score = more relevant)
                        doc['rerank_score'] = 1.0 - (ranking_indices.index(idx) / len(ranking_indices))
                        reranked_docs.append(doc)
                
                # Return top_n if specified
                result = reranked_docs[:top_n] if top_n else reranked_docs
                
                logger.info(f"LLM reranking completed: {len(documents)} -> {len(result)} documents")
                return result
                
            else:
                logger.warning("Invalid LLM response format for reranking")
                return documents[:top_n] if top_n else documents
                
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            # Return original order as fallback
            return documents[:top_n] if top_n else documents


# Global reranker instance
_llm_reranker = None

def get_llm_reranker() -> LLMReranker:
    """Get or create global LLM reranker instance"""
    global _llm_reranker
    if _llm_reranker is None:
        _llm_reranker = LLMReranker()
    return _llm_reranker


async def llm_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    top_n: Optional[int] = None,
    llm_func=None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    LightRAG compatible LLM reranking function
    
    Args:
        query: The search query
        documents: List of documents to rerank
        top_n: Number of top documents to return
        llm_func: LLM function to use for reranking
        **kwargs: Additional parameters
        
    Returns:
        Reranked list of documents
    """
    reranker = get_llm_reranker()
    
    if llm_func:
        reranker.set_llm_function(llm_func)
    
    return await reranker.rerank(query, documents, top_n)


def create_llm_rerank_function(llm_func):
    """
    Create a rerank function that uses the specified LLM function
    
    Args:
        llm_func: The LLM function to use for reranking
        
    Returns:
        Async rerank function compatible with LightRAG
    """
    async def rerank_with_llm(
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        return await llm_rerank(query, documents, top_n, llm_func, **kwargs)
    
    return rerank_with_llm


# Utility functions for testing and validation
async def test_llm_reranking(llm_func, test_query: str, test_docs: List[str]):
    """Test LLM reranking functionality"""
    
    # Convert to document format
    documents = [{"content": doc, "index": i} for i, doc in enumerate(test_docs)]
    
    try:
        reranker = LLMReranker()
        reranker.set_llm_function(llm_func)
        
        print(f"Testing LLM reranking with query: '{test_query}'")
        print(f"Original document order:")
        for i, doc in enumerate(documents):
            print(f"  {i+1}. {doc['content'][:50]}...")
        
        reranked = await reranker.rerank(test_query, documents)
        
        print(f"\nReranked document order:")
        for i, doc in enumerate(reranked):
            score = doc.get('rerank_score', 0)
            print(f"  {i+1}. (score: {score:.3f}) {doc['content'][:50]}...")
        
        return reranked
        
    except Exception as e:
        print(f"Test failed: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    print("LLM Reranker module loaded successfully")
    print("Use create_llm_rerank_function(llm_func) to create a rerank function for LightRAG") 