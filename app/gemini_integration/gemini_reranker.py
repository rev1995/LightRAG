"""
Gemini LLM-Based Reranking Implementation
Advanced relevance scoring using Gemini models for better retrieval results
"""

import os
import asyncio
import json
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai import types

# Import from local LightRAG
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LightRAG"))

from lightrag.utils import logger


class GeminiReranker:
    """LLM-based reranking using Gemini models"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 2000
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Initialize Gemini client
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        
        logger.info(f"✅ Initialized Gemini Reranker: {model}")
    
    async def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using Gemini LLM for relevance scoring
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of top documents to return (None for all)
            
        Returns:
            List of reranked documents with relevance scores
        """
        
        if not documents:
            return []
        
        if len(documents) == 1:
            # Single document - just add a default score
            documents[0]["rerank_score"] = 1.0
            return documents
        
        try:
            # Prepare documents for scoring
            doc_texts = []
            for i, doc in enumerate(documents):
                # Extract text content from document
                content = self._extract_document_content(doc)
                doc_texts.append({
                    "id": i,
                    "content": content[:2000]  # Truncate for token limits
                })
            
            # Generate relevance scores using LLM
            scores = await self._score_documents_with_llm(query, doc_texts)
            
            # Apply scores to documents
            scored_documents = []
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = scores.get(i, 0.0)
                scored_documents.append(doc_copy)
            
            # Sort by relevance score (descending)
            scored_documents.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            
            # Apply top_n filtering
            if top_n is not None:
                scored_documents = scored_documents[:top_n]
            
            logger.info(f"Reranked {len(documents)} documents, returning top {len(scored_documents)}")
            
            return scored_documents
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Return original documents if reranking fails
            return documents[:top_n] if top_n else documents
    
    async def _score_documents_with_llm(
        self,
        query: str,
        doc_texts: List[Dict[str, Any]]
    ) -> Dict[int, float]:
        """Score documents using LLM reasoning"""
        
        # Prepare prompt for document scoring
        prompt = self._build_scoring_prompt(query, doc_texts)
        
        try:
            # Generate scoring response
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.generate_content(
                    prompt,
                    generation_config=types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens
                    )
                )
            )
            
            # Parse scores from response
            scores = self._parse_scoring_response(response.text, len(doc_texts))
            
            return scores
            
        except Exception as e:
            logger.error(f"LLM scoring failed: {str(e)}")
            # Return uniform scores if LLM fails
            return {i: 0.5 for i in range(len(doc_texts))}
    
    def _build_scoring_prompt(self, query: str, doc_texts: List[Dict[str, Any]]) -> str:
        """Build prompt for document relevance scoring"""
        
        prompt = f"""
You are an expert document relevance assessor. Your task is to score the relevance of documents to a given query.

Query: "{query}"

Please analyze each document and assign a relevance score from 0.0 to 1.0, where:
- 1.0 = Highly relevant, directly answers the query
- 0.8 = Very relevant, contains important related information
- 0.6 = Moderately relevant, has some useful information
- 0.4 = Somewhat relevant, tangentially related
- 0.2 = Low relevance, minimal connection
- 0.0 = Not relevant, no useful connection

Documents to score:

"""
        
        for doc in doc_texts:
            prompt += f"""
Document {doc['id']}:
{doc['content']}

---
"""
        
        prompt += """
Please provide your scoring in the following JSON format:
{
    "reasoning": "Brief explanation of your scoring approach",
    "scores": {
        "0": score_for_document_0,
        "1": score_for_document_1,
        ...
    }
}

Focus on:
1. Direct relevance to the query
2. Quality and specificity of information
3. Completeness of the answer provided
4. Factual accuracy and reliability

Provide scores as numbers between 0.0 and 1.0.
"""
        
        return prompt
    
    def _parse_scoring_response(self, response_text: str, num_docs: int) -> Dict[int, float]:
        """Parse LLM response to extract document scores"""
        
        try:
            # Try to extract JSON from response
            response_text = response_text.strip()
            
            # Find JSON block
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[start_idx:end_idx]
            parsed = json.loads(json_text)
            
            # Extract scores
            scores = {}
            if "scores" in parsed:
                for doc_id_str, score in parsed["scores"].items():
                    try:
                        doc_id = int(doc_id_str)
                        score_val = float(score)
                        
                        # Clamp score to valid range
                        score_val = max(0.0, min(1.0, score_val))
                        scores[doc_id] = score_val
                        
                    except (ValueError, TypeError):
                        continue
            
            # Fill missing scores with default
            for i in range(num_docs):
                if i not in scores:
                    scores[i] = 0.5  # Default middle score
            
            logger.info(f"Parsed scores for {len(scores)} documents")
            return scores
            
        except Exception as e:
            logger.error(f"Failed to parse scoring response: {str(e)}")
            
            # Fallback: try to extract numbers from response
            return self._fallback_score_extraction(response_text, num_docs)
    
    def _fallback_score_extraction(self, response_text: str, num_docs: int) -> Dict[int, float]:
        """Fallback method to extract scores from response text"""
        
        import re
        
        scores = {}
        
        # Look for patterns like "Document 0: 0.8" or "0: 0.8"
        patterns = [
            r'[Dd]ocument\s*(\d+)[:\s]+([0-1]\.?\d*)',
            r'(\d+)[:\s]+([0-1]\.?\d*)',
            r'([0-1]\.?\d+)'  # Just look for decimal numbers
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text)
            
            if len(matches) >= num_docs:
                for i, match in enumerate(matches[:num_docs]):
                    try:
                        if len(match) == 2:  # (doc_id, score)
                            score = float(match[1])
                        else:  # just score
                            score = float(match[0] if isinstance(match, str) else match)
                        
                        scores[i] = max(0.0, min(1.0, score))
                        
                    except (ValueError, TypeError):
                        continue
                
                if len(scores) >= num_docs:
                    break
        
        # Fill any missing scores
        for i in range(num_docs):
            if i not in scores:
                scores[i] = 0.5
        
        return scores
    
    def _extract_document_content(self, document: Dict[str, Any]) -> str:
        """Extract text content from document object"""
        
        # Try different possible content fields
        content_fields = [
            "content", "text", "body", "chunk_content", 
            "document_content", "passage", "snippet"
        ]
        
        for field in content_fields:
            if field in document and document[field]:
                return str(document[field])
        
        # If no content field found, try to stringify the whole document
        if isinstance(document, dict):
            # Filter out metadata fields and concatenate text values
            text_parts = []
            skip_fields = {"id", "score", "embedding", "metadata", "rerank_score"}
            
            for key, value in document.items():
                if key not in skip_fields and isinstance(value, str) and len(value) > 10:
                    text_parts.append(value)
            
            if text_parts:
                return " ".join(text_parts)
        
        # Last resort - stringify the document
        return str(document)[:1000]


# LightRAG-compatible reranking function
async def gemini_llm_rerank(
    query: str,
    documents: List[Dict[str, Any]], 
    top_n: Optional[int] = None,
    model: str = "gemini-2.0-flash",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    LightRAG-compatible Gemini reranking function
    
    Args:
        query: Search query
        documents: Documents to rerank
        top_n: Number of top documents to return
        model: Gemini model to use
        **kwargs: Additional parameters
        
    Returns:
        Reranked documents with relevance scores
    """
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found, skipping reranking")
        return documents[:top_n] if top_n else documents
    
    # Get model from environment or use parameter
    model = os.getenv("RERANK_MODEL", model)
    
    # Create reranker instance
    reranker = GeminiReranker(api_key=api_key, model=model)
    
    # Perform reranking
    return await reranker.rerank_documents(query, documents, top_n, **kwargs)


# Alternative: Lightweight scoring function for faster reranking
async def gemini_fast_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    top_n: Optional[int] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Faster reranking using simpler prompting
    
    This is a lighter-weight alternative that uses a simpler prompt
    for faster processing when you have many documents.
    """
    
    if not documents:
        return []
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return documents[:top_n] if top_n else documents
    
    try:
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel("gemini-2.0-flash")
        
        # Simple scoring prompt
        doc_summaries = []
        for i, doc in enumerate(documents[:20]):  # Limit for speed
            content = str(doc.get("content", doc))[:200]  # Short snippets
            doc_summaries.append(f"{i}: {content}")
        
        prompt = f"""
Query: {query}

Rank these document snippets by relevance (0-10 scale):
{chr(10).join(doc_summaries)}

Return only numbers: doc_id:score, doc_id:score, ...
"""
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.generate_content(
                prompt,
                generation_config=types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=500
                )
            )
        )
        
        # Parse simple response
        scores = {}
        for line in response.text.split(','):
            if ':' in line:
                try:
                    doc_id, score = line.strip().split(':', 1)
                    scores[int(doc_id)] = float(score) / 10.0  # Normalize to 0-1
                except:
                    continue
        
        # Apply scores and sort
        for i, doc in enumerate(documents):
            doc["rerank_score"] = scores.get(i, 0.5)
        
        documents.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        return documents[:top_n] if top_n else documents
        
    except Exception as e:
        logger.error(f"Fast reranking failed: {str(e)}")
        return documents[:top_n] if top_n else documents 