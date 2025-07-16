# LLM-Based Reranking Execution Flow

## Overview

The LLM-based reranking in LightRAG is executed as part of the query processing pipeline. It uses the same LLM that's configured for generation, making it cost-effective and semantically consistent.

## Execution Flow

### 1. Query Initiation
```
User Query → LightRAG.aquery() → QueryParam.enable_rerank check
```

### 2. Document Retrieval Phase
```
Vector Search → Retrieve Documents → Apply Reranking (if enabled)
```

### 3. Reranking Integration Points

#### A. In `operate.py` - `apply_rerank_if_enabled()`
```python
async def apply_rerank_if_enabled(
    query: str,
    retrieved_docs: list[dict],
    global_config: dict,
    enable_rerank: bool = True,
    top_k: int = None,
) -> list[dict]:
```

**Execution Steps:**
1. **Check if reranking is enabled** (`enable_rerank` parameter)
2. **Get rerank function** from `global_config["rerank_model_func"]`
3. **Apply LLM-based reranking** using the configured function
4. **Return reranked documents** with scores and reasoning

#### B. In `operate.py` - `process_chunks_unified()`
```python
# 2. Apply reranking if enabled and query is provided
if query_param.enable_rerank and query and unique_chunks:
    rerank_top_k = query_param.chunk_top_k or len(unique_chunks)
    unique_chunks = await apply_rerank_if_enabled(
        query=query,
        retrieved_docs=unique_chunks,
        global_config=global_config,
        enable_rerank=query_param.enable_rerank,
        top_k=rerank_top_k,
    )
```

### 4. LLM-Based Reranking Implementation

#### A. Reranker Initialization (in `production_rag_pipeline.py`)
```python
async def _rerank_func(self, query: str, documents: List[Dict], top_k: int = None, **kwargs):
    """LLM-based rerank function using the same LLM for generation and reranking"""
    if not self.config.ENABLE_RERANK:
        return documents[:top_k] if top_k else documents
    
    try:
        # Use LLM-based reranking instead of external service
        llm_reranker = LLMReranker(
            llm_func=self._llm_model_func,
            batch_size=self.config.RERANK_BATCH_SIZE,
            max_concurrent=self.config.RERANK_MAX_CONCURRENT,
            cache_enabled=self.config.RERANK_CACHE_ENABLED,
            strategy=self.config.RERANK_STRATEGY
        )
        
        return await llm_reranker.rerank(
            query=query,
            documents=documents,
            top_k=top_k or self.config.CHUNK_TOP_K,
            **kwargs,
        )
    except Exception as e:
        self.logger.warning(f"LLM rerank failed, returning original documents: {e}")
        return documents[:top_k] if top_k else documents
```

#### B. LLM Reranker Core Logic (in `llm_rerank.py`)
```python
async def rerank(
    self,
    query: str,
    documents: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    strategy: str = None,
    **kwargs
) -> List[Dict[str, Any]]:
```

**Execution Steps:**
1. **Cache Check**: Look for cached reranking results
2. **Batch Processing**: Process documents in configurable batches
3. **Prompt Generation**: Create reranking prompt based on strategy
4. **LLM Call**: Use the same LLM for reranking
5. **Result Parsing**: Parse JSON response and extract scores
6. **Fallback Handling**: Graceful error handling with fallback results

### 5. Reranking Strategies

#### A. Semantic Scoring Strategy
```python
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

Respond only with valid JSON:"""
```

#### B. Relevance Ranking Strategy
```python
def _create_relevance_ranking_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
    """Create prompt for relevance ranking"""
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
```

### 6. Batch Processing

```python
# Process documents in batches
for i in range(0, len(documents), self.batch_size):
    batch = documents[i:i + self.batch_size]
    
    # Process batch
    batch_results = await self._process_batch(query, batch, strategy)
    all_results.extend(batch_results)
```

### 7. Caching Mechanism

```python
def _generate_cache_key(self, query: str, documents: List[Dict[str, Any]]) -> str:
    """Generate cache key for reranking results"""
    content_hash = hashlib.md5(
        json.dumps([doc.get('content', str(doc)) for doc in documents], sort_keys=True).encode()
    ).hexdigest()
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return f"rerank_{query_hash}_{content_hash}_{self.strategy}"
```

### 8. Error Handling and Fallback

```python
try:
    # Get LLM response
    response = await self.llm_func(prompt)
    
    # Parse JSON response
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
```

## Configuration Parameters

### Environment Variables
```bash
ENABLE_RERANK=true
RERANK_STRATEGY=semantic_scoring
RERANK_BATCH_SIZE=5
RERANK_MAX_CONCURRENT=3
RERANK_CACHE_ENABLED=true
```

### Query Parameters
```python
QueryParam(
    enable_rerank=True,  # Enable/disable reranking per query
    chunk_top_k=10,      # Number of chunks to keep after reranking
    mode="mix"           # Query mode that supports reranking
)
```

## Performance Optimizations

### 1. Caching
- **Cache Key**: Based on query hash + document content hash + strategy
- **Cache Hit**: Returns cached results without LLM call
- **Cache Miss**: Processes through LLM and caches results

### 2. Batch Processing
- **Configurable Batch Size**: Process multiple documents per LLM call
- **Concurrent Processing**: Multiple batches processed concurrently
- **Memory Efficiency**: Avoids loading all documents at once

### 3. Fallback Mechanisms
- **JSON Parsing Errors**: Returns documents with default scores
- **LLM Call Failures**: Returns original documents
- **Empty Results**: Uses original documents with warning

## Integration Points

### 1. LightRAG Core
```python
# In LightRAG.__post_init__()
if self.rerank_model_func:
    logger.info("Rerank model initialized for improved retrieval quality")
else:
    logger.warning("Rerank is enabled but no rerank_model_func provided. Reranking will be skipped.")
```

### 2. Query Processing
```python
# In operate.py - kg_query()
chunks = await process_chunks_unified(
    query=query,
    chunks=chunks,
    query_param=param,
    global_config=global_config,
    source_type="mixed"
)
```

### 3. Production Pipeline
```python
# In production_rag_pipeline.py
rag = LightRAG(
    rerank_model_func=self._rerank_func,  # LLM-based reranking
    # ... other configurations
)
```

## Monitoring and Logging

### 1. Debug Logs
```python
logger.debug(f"Applying rerank to {len(retrieved_docs)} documents, returning top {top_k}")
logger.debug(f"Rerank: {len(unique_chunks)} chunks (source: {source_type})")
```

### 2. Performance Metrics
```python
logger.info(f"Successfully reranked {len(retrieved_docs)} documents to {len(reranked_docs)}")
```

### 3. Error Handling
```python
logger.warning("Rerank returned empty results, using original documents")
logger.error(f"Error during reranking: {e}, using original documents")
```

## Benefits of LLM-Based Reranking

### 1. Cost-Effectiveness
- **Same LLM**: Uses existing LLM for both generation and reranking
- **No Additional APIs**: No need for external reranking services
- **Reduced Latency**: No network calls to external services

### 2. Semantic Consistency
- **Same Model**: Ensures consistent understanding between generation and reranking
- **Better Context**: LLM understands the full context of the query
- **Adaptive Scoring**: Can adapt to different query types and domains

### 3. Configurability
- **Multiple Strategies**: semantic_scoring, relevance_ranking, hybrid
- **Per-Query Control**: Enable/disable reranking per query
- **Batch Processing**: Configurable batch sizes and concurrency

### 4. Reliability
- **Fallback Mechanisms**: Graceful handling of errors
- **Caching**: Performance optimization for repeated queries
- **Error Recovery**: Continues processing even if reranking fails

## Example Execution Timeline

```
1. User submits query: "What is machine learning?"
2. LightRAG retrieves 20 documents from vector search
3. Reranking enabled: enable_rerank=True
4. LLM reranker processes documents in batches of 5
5. LLM evaluates relevance and returns scores
6. Top 10 documents selected based on scores
7. Final response generated using reranked documents
8. Results cached for future similar queries
```

This LLM-based reranking approach provides a reliable, efficient, and cost-effective solution that integrates seamlessly with the existing LightRAG pipeline while maintaining semantic consistency and providing excellent performance. 