# Rerank Integration Guide

LightRAG supports reranking functionality to improve retrieval quality by re-ordering documents based on their relevance to the query. Reranking is now controlled per query via the `enable_rerank` parameter (default: True).

## Quick Start

### Environment Variables

Set these variables in your `.env` file or environment for rerank configuration:

```bash
# Rerank configuration (required when enable_rerank=True in queries)
ENABLE_RERANK=true
# Gemini API key is used for reranking
GEMINI_API_KEY=your_gemini_api_key_here
```

### Programmatic Configuration

```python
from lightrag import LightRAG, QueryParam
from lightrag.rerank import gemini_llm_rerank, RerankModel

# Method 1: Using Gemini LLM rerank function with all settings included
async def my_rerank_func(query: str, documents: list, top_k: int = None, **kwargs):
    return await gemini_llm_rerank(
        query=query,
        documents=documents,
        api_key="your_gemini_api_key_here",
        model="gemini-2.0-flash",  # Default model
        top_k=top_k or 10,  # Handle top_k within the function
        temperature=0.0,  # Lower temperature for more consistent scoring
        **kwargs
    )

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=your_llm_func,
    embedding_func=your_embedding_func,
    rerank_model_func=my_rerank_func,  # Configure rerank function
)

# Query with rerank enabled (default)
result = await rag.aquery(
    "your query",
    param=QueryParam(enable_rerank=True)  # Control rerank per query
)

# Query with rerank disabled
result = await rag.aquery(
    "your query",
    param=QueryParam(enable_rerank=False)
)

# Method 2: Using RerankModel wrapper
rerank_model = RerankModel(
    rerank_func=gemini_llm_rerank,
    kwargs={
        "api_key": "your_gemini_api_key_here",
        "model": "gemini-2.0-flash",
        "temperature": 0.0,
    }
)

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=your_llm_func,
    embedding_func=your_embedding_func,
    rerank_model_func=rerank_model.rerank,
)

# Control rerank per query
result = await rag.aquery(
    "your query",
    param=QueryParam(
        enable_rerank=True,  # Enable rerank for this query
        chunk_top_k=5       # Number of chunks to keep after reranking
    )
)
```

## Supported Provider

### Gemini LLM Reranking

LightRAG now exclusively uses Gemini LLM for reranking:

```python
from lightrag.rerank import gemini_llm_rerank

# Using Gemini LLM for reranking
result = await gemini_llm_rerank(
    query="your query",
    documents=documents,
    api_key="your_gemini_api_key",
    model="gemini-2.0-flash",  # Default model
    temperature=0.0,  # Lower temperature for more consistent scoring
    top_k=10
)
```

## Integration Points

Reranking is automatically applied at these key retrieval stages:

1. **Naive Mode**: After vector similarity search in `_get_vector_context`
2. **Local Mode**: After entity retrieval in `_get_node_data`
3. **Global Mode**: After relationship retrieval in `_get_edge_data`
4. **Hybrid/Mix Modes**: Applied to all relevant components

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_rerank` | bool | False | Enable/disable reranking |
| `rerank_model_func` | callable | None | Custom rerank function containing all configurations (model, API keys, top_k, etc.) |

## Example Usage

### Basic Usage

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_complete, gemini_embedding
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.rerank import gemini_llm_rerank

async def my_rerank_func(query: str, documents: list, top_k: int = None, **kwargs):
    """Gemini LLM rerank function with all settings included"""
    return await gemini_llm_rerank(
        query=query,
        documents=documents,
        api_key="your_gemini_api_key_here",
        model="gemini-2.0-flash",
        temperature=0.0,
        top_k=top_k or 10,  # Default top_k if not provided
        **kwargs
    )

async def main():
    # Initialize with rerank enabled
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gemini_complete,
        embedding_func=gemini_embedding,
        rerank_model_func=my_rerank_func,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    # Insert documents
    await rag.ainsert([
        "Document 1 content...",
        "Document 2 content...",
    ])

    # Query with rerank (automatically applied)
    result = await rag.aquery(
        "Your question here",
        param=QueryParam(enable_rerank=True)  # This top_k is passed to rerank function
    )

    print(result)

asyncio.run(main())
```

### Direct Rerank Usage

```python
from lightrag.rerank import gemini_llm_rerank

async def test_rerank():
    documents = [
        {"content": "Text about topic A"},
        {"content": "Text about topic B"},
        {"content": "Text about topic C"},
    ]

    reranked = await gemini_llm_rerank(
        query="Tell me about topic A",
        documents=documents,
        api_key="your_gemini_api_key_here",
        model="gemini-2.0-flash",
        temperature=0.0,
        top_k=2
    )

    for doc in reranked:
        print(f"Score: {doc.get('rerank_score')}, Content: {doc.get('content')}")
```

## Best Practices

1. **Self-Contained Functions**: Include all necessary configurations (API keys, models, top_k handling) within your rerank function
2. **Performance**: Use reranking selectively for better performance vs. quality tradeoff
3. **API Limits**: Monitor API usage and implement rate limiting within your rerank function
4. **Fallback**: Always handle rerank failures gracefully (returns original results)
5. **Top-k Handling**: Handle top_k parameter appropriately within your rerank function
6. **Cost Management**: Consider rerank API costs in your budget planning

## Troubleshooting

### Common Issues

1. **API Key Missing**: Ensure API keys are properly configured within your rerank function
2. **Network Issues**: Check API endpoints and network connectivity
3. **Model Errors**: Verify the rerank model name is supported by your API
4. **Document Format**: Ensure documents have `content` or `text` fields

### Debug Mode

Enable debug logging to see rerank operations:

```python
import logging
logging.getLogger("lightrag.rerank").setLevel(logging.DEBUG)
```

### Error Handling

The rerank integration includes automatic fallback:

```python
# If rerank fails, original documents are returned
# No exceptions are raised to the user
# Errors are logged for debugging
```

## Gemini LLM Reranking

The Gemini LLM reranking function uses a prompt-based approach to score documents based on their relevance to the query. The function:

1. Sends a prompt to Gemini asking it to rate each document's relevance to the query on a scale of 0-10
2. Parses the scores from Gemini's response
3. Sorts documents by their relevance scores
4. Returns the top-k most relevant documents

This approach leverages Gemini's understanding of language and context to provide high-quality reranking without requiring additional external APIs.
