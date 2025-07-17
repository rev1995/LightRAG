import os
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs
import asyncio

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # 1. Initialize the GenAI Client with your Gemini API Key
    client = genai.Client(api_key=gemini_api_key)

    # 2. Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        # Each msg is expected to be a dict: {"role": "...", "content": "..."}
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"

    # 3. Call the Gemini model
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[combined_prompt],
        config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1),
    )

    # 4. Return the response text
    return response.text

@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
async def embedding_func(texts: list[str]) -> np.ndarray:
    client = genai.Client(api_key=gemini_api_key)
    
    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        # Extract the actual embedding values from ContentEmbedding object
        embedding_values = response.embeddings[0].values
        embeddings.append(embedding_values)
    
    return np.array(embeddings)