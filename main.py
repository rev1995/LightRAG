import os
import asyncio
import logging
from pathlib import Path
import hashlib
import dataclasses
import requests

# Ensure the logger is configured
from lightrag.utils import setup_logger, EmbeddingFunc, Tokenizer
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status

# --- Dependency Imports ---
import numpy as np
import google.generativeai as genai
import sentencepiece as spm

# --- Configuration ---
# Setup logging to see the process flow
setup_logger("lightrag", level="INFO")
log = logging.getLogger("lightrag")

# The working directory for LightRAG's cache and other files
WORKING_DIR = "./rag_gemini_neo4j_storage"

# The directory containing your markdown files
DATA_DIR = "./data"

# --- Custom Tokenizer (from lightrag_gemini_demo_no_tiktoken.py) ---
# This avoids the dependency on OpenAI's tiktoken library.
class GemmaTokenizer(Tokenizer):
    @dataclasses.dataclass(frozen=True)
    class _TokenizerConfig:
        tokenizer_model_url: str
        tokenizer_model_hash: str

    _TOKENIZERS = {
        "google/gemma2": _TokenizerConfig(
            tokenizer_model_url="https://raw.githubusercontent.com/google/gemma_pytorch/33b652c465537c6158f9a472ea5700e5e770ad3f/tokenizer/tokenizer.model",
            tokenizer_model_hash="61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
        ),
    }

    def __init__(self, tokenizer_dir: str):
        tokenizer_name = "google/gemma2"
        file_url = self._TOKENIZERS[tokenizer_name].tokenizer_model_url
        tokenizer_model_name = file_url.rsplit("/", 1)[1]
        expected_hash = self._TOKENIZERS[tokenizer_name].tokenizer_model_hash

        file_path = Path(tokenizer_dir) / tokenizer_model_name
        model_data = self._maybe_load_from_cache(file_path=file_path, expected_hash=expected_hash)

        if not model_data:
            model_data = self._load_from_url(file_url=file_url, expected_hash=expected_hash)
            self._save_tokenizer_to_cache(cache_path=file_path, model_data=model_data)

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.LoadFromSerializedProto(model_data)
        super().__init__(model_name="gemma2", tokenizer=tokenizer)

    def _is_valid_model(self, model_data: bytes, expected_hash: str) -> bool:
        return hashlib.sha256(model_data).hexdigest() == expected_hash

    def _maybe_load_from_cache(self, file_path: Path, expected_hash: str) -> bytes | None:
        if not file_path.is_file():
            return None
        with open(file_path, "rb") as f:
            content = f.read()
        if self._is_valid_model(model_data=content, expected_hash=expected_hash):
            return content
        self._maybe_remove_file(file_path)
        return None

    def _load_from_url(self, file_url: str, expected_hash: str) -> bytes:
        resp = requests.get(file_url)
        resp.raise_for_status()
        content = resp.content
        if not self._is_valid_model(model_data=content, expected_hash=expected_hash):
            raise ValueError("Downloaded model file is corrupted.")
        return content

    @staticmethod
    def _save_tokenizer_to_cache(cache_path: Path, model_data: bytes) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(model_data)
        except OSError:
            pass

    @staticmethod
    def _maybe_remove_file(file_path: Path) -> None:
        if file_path.is_file():
            try:
                file_path.unlink()
            except OSError:
                pass


# --- LLM and Embedding Functions ---

# Gemini LLM Function
async def gemini_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    if history_messages is None:
        history_messages = []
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_history = [{"role": "user" if msg["role"] == "user" else "model", "parts": [{"text": msg["content"]}]} for msg in history_messages]
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = await model.generate_content_async(full_prompt, generation_config=genai.types.GenerationConfig(temperature=0.1))
        return response.text
    except Exception as e:
        log.error(f"Error calling Gemini API: {e}")
        return "Error: Could not get a response from the LLM."

# NEW: Gemini Embedding Function
async def gemini_embedding_func(texts: list[str]) -> np.ndarray:
    """
    Generates embeddings using the Google Generative AI API.
    """
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        # Using the recommended model for text embeddings
        result = await genai.embed_content_async(
            model="models/embedding-001",
            content=texts,
            task_type="retrieval_document" # Use "retrieval_query" for queries
        )
        return np.array(result['embedding'])
    except Exception as e:
        log.error(f"Error calling Gemini Embedding API: {e}")
        # Return a zero vector of the correct shape on failure
        # to avoid crashing the whole process.
        return np.zeros((len(texts), 768))

# --- RAG System Initialization ---

async def initialize_rag_system() -> LightRAG:
    """
    Initializes the LightRAG instance with Gemini LLM, Gemini Embeddings, and Neo4j storage.
    """
    log.info("Initializing LightRAG with Gemini and Neo4j...")
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    tokenizer_path = os.path.join(WORKING_DIR, "gemma_tokenizer")
    custom_tokenizer = GemmaTokenizer(tokenizer_dir=tokenizer_path)

    rag = LightRAG(
        # --- Storage Configuration ---
        working_dir=WORKING_DIR,
        graph_storage="Neo4JStorage",

        # --- LLM and Embedding Injection (UPDATED) ---
        llm_model_func=gemini_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,  # Dimension for 'models/embedding-001'
            max_token_size=8192,
            func=gemini_embedding_func, # Use the new Gemini embedding function
        ),
        
        # --- Tokenizer ---
        tokenizer=custom_tokenizer,

        # --- Caching and Performance ---
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    log.info("LightRAG system initialized successfully.")
    return rag

# --- Document Processing ---

async def process_markdown_files(rag_instance: LightRAG, data_directory: str):
    """
    Recursively finds all markdown files in a directory and ingests them into LightRAG.
    """
    log.info(f"Scanning for markdown files in '{data_directory}'...")
    data_path = Path(data_directory)
    md_files = list(data_path.rglob("*.md"))

    if not md_files:
        log.warning(f"No markdown files found in '{data_directory}'.")
        return

    log.info(f"Found {len(md_files)} markdown files to process.")
    
    # Ingest documents in batches for better performance and resource management
    batch_size = 5
    for i in range(0, len(md_files), batch_size):
        batch_files = md_files[i:i+batch_size]
        contents = []
        ids = []
        for file_path in batch_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    contents.append(content)
                    ids.append(str(file_path))
                else:
                    log.warning(f"Skipping empty file: {file_path}")
            except Exception as e:
                log.error(f"Failed to read file {file_path}: {e}")
        
        if contents:
            log.info(f"Inserting batch of {len(contents)} documents...")
            await rag_instance.ainsert(contents, ids=ids)

# --- Main Execution (with Dynamic Conversational History) ---

async def main():
    """
    Main function to set up, process data, and query the RAG system with generic analytical questions.
    This version now uses a dynamic conversation history built from actual RAG responses.
    """
    # 1. Check for required environment variables
    if not all(os.getenv(key) for key in ["GEMINI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]):
        log.error("Missing one or more required environment variables. Please check your .env file.")
        return

    rag_system = await initialize_rag_system()

    # A helper function to print streaming responses
    async def print_stream(stream):
        log.info("LightRAG (streaming) > ")
        full_response = ""
        async for chunk in stream:
            print(chunk, end="", flush=True)
            full_response += chunk
        print() # for a newline after the stream
        return full_response

    try:
        # 2. Ingest all documents from the data directory
        await process_markdown_files(rag_system, DATA_DIR)

        log.info("\n--- 🚀 Starting Generic Analytical Queries ---")

        # --- Query 1: High-Level Thematic Analysis ---
        log.info("\n--- Query 1: Thematic Analysis (Hybrid Mode, Multiple Paragraphs) ---")
        query1_text = "What are the main themes and key topics in the provided text?"
        log.info(f"User > {query1_text}")
        response1 = await rag_system.aquery(
            query1_text,
            param=QueryParam(mode="hybrid", response_type="Multiple Paragraphs")
        )
        log.info(f"LightRAG > {response1}")

        # --- Query 2: Dynamic Contextual Follow-up using a REAL RAG response ---
        log.info("\n--- Query 2: Dynamic Contextual Follow-up (With Real RAG History) ---")
        # Step 2a: Ask the initial question to establish context.
        # We must ensure this response is not streamed so we can capture the full text.
        query2_part1_text = "What are the primary subjects or entities discussed in the text?"
        log.info(f"(Part 1) User > {query2_part1_text}")
        response2_part1 = await rag_system.aquery(query2_part1_text, param=QueryParam(mode="hybrid", stream=False))
        log.info(f"(Part 1) LightRAG > {response2_part1}")

        # Step 2b: Now, create the history dynamically using the actual response we just received.
        log.info("--- Creating dynamic history from the previous turn ---")
        dynamic_history = [
            {"role": "user", "content": query2_part1_text},
            {"role": "assistant", "content": response2_part1} # Using the actual response
        ]

        # Step 2c: Ask the follow-up question, which relies on the dynamically created history.
        query2_part2_text = "Tell me more about the first one mentioned."
        log.info(f"(Part 2) User > {query2_part2_text} (with real history)")
        response2_part2 = await rag_system.aquery(
            query2_part2_text,
            param=QueryParam(
                mode="hybrid",
                conversation_history=dynamic_history,
                history_turns=1
            )
        )
        log.info(f"(Part 2) LightRAG > {response2_part2}")

        # --- Query 3: Structured Data Extraction with Formatting Instructions ---
        log.info("\n--- Query 3: Structured Data Extraction (Mix Mode, Bullet Points) ---")
        query3_text = "Identify the main entities in the text and describe their primary attributes or characteristics."
        user_prompt_instruction = "Present the answer as a bulleted list. Each bullet point should be in the format: 'Entity Name: Key Attribute or Description'."
        log.info(f"User > {query3_text}")
        log.info(f"Instruction > {user_prompt_instruction}")
        response3 = await rag_system.aquery(
            query3_text,
            param=QueryParam(
                mode="mix",
                user_prompt=user_prompt_instruction,
                response_type="Bullet Points"
            )
        )
        log.info(f"LightRAG > \n{response3}")

        # --- Query 4: Caching and Streaming Demonstration ---
        log.info("\n--- Query 4: Caching and Streaming for a Comprehensive Summary ---")
        query4_text = "Provide a comprehensive summary of the entire document, including all key entities and their interactions."
        
        log.info("(First Call - Streaming Enabled)")
        log.info(f"User > {query4_text}")
        response4_streamed = await rag_system.aquery(query4_text, param=QueryParam(mode="hybrid", stream=True))
        await print_stream(response4_streamed)
        
        log.info("\n(Second Call - Hitting Cache, No Streaming)")
        log.info(f"User > {query4_text}")
        response4_cached = await rag_system.aquery(query4_text, param=QueryParam(mode="hybrid", stream=False))
        log.info(f"LightRAG (Cached) > {response4_cached}")

        # --- Query 5: Comparing Global vs. Local Search for Relationship Analysis ---
        log.info("\n--- Query 5: Comparing Global vs. Local Search for Relationship Analysis ---")
        query5_text = "Describe the relationship between the two most frequently mentioned entities in the text."
        log.info(f"User > {query5_text}")

        log.info("\n(Global Mode - focuses on high-level relationship themes from the Knowledge Graph)")
        response5_global = await rag_system.aquery(query5_text, param=QueryParam(mode="global"))
        log.info(f"LightRAG (Global) > {response5_global}")

        log.info("\n(Local Mode - focuses on specific entity details and their direct textual context)")
        response5_local = await rag_system.aquery(query5_text, param=QueryParam(mode="local"))
        log.info(f"LightRAG (Local) > {response5_local}")

    except Exception as e:
        log.error(f"An unexpected error occurred during the RAG process: {e}")
    finally:
        # 4. Important: Finalize storages to ensure data is saved correctly
        if rag_system:
            log.info("Finalizing storage connections...")
            await rag_system.finalize_storages()
            log.info("RAG system finalized.")

if __name__ == "__main__":
    asyncio.run(main())