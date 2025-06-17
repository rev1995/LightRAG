# LightRAG Chatbot System: Developer Onboarding & Features Guide

This document provides a comprehensive overview of the features, configurations, and internal workings of the production-grade RAG chatbot system. It is intended for new developers to understand the architecture and logic from end to end.

## I. Core Architecture

The system is built on **LightRAG**, a modular Retrieval-Augmented Generation framework. It is powered by:
- **Language & Embedding Models**: Google **Gemini** (`gemini-1.5-flash` and `embedding-001`).
- **Knowledge Graph Storage**: **Neo4j**, for storing structured entities and their relationships.
- **API Server**: **FastAPI**, exposing the RAG capabilities through a robust API.
- **User Interface**: **Streamlit**, providing a dynamic and interactive chat experience.

The system operates in two main phases:
1.  **Data Ingestion**: It reads Markdown files, uses an LLM to perform structured extraction of entities and relationships, and populates both a vector database and the Neo4j Knowledge Graph.
2.  **Query Engine**: It receives user queries, uses a multi-modal retrieval strategy to gather relevant context, constructs a detailed prompt, and generates a final answer using an LLM.

## II. Query Configuration & API Usage

All query configurations are controlled by the client (e.g., the Streamlit UI) via the JSON payload sent to the `/query` endpoint.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`mode`** | `string` | `"mix"` | The retrieval strategy. **`"mix"` is the recommended production setting**. It combines the precision of a Knowledge Graph (KG) search with the broad semantic coverage of a traditional vector search. |
| **`stream`** | `boolean`| `True` | Determines the response type. `True` provides a token-by-token stream, ideal for responsive UIs. `False` provides a single, complete response string. **This setting is locked per chat session.** |
| **`conversation_history`** | `list` | `[]` | A list of `{"role": "user", ...}` and `{"role": "assistant", ...}` objects. The UI client sends the entire session history with each request. |
| **`history_turns`** | `int` | `3` | **(Backend Setting)** Despite receiving the full history, the LightRAG backend is hard-coded to only use the **last 3 user-assistant turns**. This is a crucial optimization to manage the LLM's context window, reduce costs, and keep the conversation focused. |
| **`user_prompt`**| `string`| `None` | An optional, high-level instruction to the LLM on how to style or format its final answer (e.g., "Respond in a professional tone"). This does not affect the data retrieval process. |

## III. Caching Mechanism: Speed and Efficiency

Caching is a critical performance feature. It is enabled for all LLM calls (ingestion, keyword extraction, and final response generation).

#### How It Works: Direct-Hit Caching

1.  **Cache Key Generation**: Before any LLM call, the system generates a unique MD5 hash from all inputs that define the request. This includes the prompt text, the `mode`, and the `stream` flag.
2.  **Cache Check**: The system looks up this hash key in its key-value cache store.
3.  **Cache Hit**: If the exact key is found, the stored response is returned instantly. The expensive LLM call is **skipped**. This provides a massive speed boost for repeated questions.
4.  **Cache Miss**: If the key is not found, the system proceeds with the full RAG pipeline and calls the LLM.
5.  **Cache Save**: The new response from the LLM is saved in the cache with the corresponding hash key, making the next identical request a cache hit.

#### Example: Caching Similar but Different Queries

**Question**: How does the cache handle `"What is project Alpha?"` vs. `"What is the project Alpha?"`?

**Answer**: It treats them as **two different queries**.

- The cache is a **direct-hit cache**, meaning it requires the inputs to be *exactly identical* to score a hit.
- The first query generates hash `H1` and results in a **Cache Miss**. The response is generated and stored as `cache[H1]`.
- The second query, though semantically identical, is a different string. It generates a different hash `H2` and also results in a **Cache Miss**. Its response is stored as `cache[H2]`.

This behavior is intentional for production safety, as it prevents the system from incorrectly assuming two subtly different questions are the same.

#### Critical Feature: Caching for Streaming vs. Non-Streaming

The system's `PatchedLightRAG` class correctly generates **different cache keys** for streaming and non-streaming requests. A request for `"What is project Alpha?"` with `stream=True` has a different hash than the same query with `stream=False`. This is essential to prevent the system from incorrectly returning a single, complete text block to a UI expecting a stream of tokens.

## IV. UI Feature: Session Settings Lock

To ensure a consistent and predictable user experience, the chat settings are **locked** for the duration of a single conversation.

#### How It Works

1.  **Initial State**: When a user opens the app, the chat is new. The configuration options in the sidebar (Query Mode, Enable Streaming, etc.) are **enabled**. The user can configure their desired settings.
2.  **First Message Sent**: As soon as the user sends their first message, a flag `st.session_state.chat_settings_locked` is set to `True`.
3.  **Locked State**: Once this flag is `True`, the configuration widgets in the sidebar are automatically set to `disabled=True`. They become read-only. A message appears informing the user that the settings are locked.
4.  **Consistency**: For every subsequent message in that conversation, the system will use the settings that were locked in at the beginning (which are stored in `st.session_state`). This prevents confusing behavior, like switching from a streaming to a non-streaming response mid-dialogue.
5.  **Resetting**: To change the settings, the user must explicitly click the **"➕ New Chat"** button. This action clears the conversation history and resets the `chat_settings_locked` flag to `False`, re-enabling the configuration widgets for the new session.

This feature is a key part of creating a stable and professional-grade user interface.

## V. Other Key UI Features

- **Regenerate Response**: A "Regenerate" button appears after the assistant replies. It allows the user to re-run their last query, which is useful for getting an alternative answer if the first one was unsatisfactory.
- **API Health Check**: On startup, the UI pings the FastAPI server's `/docs` endpoint to ensure it's running. It displays a clear error if the connection fails, guiding the user on how to resolve the issue.
- **Asynchronous Ingestion**: The "Ingest Documents" button triggers a background task on the server, allowing the UI to remain responsive while documents are processed.

## VI. Explanation of Roles in Conversation History

In conversational AI, the `role` field is used to structure the dialogue for the LLM, indicating who said what.

- **`role: "user"`**: Represents input from the human user. It's the prompt, question, or instruction that the model needs to respond to.
- **`role: "assistant"`**: Represents a previous response from the AI model itself. It is used to provide the model with a memory of its own past statements, which is essential for maintaining context and coherence in a conversation.
