import streamlit as st
import requests
import os
from typing import List

# --- Configuration ---
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:5000")
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload"
QUERY_ENDPOINT = f"{API_BASE_URL}/query"

# --- UI Setup ---
st.set_page_config(page_title="LightRAG Chat", layout="wide", initial_sidebar_state="expanded")
st.title("💡 LightRAG Chat Interface")
st.markdown("Interact with your documents through a powerful RAG pipeline with Knowledge Graph capabilities.")

# --- Sidebar for Controls and Uploading ---
with st.sidebar:
    st.header("⚙️ Controls & Settings")
    
    # File Uploader
    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files",
        accept_multiple_files=True,
        type=["txt", "md", "pdf"]
    )
    
    if st.button("Process Uploaded Files", use_container_width=True):
        if uploaded_files:
            files_to_send = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
            with st.spinner("Uploading and starting ingestion... This may take a moment. The server will process files in the background."):
                try:
                    response = requests.post(UPLOAD_ENDPOINT, files=files_to_send, timeout=10)
                    if response.status_code == 202:
                        st.success(response.json().get("message", "Files uploaded! Ingestion started."))
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection Error: Could not connect to the RAG server at {API_BASE_URL}. Is it running?")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please upload at least one file to process.")

    st.divider()

    # Query Controls
    st.subheader("2. Query Settings")
    query_mode = st.selectbox(
        "Select Query Mode:",
        ("hybrid", "mix", "local", "global", "naive", "bypass"),
        index=0,
        help="""
        - **hybrid**: Best default. Combines KG entities and relationships.
        - **mix**: Most comprehensive. Combines KG and raw text chunks.
        - **local**: Focuses on specific entities.
        - **global**: Focuses on high-level relationships.
        - **naive**: Simple vector search on text chunks.
        - **bypass**: Chat directly with the LLM without RAG.
        """
    )
    
    use_history = st.checkbox("Enable Conversation History", value=True)
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.info(f"History turns (server-side): {os.getenv('HISTORY_TURNS', 3)}")


# --- Chat Interface ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to UI and history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare and display assistant's streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare payload for the API
        payload = {
            "query": prompt,
            "mode": query_mode,
            "use_history": use_history,
            "history": st.session_state.messages[:-1] # Send all history except the current user prompt
        }

        try:
            # Call the query endpoint and stream the response
            with requests.post(QUERY_ENDPOINT, json=payload, stream=True, timeout=300) as r:
                r.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
        
        except requests.exceptions.RequestException:
            error_message = f"Connection Error: Could not connect to the RAG server. Please ensure it is running and accessible at {API_BASE_URL}."
            st.error(error_message)
            full_response = error_message
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            st.error(error_message)
            full_response = error_message

    # Add the final assistant response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})