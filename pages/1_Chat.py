# pages/1_Chat.py

import streamlit as st
import requests
from typing import Generator

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="LightRAG Chat",
    page_icon="💬",
    layout="centered"
)

# --- Helper Functions ---
def check_server_status():
    """Checks if the backend API server is online."""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=2)
        if response.status_code == 200 and response.json().get("rag_initialized"):
            return True, "Server is online and RAG system is ready."
        return False, "Server is online, but RAG system is not ready."
    except requests.ConnectionError:
        return False, "Server is offline. Please start the API server."
    except Exception as e:
        return False, f"An error occurred: {e}"

def stream_rag_response(query: str, mode: str, history: list, response_type: str) -> Generator[str, None, None]:
    """Streams the response from the RAG API, now including response_type."""
    payload = {
        "query": query,
        "mode": mode,
        "history": history,
        "response_type": response_type
    }
    try:
        with requests.post(f"{API_BASE_URL}/query/stream", json=payload, stream=True, timeout=120) as r:
            if r.status_code != 200:
                error_detail = r.text
                yield f"Error: Failed to get response from server. Status: {r.status_code}. Detail: {error_detail}"
                return
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    yield chunk.decode('utf-8')
    except requests.exceptions.RequestException as e:
        yield f"Error connecting to the RAG server: {e}"

def handle_regeneration():
    """Handles the logic for regenerating the last response."""
    if "messages" not in st.session_state or len(st.session_state.messages) < 2:
        st.toast("Not enough conversation history to regenerate.", icon="⚠️")
        return

    # The last message is the assistant's, the one before is the user's.
    last_user_prompt = st.session_state.messages[-2]['content']
    
    # The history is everything before the last user-assistant pair.
    st.session_state.history_for_api = st.session_state.messages[:-2]
    
    # Remove the last two messages (user prompt and assistant response) to replace them.
    # We will re-add the user prompt before displaying the new response.
    st.session_state.messages = st.session_state.messages[:-2]

    # Store the prompt and set the flag to trigger regeneration on the next run.
    st.session_state.last_prompt_for_regen = last_user_prompt
    st.session_state.regenerate_in_progress = True


# --- UI Initialization ---
st.title("LightRAG 💬 - Conversational RAG")
st.caption("Chat with your documents using a custom LightRAG setup.")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "regenerate_in_progress" not in st.session_state:
    st.session_state.regenerate_in_progress = False

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    
    st.subheader("Server Status")
    status_ok, status_message = check_server_status()
    if status_ok: st.success(status_message)
    else: st.error(status_message)
    if st.button("Refresh Status"): st.rerun()

    st.divider()

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload .txt, .md, or .pdf files", accept_multiple_files=True, type=["txt", "md", "pdf"])
    if uploaded_files:
        all_success = True
        for uploaded_file in uploaded_files:
            with st.spinner(f"Uploading '{uploaded_file.name}'..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    if response.status_code != 200:
                        st.error(f"❌ Failed to upload '{uploaded_file.name}': {response.text}"); all_success = False
                except Exception as e:
                    st.error(f"❌ An error occurred during upload: {e}"); all_success = False
        if all_success:
            st.success(f"✅ All files uploaded. They will be indexed in the background.")
    
    st.divider()

    st.subheader("Query Settings")
    
    mode_options = {"Hybrid (recommended)": "hybrid", "Naive (Vector Search Only)": "naive", "Local (Entity Focused)": "local", "Global (Relationship Focused)": "global_", "Mix (KG + Vector)": "mix"}
    if "query_mode" not in st.session_state: st.session_state.query_mode = "hybrid"
    selected_display_mode = st.selectbox("Query Mode:", options=list(mode_options.keys()), index=list(mode_options.values()).index(st.session_state.query_mode))
    st.session_state.query_mode = mode_options[selected_display_mode]

    response_type_options = ["Multiple Paragraphs", "Single Paragraph", "Bullet Points", "JSON object"]
    if "response_type" not in st.session_state: st.session_state.response_type = "Multiple Paragraphs"
    selected_response_type = st.selectbox("Response Format:", options=response_type_options, index=response_type_options.index(st.session_state.response_type), help="Instruct the LLM on how to format its final answer.")
    st.session_state.response_type = selected_response_type

    st.divider()

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---

# Display existing messages from history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # --- MODIFICATION: Add regenerate button ONLY for the last assistant message ---
        if message["role"] == "assistant" and i == len(st.session_state.messages) - 1:
            st.button("🔄", key=f"regenerate_{i}", on_click=handle_regeneration, help="Regenerate this response")

# This block handles the actual regeneration action after the button click
if st.session_state.regenerate_in_progress:
    # Get the prompt that was stored by the callback
    prompt_to_regenerate = st.session_state.last_prompt_for_regen
    
    # Add the user prompt back to the UI display
    st.session_state.messages.append({"role": "user", "content": prompt_to_regenerate})
    with st.chat_message("user"):
        st.markdown(prompt_to_regenerate)
    
    # Generate and display the new response
    with st.chat_message("assistant"):
        response_generator = stream_rag_response(
            prompt_to_regenerate, 
            st.session_state.query_mode, 
            st.session_state.history_for_api,
            st.session_state.response_type
        )
        full_response = st.write_stream(response_generator)
        st.button("🔄", key=f"regenerate_new", on_click=handle_regeneration, help="Regenerate this response")
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Reset the flag and rerun to finalize the display
    st.session_state.regenerate_in_progress = False
    st.rerun()

# Handle new user input from the chat box
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # History for API call is everything *before* the current user prompt
        history_for_api = [msg for msg in st.session_state.messages[:-1]]
        
        response_generator = stream_rag_response(
            prompt, 
            st.session_state.query_mode, 
            history_for_api,
            st.session_state.response_type
        )
        full_response = st.write_stream(response_generator)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()