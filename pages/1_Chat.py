# pages/1_Chat.py

import streamlit as st
import requests
from typing import Generator
from datetime import datetime

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
    """Streams the response from the RAG API."""
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

def start_new_chat():
    """Creates a new chat session."""
    # Generate a unique ID for the new chat using the current timestamp
    chat_id = f"chat_{datetime.now().timestamp()}"
    st.session_state.current_chat_id = chat_id
    st.session_state.chats[chat_id] = {
        "name": "New Chat",
        "messages": []
    }

def switch_chat(chat_id: str):
    """Switches the view to a different chat."""
    st.session_state.current_chat_id = chat_id

def handle_regeneration():
    """Handles the logic for regenerating the last response in the current chat."""
    chat_id = st.session_state.current_chat_id
    current_chat = st.session_state.chats[chat_id]
    
    if len(current_chat["messages"]) < 2:
        st.toast("Not enough history to regenerate.", icon="⚠️")
        return

    last_user_prompt = current_chat["messages"][-2]['content']
    history_for_api = current_chat["messages"][:-2]
    
    # Remove the last user-assistant pair to replace it
    current_chat["messages"] = current_chat["messages"][:-2]

    st.session_state.regenerate_info = {
        "prompt": last_user_prompt,
        "history": history_for_api
    }
    st.session_state.regenerate_in_progress = True

# --- UI Initialization ---
st.title("LightRAG 💬 - Conversational RAG")

# Initialize the main session state structure for multi-chat
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "regenerate_in_progress" not in st.session_state:
    st.session_state.regenerate_in_progress = False

# If there are no chats, create one automatically
if not st.session_state.chats:
    start_new_chat()

# --- Sidebar ---
with st.sidebar:
    st.header("Chats")
    
    # "New Chat" button at the top of the sidebar
    st.button("➕ New Chat", on_click=start_new_chat, use_container_width=True)
    
    st.divider()

    # Display list of chats for navigation
    # Sort chats by timestamp (newest first)
    sorted_chat_ids = sorted(st.session_state.chats.keys(), reverse=True)
    for chat_id in sorted_chat_ids:
        chat_name = st.session_state.chats[chat_id]["name"]
        if st.button(chat_name, key=chat_id, use_container_width=True):
            switch_chat(chat_id)

    st.divider()
    st.header("Controls")
    
    st.subheader("Server Status")
    status_ok, status_message = check_server_status()
    if status_ok: st.success(status_message)
    else: st.error(status_message)
    
    st.divider()

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload .txt, .md, or .pdf files", accept_multiple_files=True, type=["txt", "md", "pdf"])
    if uploaded_files:
        # ... (file upload logic remains the same)
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
    
    mode_options = {"Hybrid (recommended)": "hybrid", "Naive (Vector Search)": "naive", "Local (Entity Focused)": "local", "Global (Relationship)": "global_", "Mix (KG + Vector)": "mix"}
    if "query_mode" not in st.session_state: st.session_state.query_mode = "hybrid"
    selected_display_mode = st.selectbox("Query Mode:", options=list(mode_options.keys()), index=list(mode_options.values()).index(st.session_state.query_mode))
    st.session_state.query_mode = mode_options[selected_display_mode]

    response_type_options = ["Multiple Paragraphs", "Single Paragraph", "Bullet Points", "JSON object"]
    if "response_type" not in st.session_state: st.session_state.response_type = "Multiple Paragraphs"
    selected_response_type = st.selectbox("Response Format:", options=response_type_options, index=response_type_options.index(st.session_state.response_type))
    st.session_state.response_type = selected_response_type

# --- Main Chat Interface ---

# Get the message list for the currently active chat
current_chat_id = st.session_state.current_chat_id
messages = st.session_state.chats[current_chat_id]["messages"]

# Display existing messages from the current chat's history
for i, message in enumerate(messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and i == len(messages) - 1:
            st.button("🔄", key=f"regenerate_{i}", on_click=handle_regeneration, help="Regenerate this response")

# Handle regeneration flow
if st.session_state.regenerate_in_progress:
    regen_info = st.session_state.regenerate_info
    
    # Re-display the user prompt that is being regenerated
    st.chat_message("user").markdown(regen_info["prompt"])
    st.session_state.chats[current_chat_id]["messages"].append({"role": "user", "content": regen_info["prompt"]})
    
    # Generate and display the new response
    with st.chat_message("assistant"):
        response_generator = stream_rag_response(
            regen_info["prompt"], 
            st.session_state.query_mode, 
            regen_info["history"],
            st.session_state.response_type
        )
        full_response = st.write_stream(response_generator)
        st.button("🔄", key="regenerate_new", on_click=handle_regeneration, help="Regenerate this response")
    
    st.session_state.chats[current_chat_id]["messages"].append({"role": "assistant", "content": full_response})
    
    st.session_state.regenerate_in_progress = False
    st.rerun()

# Handle new user input from the chat box
if prompt := st.chat_input("Ask a question..."):
    # If this is the first message in a "New Chat", name the chat
    if st.session_state.chats[current_chat_id]["name"] == "New Chat":
        st.session_state.chats[current_chat_id]["name"] = prompt[:30] # Use first 30 chars as name

    # Add user message to the current chat's history
    st.session_state.chats[current_chat_id]["messages"].append({"role": "user", "content": prompt})
    
    # Display the new user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        # History for API call is everything before the current user prompt in the current chat
        history_for_api = [msg for msg in messages] # Already includes the new user prompt
        
        response_generator = stream_rag_response(
            prompt, 
            st.session_state.query_mode, 
            history_for_api,
            st.session_state.response_type
        )
        full_response = st.write_stream(response_generator)
    
    # Add the full assistant response to the current chat's history
    st.session_state.chats[current_chat_id]["messages"].append({"role": "assistant", "content": full_response})
    st.rerun()