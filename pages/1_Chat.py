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
            return True
        return False
    except:
        return False

def stream_rag_response(query: str, mode: str, history: list, response_type: str) -> Generator[str, None, None]:
    """Streams the response from the RAG API."""
    payload = {"query": query, "mode": mode, "history": history, "response_type": response_type}
    try:
        with requests.post(f"{API_BASE_URL}/query/stream", json=payload, stream=True, timeout=120) as r:
            if r.status_code != 200:
                error_detail = r.text
                yield f"Error: Failed to get response from server. Status: {r.status_code}. Detail: {error_detail}"
                return
            for chunk in r.iter_content(chunk_size=None):
                if chunk: yield chunk.decode('utf-8')
    except requests.exceptions.RequestException as e:
        yield f"Error connecting to the RAG server: {e}"

def start_new_chat():
    """Creates a new chat session."""
    chat_id = f"chat_{datetime.now().timestamp()}"
    st.session_state.current_chat_id = chat_id
    st.session_state.chats[chat_id] = {"name": "New Chat", "messages": []}

def switch_chat(chat_id: str):
    """Switches the view to a different chat."""
    st.session_state.current_chat_id = chat_id

def handle_regeneration():
    """Handles the logic for regenerating the last response."""
    chat_id = st.session_state.current_chat_id
    current_chat = st.session_state.chats[chat_id]
    if len(current_chat["messages"]) < 2:
        st.toast("Not enough history to regenerate.", icon="⚠️"); return
    last_user_prompt = current_chat["messages"][-2]['content']
    history_for_api = current_chat["messages"][:-2]
    current_chat["messages"] = current_chat["messages"][:-2]
    st.session_state.regenerate_info = {"prompt": last_user_prompt, "history": history_for_api}
    st.session_state.regenerate_in_progress = True

# --- UI Initialization ---
st.title("LightRAG 💬 - Conversational RAG")

if "chats" not in st.session_state: st.session_state.chats = {}
if "current_chat_id" not in st.session_state: st.session_state.current_chat_id = None
if "regenerate_in_progress" not in st.session_state: st.session_state.regenerate_in_progress = False
if not st.session_state.chats: start_new_chat()

# --- MODIFIED MINIMALIST SIDEBAR ---
with st.sidebar:
    # Top section with "New Chat" and "Server Status"
    col1, col2 = st.columns([3, 1])
    with col1:
        st.button("➕", on_click=start_new_chat, use_container_width=True, help="Start a New Chat")
    with col2:
        # Server status indicator with tooltip
        if check_server_status():
            st.button("🟢", use_container_width=True, help="Server is Online")
        else:
            st.button("🔴", use_container_width=True, help="Server is Offline")

    st.divider()

    # Chat history section with icons
    st.markdown("<h3 style='text-align: center; margin-bottom: 1rem;'>Chats</h3>", unsafe_allow_html=True)
    
    chat_container = st.container(height=300) # Scrollable container for chats
    with chat_container:
        sorted_chat_ids = sorted(st.session_state.chats.keys(), reverse=True)
        for chat_id in sorted_chat_ids:
            chat_name = st.session_state.chats[chat_id]["name"]
            # Use an icon and the chat name as a tooltip
            if st.button(f"💬", key=f"switch_{chat_id}", use_container_width=True, help=chat_name):
                 switch_chat(chat_id)


    st.divider()

    # Query Settings section (kept as is for clarity)
    st.subheader("Query Settings")
    mode_options = {"Hybrid (recommended)": "hybrid", "Naive (Vector Search)": "naive", "Local (Entity Focused)": "local", "Global (Relationship)": "global_", "Mix (KG + Vector)": "mix"}
    if "query_mode" not in st.session_state: st.session_state.query_mode = "hybrid"
    selected_display_mode = st.selectbox("Mode:", options=list(mode_options.keys()), index=list(mode_options.values()).index(st.session_state.query_mode))
    st.session_state.query_mode = mode_options[selected_display_mode]

    response_type_options = ["Multiple Paragraphs", "Single Paragraph", "Bullet Points", "JSON object"]
    if "response_type" not in st.session_state: st.session_state.response_type = "Multiple Paragraphs"
    selected_response_type = st.selectbox("Format:", options=response_type_options, index=response_type_options.index(st.session_state.response_type))
    st.session_state.response_type = selected_response_type
    
    st.divider()

    # Document upload section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload .txt, .md, or .pdf files", accept_multiple_files=True, type=["txt", "md", "pdf"], label_visibility="collapsed")
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
            st.success(f"✅ All files uploaded and are being indexed.")

# --- Main Chat Interface (no changes from previous version) ---
current_chat_id = st.session_state.current_chat_id
messages = st.session_state.chats[current_chat_id]["messages"]

for i, message in enumerate(messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and i == len(messages) - 1:
            st.button("🔄", key=f"regenerate_{i}", on_click=handle_regeneration, help="Regenerate this response")

if st.session_state.regenerate_in_progress:
    regen_info = st.session_state.regenerate_info
    st.chat_message("user").markdown(regen_info["prompt"])
    st.session_state.chats[current_chat_id]["messages"].append({"role": "user", "content": regen_info["prompt"]})
    
    with st.chat_message("assistant"):
        response_generator = stream_rag_response(regen_info["prompt"], st.session_state.query_mode, regen_info["history"], st.session_state.response_type)
        full_response = st.write_stream(response_generator)
        st.button("🔄", key="regenerate_new", on_click=handle_regeneration, help="Regenerate this response")
    
    st.session_state.chats[current_chat_id]["messages"].append({"role": "assistant", "content": full_response})
    st.session_state.regenerate_in_progress = False
    st.rerun()

if prompt := st.chat_input("Ask a question..."):
    if st.session_state.chats[current_chat_id]["name"] == "New Chat":
        st.session_state.chats[current_chat_id]["name"] = prompt[:30]

    st.session_state.chats[current_chat_id]["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        history_for_api = [msg for msg in messages]
        response_generator = stream_rag_response(prompt, st.session_state.query_mode, history_for_api, st.session_state.response_type)
        full_response = st.write_stream(response_generator)
    
    st.session_state.chats[current_chat_id]["messages"].append({"role": "assistant", "content": full_response})
    st.rerun()