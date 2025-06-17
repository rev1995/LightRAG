import streamlit as st
import requests
import json
import time

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
INGEST_ENDPOINT = f"{API_URL}/ingest"
QUERY_ENDPOINT = f"{API_URL}/query"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="LightRAG Production Chatbot",
    page_icon="🤖",
    layout="wide",
)

# --- Helper Functions ---
def check_api_health():
    """Checks if the backend API is reachable."""
    try:
        response = requests.get(f"{API_URL}/docs", timeout=2)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

# --- Main Application UI ---
st.title("🤖 LightRAG Production Chatbot")
st.caption("Interacting with a Gemini & Neo4j Powered RAG System")

if not check_api_health():
    st.error(
        "**Connection Error:** Cannot reach the LightRAG API server. "
        "Please ensure the `api_server.py` is running and accessible."
    )
    st.stop()

# --- Sidebar for Configuration and Actions ---
with st.sidebar:
    st.header("Chat Session Controls")
    
    ## NEW ##: New Chat button to reset the session
    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.chat_settings_locked = False
        st.rerun()

    st.divider()
    st.header("Configuration & Actions")

    if st.button("🚀 Ingest Documents", help="Triggers the backend to scan and process all markdown files."):
        with st.spinner("Sending ingestion request... Check server logs for progress."):
            try:
                response = requests.post(INGEST_ENDPOINT)
                if response.status_code == 200:
                    st.success("Ingestion process started!")
                    st.info(response.json()["message"])
                else:
                    st.error(f"Failed to start ingestion: {response.text}")
            except requests.ConnectionError as e:
                st.error(f"Connection failed: {e}")

    st.divider()
    
    st.subheader("Query Parameters")
    
    ## NEW ##: Initialize lock state
    if "chat_settings_locked" not in st.session_state:
        st.session_state.chat_settings_locked = False
    
    # Query Mode Selection
    query_mode = st.selectbox(
        "Query Mode",
        options=["mix", "hybrid", "local", "global", "naive", "bypass"],
        index=0,
        help="**Mix**: Best quality (KG + Vector). **Hybrid**: KG only. **Naive**: Vector only.",
        key="query_mode",
        disabled=st.session_state.chat_settings_locked ## NEW ##: Disable if chat has started
    )

    # Streaming Toggle
    is_streaming = st.toggle(
        "Enable Streaming",
        value=True,
        help="Receive the response token-by-token.",
        key="is_streaming",
        disabled=st.session_state.chat_settings_locked ## NEW ##: Disable if chat has started
    )
    
    user_prompt = st.text_area(
        "Response Formatting Prompt (Optional)",
        placeholder="e.g., 'Respond as a helpful expert.'",
        key="user_prompt",
        disabled=st.session_state.chat_settings_locked ## NEW ##: Disable if chat has started
    )
    if st.session_state.chat_settings_locked:
        st.info("Chat settings are locked. Start a 'New Chat' to change them.")

# --- Chat Interface ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

## NEW ##: Function to handle the API call and response generation
def run_query():
    # Lock the settings on the first message
    st.session_state.chat_settings_locked = True
    
    # Prepare and display the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Use settings from session state
        api_request_payload = {
            "query": st.session_state.messages[-1]["content"], # Use the last user message
            "mode": st.session_state.query_mode,
            "stream": st.session_state.is_streaming,
            "conversation_history": [
                msg for msg in st.session_state.messages[:-1] # History is everything *before* the last prompt
            ],
            "user_prompt": st.session_state.user_prompt if st.session_state.user_prompt else None,
        }

        try:
            if st.session_state.is_streaming:
                with st.spinner("Assistant is thinking..."):
                    with requests.post(QUERY_ENDPOINT, json=api_request_payload, stream=True) as r:
                        if r.status_code != 200:
                            st.error(f"Error from API: {r.status_code} - {r.text}")
                            return # Stop execution on error
                        for line in r.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8')
                                if decoded_line.startswith('data:'):
                                    try:
                                        content = json.loads(decoded_line[5:])
                                        if "chunk" in content:
                                            full_response += content["chunk"]
                                            message_placeholder.markdown(full_response + "▌")
                                        elif "error" in content:
                                            st.error(f"Streaming error: {content['error']}")
                                            break
                                    except json.JSONDecodeError:
                                        st.warning(f"Could not decode line: {decoded_line}")
                message_placeholder.markdown(full_response)
            else: # Non-streaming
                with st.spinner("Assistant is thinking..."):
                    response = requests.post(QUERY_ENDPOINT, json=api_request_payload)
                    if response.status_code == 200:
                        full_response = response.json().get("response", "No response found.")
                        message_placeholder.markdown(full_response)
                    else:
                        st.error(f"Error from API: {response.status_code} - {response.text}")
                        full_response = f"Error: Status {response.status_code}"
        
        except requests.exceptions.RequestException as e:
            st.error(f"Connection to API server failed: {e}")
            full_response = "Error: Could not connect to the backend."
            message_placeholder.markdown(full_response)
            
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})

## MODIFIED ##: Handle user input and regenerate button
# Place regenerate button logic outside the main input block
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    if st.button("🔄 Regenerate Response"):
        # Remove the last user prompt and assistant response
        st.session_state.messages.pop() # Remove assistant response
        # We re-add the user prompt before running the query again
        run_query()
        st.rerun()

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    run_query()
    st.rerun() # Rerun to display the new messages and regenerate button