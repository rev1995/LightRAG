# streamlit_app.py

import streamlit as st
from PIL import Image
import os

# --- Configuration ---
# Use an environment variable for the icon, or a default
APP_ICON_PATH = os.getenv("APP_ICON_PATH", "🚀")

st.set_page_config(
    page_title="LightRAG Home",
    page_icon=APP_ICON_PATH,
    layout="centered"
)

st.title("Welcome to the LightRAG Portal 🚀")

st.markdown("""
This portal provides a suite of tools to interact with your custom Retrieval-Augmented Generation (RAG) system, powered by `LightRAG`.

Navigate using the sidebar on the left to access the different functionalities.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.header("💬 Chat with Documents")
    st.markdown("""
    Engage in a conversation with your documents. Upload your files, and the system will index them to build a knowledge base. Ask questions and receive context-aware, streaming answers.
    
    **Features:**
    - File uploads (.txt, .md, .pdf)
    - Selectable RAG query modes
    - Conversational history
    
    **Go to a page from here:**
    """)
    st.page_link("pages/1_Chat.py", label="Go to Chat Page", icon="💬")


with col2:
    st.header("📊 Monitor Metrics")
    st.markdown("""
    Keep an eye on the performance and cost of your RAG system. This dashboard provides a live view of the calls being made to the underlying Large Language Model (LLM).

    **Features:**
    - Real-time call log
    - Calls-per-minute chart
    - Filter by call purpose (Query, Ingestion, etc.)

    **Go to a page from here:**
    """)
    st.page_link("pages/2_Metrics_Dashboard.py", label="Go to Metrics Dashboard", icon="📊")


st.divider()

st.info("To get started, upload your documents on the **Chat** page and then start asking questions!", icon="💡")