"""
LightRAG Gemini Streamlit Frontend
Using local LightRAG source code
"""

# Setup LightRAG path first
from setup_lightrag import setup_lightrag_path, verify_lightrag_import
setup_lightrag_path()
verify_lightrag_import()

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🤖 LightRAG Gemini Chat & Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/HKUDS/LightRAG',
        'Report a bug': 'https://github.com/HKUDS/LightRAG/issues',
        'About': "LightRAG with Gemini Integration - Advanced RAG System"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding: 2rem 1rem;
    }
    
    .stAlert {
        margin-bottom: 1rem;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .chat-message.user {
        background-color: #e8f4fd;
        border-left-color: #1f77b4;
    }
    
    .chat-message.assistant {
        background-color: #f0f8f0;
        border-left-color: #28a745;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your LightRAG assistant with Gemini integration. How can I help you today?"}
        ]
    
    # Token usage tracking
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "api_calls": 0
        }
    
    # Configuration
    if "config" not in st.session_state:
        st.session_state.config = {
            "lightrag_api_base": os.getenv("LIGHTRAG_API_BASE", "http://localhost:9621"),
            "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
            "llm_model": os.getenv("LLM_MODEL", "gemini-2.0-flash"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-004")
        }
    
    # API client
    if "api_client" not in st.session_state:
        from utils.api_client import LightRAGAPIClient
        st.session_state.api_client = LightRAGAPIClient(
            base_url=st.session_state.config["lightrag_api_base"]
        )

def main():
    """Main application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🚀 LightRAG Gemini")
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "📄 Navigate",
            [
                "💬 Chat Interface",
                "📊 Analytics Dashboard", 
                "🕸️ Graph Visualizer",
                "⚙️ Configuration",
                "📁 Document Manager"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📈 Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Session Tokens", 
                f"{st.session_state.token_usage['total_tokens']:,}"
            )
        with col2:
            st.metric(
                "API Calls", 
                st.session_state.token_usage['api_calls']
            )
        

        
        # System status
        st.markdown("---")
        st.subheader("🟢 System Status")
        
        # Check API connectivity
        try:
            status = st.session_state.api_client.check_health()
            if status:
                st.success("✅ LightRAG API Connected")
            else:
                st.error("❌ LightRAG API Disconnected")
        except:
            st.error("❌ LightRAG API Unreachable")
        
        # Check Gemini API key
        if st.session_state.config["gemini_api_key"]:
            st.success("✅ Gemini API Key Set")
        else:
            st.warning("⚠️ Gemini API Key Missing")
    
    # Main content area
    if page == "💬 Chat Interface":
        from streamlit_components.chat_interface import ChatInterface
        chat = ChatInterface()
        chat.render()
        
    elif page == "📊 Analytics Dashboard":
        from streamlit_components.monitoring_dashboard import MonitoringDashboard
        dashboard = MonitoringDashboard()
        dashboard.render()
        
    elif page == "🕸️ Graph Visualizer":
        from streamlit_components.graph_visualizer import GraphVisualizer
        visualizer = GraphVisualizer()
        visualizer.render()
        
    elif page == "⚙️ Configuration":
        from streamlit_components.configuration_panel import ConfigurationPanel
        config = ConfigurationPanel()
        config.render()
        
    elif page == "📁 Document Manager":
        from streamlit_components.document_manager import DocumentManager
        doc_manager = DocumentManager()
        doc_manager.render()

def check_setup():
    """Check if the application is properly set up"""
    
    issues = []
    
    # Check if LightRAG source is available
    lightrag_path = Path(__file__).parent.parent / "LightRAG"
    if not lightrag_path.exists():
        issues.append("❌ LightRAG source directory not found")
    
    # Check environment variables
    required_env_vars = ["GEMINI_API_KEY"]
    for var in required_env_vars:
        if not os.getenv(var):
            issues.append(f"❌ Environment variable {var} is not set")
    
    if issues:
        st.error("🚨 Setup Issues Detected:")
        for issue in issues:
            st.error(issue)
        
        st.info("""
        **Setup Instructions:**
        
        1. **LightRAG Source**: Ensure the LightRAG directory is in the parent directory
        2. **Environment Variables**: Set required variables in `.env` file:
           ```
           GEMINI_API_KEY=your_gemini_api_key_here
           LIGHTRAG_API_BASE=http://localhost:9621
           ```
        3. **Start LightRAG Server**: Run `python main_server.py` in the app directory
        """)
        
        return False
    
    return True

if __name__ == "__main__":
    # Check setup first
    if check_setup():
        main()
    else:
        st.stop() 