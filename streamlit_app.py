#!/usr/bin/env python3
"""
LightRAG Gemini 2.0 Flash Streamlit Application
Comprehensive interface with configurable parameters, chat, knowledge graph visualization, and monitoring dashboards.
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from io import StringIO
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables
STREAMLIT_TITLE = os.getenv("STREAMLIT_TITLE", "LightRAG Gemini 2.0 Flash")
STREAMLIT_PAGE_ICON = os.getenv("STREAMLIT_PAGE_ICON", "🧠")
STREAMLIT_LAYOUT = os.getenv("STREAMLIT_LAYOUT", "wide")
STREAMLIT_SIDEBAR_STATE = os.getenv("STREAMLIT_SIDEBAR_STATE", "expanded")

# API Configuration
DEFAULT_API_BASE = f"http://{os.getenv('HOST', 'localhost')}:{os.getenv('PORT', '9621')}"
API_BASE = os.getenv("LIGHTRAG_API_BASE", DEFAULT_API_BASE)
API_REQUEST_TIMEOUT = int(os.getenv("API_REQUEST_TIMEOUT", "30"))
API_UPLOAD_TIMEOUT = int(os.getenv("API_UPLOAD_TIMEOUT", "60"))
API_RETRY_ATTEMPTS = int(os.getenv("API_RETRY_ATTEMPTS", "3"))
API_RETRY_DELAY = int(os.getenv("API_RETRY_DELAY", "1"))

# Default query parameters from environment
DEFAULT_QUERY_MODE = os.getenv("DEFAULT_QUERY_MODE", "mix")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
DEFAULT_CHUNK_TOP_K = int(os.getenv("DEFAULT_CHUNK_TOP_K", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("DEFAULT_MAX_OUTPUT_TOKENS", "2048"))
DEFAULT_ENABLE_RERANK = os.getenv("DEFAULT_ENABLE_RERANK", "true").lower() == "true"
DEFAULT_STREAM_RESPONSE = os.getenv("DEFAULT_STREAM_RESPONSE", "true").lower() == "true"
DEFAULT_HISTORY_TURNS = int(os.getenv("DEFAULT_HISTORY_TURNS", "5"))

# Knowledge Graph visualization settings
KG_DEFAULT_MAX_DEPTH = int(os.getenv("KG_DEFAULT_MAX_DEPTH", "3"))
KG_DEFAULT_MAX_NODES = int(os.getenv("KG_DEFAULT_MAX_NODES", "500"))
KG_LAYOUT_ALGORITHM = os.getenv("KG_LAYOUT_ALGORITHM", "spring_layout")
KG_NODE_SIZE = int(os.getenv("KG_NODE_SIZE", "20"))
KG_EDGE_WIDTH = int(os.getenv("KG_EDGE_WIDTH", "1"))
KG_NODE_COLOR = os.getenv("KG_NODE_COLOR", "lightblue")
KG_EDGE_COLOR = os.getenv("KG_EDGE_COLOR", "gray")
KG_FIGURE_HEIGHT = int(os.getenv("KG_FIGURE_HEIGHT", "600"))

# UI Settings
UI_AUTO_REFRESH_INTERVAL = int(os.getenv("UI_AUTO_REFRESH_INTERVAL", "30"))
UI_MAX_CHAT_HISTORY = int(os.getenv("UI_MAX_CHAT_HISTORY", "100"))
UI_PAGE_SIZE = int(os.getenv("UI_PAGE_SIZE", "50"))
UI_CHART_HEIGHT = int(os.getenv("UI_CHART_HEIGHT", "400"))
UI_MESSAGES_PER_PAGE = int(os.getenv("UI_MESSAGES_PER_PAGE", "20"))

# Configure Streamlit page
st.set_page_config(
    page_title=STREAMLIT_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state=STREAMLIT_SIDEBAR_STATE
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .status-healthy { color: #4caf50; }
    .status-unhealthy { color: #f44336; }
    .status-warning { color: #ff9800; }
    .config-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with environment defaults
def init_session_state():
    """Initialize session state with configurable defaults"""
    defaults = {
        "messages": [],
        "api_calls": [],
        "system_metrics": {},
        "query_mode": DEFAULT_QUERY_MODE,
        "top_k": DEFAULT_TOP_K,
        "chunk_top_k": DEFAULT_CHUNK_TOP_K,
        "temperature": DEFAULT_TEMPERATURE,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "enable_rerank": DEFAULT_ENABLE_RERANK,
        "stream_response": DEFAULT_STREAM_RESPONSE,
        "history_turns": DEFAULT_HISTORY_TURNS,
        "kg_max_depth": KG_DEFAULT_MAX_DEPTH,
        "kg_max_nodes": KG_DEFAULT_MAX_NODES,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Helper functions
def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Dict:
    """Make API request with retry logic and tracking"""
    url = f"{API_BASE}{endpoint}"
    start_time = time.time()
    
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            timeout = API_UPLOAD_TIMEOUT if files else API_REQUEST_TIMEOUT
            
            if method == "GET":
                response = requests.get(url, timeout=timeout)
            elif method == "POST":
                if files:
                    response = requests.post(url, data=data, files=files, timeout=timeout)
                else:
                    response = requests.post(url, json=data, timeout=timeout)
            elif method == "DELETE":
                response = requests.delete(url, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            duration = time.time() - start_time
            success = response.status_code < 400
            
            # Track API call
            st.session_state.api_calls.append({
                "timestamp": datetime.now(),
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "duration": duration,
                "success": success,
                "attempt": attempt + 1
            })
            
            if success:
                return response.json() if response.content else {}
            else:
                if attempt == API_RETRY_ATTEMPTS - 1:  # Last attempt
                    st.error(f"API Error {response.status_code}: {response.text}")
                return {"error": response.text}
                
        except Exception as e:
            duration = time.time() - start_time
            if attempt == API_RETRY_ATTEMPTS - 1:  # Last attempt
                st.session_state.api_calls.append({
                    "timestamp": datetime.now(),
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": 500,
                    "duration": duration,
                    "success": False,
                    "error": str(e),
                    "attempt": attempt + 1
                })
                st.error(f"Request failed after {API_RETRY_ATTEMPTS} attempts: {str(e)}")
                return {"error": str(e)}
            else:
                time.sleep(API_RETRY_DELAY)
    
    return {"error": "Max retry attempts exceeded"}

def get_system_health() -> Dict:
    """Get system health information"""
    return make_api_request("/health")

def upload_file_to_api(uploaded_file) -> Dict:
    """Upload file to API"""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    return make_api_request("/documents/upload", "POST", files=files)

def add_text_to_api(text: str, source: str = "Streamlit") -> Dict:
    """Add text to API"""
    data = {"text": text, "file_source": source}
    return make_api_request("/documents/text", "POST", data)

def query_api(query: str, **kwargs) -> Dict:
    """Query the API with configurable parameters"""
    data = {
        "query": query,
        "mode": kwargs.get("mode", st.session_state.query_mode),
        "top_k": kwargs.get("top_k", st.session_state.top_k),
        "chunk_top_k": kwargs.get("chunk_top_k", st.session_state.chunk_top_k),
        "temperature": kwargs.get("temperature", st.session_state.temperature),
        "max_output_tokens": kwargs.get("max_output_tokens", st.session_state.max_output_tokens),
        "enable_rerank": kwargs.get("enable_rerank", st.session_state.enable_rerank),
        "history_turns": kwargs.get("history_turns", st.session_state.history_turns),
    }
    
    endpoint = "/query/stream" if kwargs.get("stream", st.session_state.stream_response) else "/query"
    return make_api_request(endpoint, "POST", data)

def get_document_status() -> Dict:
    """Get document processing status"""
    return make_api_request("/documents")

def scan_documents() -> Dict:
    """Trigger document scan"""
    return make_api_request("/documents/scan", "POST")

def clear_cache() -> Dict:
    """Clear all caches"""
    return make_api_request("/documents/clear_cache", "POST")

def clear_documents() -> Dict:
    """Clear all documents"""
    return make_api_request("/documents", "DELETE")

def get_knowledge_graph(label: str, max_depth: int = None, max_nodes: int = None) -> Dict:
    """Get knowledge graph data"""
    max_depth = max_depth or st.session_state.kg_max_depth
    max_nodes = max_nodes or st.session_state.kg_max_nodes
    return make_api_request(f"/graph/graphs?label={label}&max_depth={max_depth}&max_nodes={max_nodes}")

# Header
st.markdown(f"""
<div class="main-header">
    <h1>{STREAMLIT_PAGE_ICON} {STREAMLIT_TITLE}</h1>
    <p>Production-Ready Knowledge Graph RAG System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Configuration
    st.subheader("🌐 API Settings")
    with st.expander("API Configuration", expanded=False):
        api_base = st.text_input("API Base URL", value=API_BASE)
        if api_base != API_BASE:
            API_BASE = api_base
            st.rerun()
    
    # Query Configuration  
    st.subheader("🎯 Query Parameters")
    with st.expander("Query Settings", expanded=True):
        st.session_state.query_mode = st.selectbox(
            "Query Mode",
            ["mix", "local", "global", "hybrid", "naive"],
            index=["mix", "local", "global", "hybrid", "naive"].index(st.session_state.query_mode),
            help="Select the retrieval strategy"
        )
        
        st.session_state.top_k = st.slider(
            "Top K (entities/relations)",
            min_value=1,
            max_value=100,
            value=st.session_state.top_k,
            help="Number of entities or relations retrieved from KG"
        )
        
        st.session_state.chunk_top_k = st.slider(
            "Chunk Top K",
            min_value=1,
            max_value=50,
            value=st.session_state.chunk_top_k,
            help="Maximum number of chunks to send to LLM"
        )
        
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Controls randomness in responses"
        )
        
        st.session_state.max_output_tokens = st.slider(
            "Max Output Tokens",
            min_value=256,
            max_value=8192,
            value=st.session_state.max_output_tokens,
            step=256,
            help="Maximum tokens in response"
        )
        
        st.session_state.history_turns = st.slider(
            "History Turns",
            min_value=0,
            max_value=20,
            value=st.session_state.history_turns,
            help="Number of conversation turns to remember"
        )
        
        st.session_state.enable_rerank = st.checkbox(
            "Enable Reranking",
            value=st.session_state.enable_rerank,
            help="Use LLM-based reranking for better results"
        )
        
        st.session_state.stream_response = st.checkbox(
            "Stream Response",
            value=st.session_state.stream_response,
            help="Stream responses in real-time"
        )
    
    # Knowledge Graph Settings
    st.subheader("🕸️ Knowledge Graph")
    with st.expander("Graph Settings", expanded=False):
        st.session_state.kg_max_depth = st.slider(
            "Max Depth",
            min_value=1,
            max_value=10,
            value=st.session_state.kg_max_depth,
            help="Maximum traversal depth in graph"
        )
        
        st.session_state.kg_max_nodes = st.slider(
            "Max Nodes",
            min_value=50,
            max_value=2000,
            value=st.session_state.kg_max_nodes,
            step=50,
            help="Maximum nodes to display"
        )
    
    # System Actions
    st.subheader("🔧 System Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Cache"):
            with st.spinner("Clearing cache..."):
                result = clear_cache()
                if "error" not in result:
                    st.success("Cache cleared!")
                else:
                    st.error(f"Failed: {result['error']}")
    
    # System Status
    st.subheader("📊 System Status")
    with st.spinner("Checking system health..."):
        health = get_system_health()
    
    if "error" not in health:
        status = health.get("status", "unknown")
        if status == "healthy":
            st.success("🟢 System Healthy")
        elif status == "degraded":
            st.warning("🟡 System Degraded")
        else:
            st.error("🔴 System Unhealthy")
        
        # Quick metrics
        perf = health.get("performance", {})
        if perf:
            st.metric("Uptime", f"{perf.get('uptime_seconds', 0)/3600:.1f}h")
            st.metric("Requests", perf.get('total_requests', 0))
            st.metric("Documents", perf.get('total_documents', 0))
    else:
        st.error("❌ API Unavailable")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Interface", "🕸️ Knowledge Graph", "📋 Document Dashboard", "📈 System Monitoring"])

# Tab 1: Enhanced Chat Interface
with tab1:
    st.header("💬 Intelligent Chat Interface")
    
    # Chat controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        current_mode = st.selectbox(
            "Quick Mode Selection",
            ["mix", "local", "global", "hybrid", "naive"],
            index=["mix", "local", "global", "hybrid", "naive"].index(st.session_state.query_mode),
            key="chat_mode"
        )
        if current_mode != st.session_state.query_mode:
            st.session_state.query_mode = current_mode
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col3:
        chat_temp = st.selectbox(
            "Temperature",
            [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
            index=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9].index(min([0.0, 0.1, 0.3, 0.5, 0.7, 0.9], key=lambda x: abs(x - st.session_state.temperature))),
            key="chat_temp"
        )
        if chat_temp != st.session_state.temperature:
            st.session_state.temperature = chat_temp
    
    # Display current configuration
    with st.expander("📋 Current Query Configuration", expanded=False):
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            st.write(f"**Mode:** {st.session_state.query_mode}")
            st.write(f"**Top K:** {st.session_state.top_k}")
            st.write(f"**Chunk Top K:** {st.session_state.chunk_top_k}")
            st.write(f"**Temperature:** {st.session_state.temperature}")
        with config_col2:
            st.write(f"**Max Tokens:** {st.session_state.max_output_tokens}")
            st.write(f"**History Turns:** {st.session_state.history_turns}")
            st.write(f"**Reranking:** {'✅' if st.session_state.enable_rerank else '❌'}")
            st.write(f"**Streaming:** {'✅' if st.session_state.stream_response else '❌'}")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history with pagination
    with chat_container:
        if st.session_state.messages:
            # Limit chat history display
            display_messages = st.session_state.messages[-UI_MAX_CHAT_HISTORY:]
            
            for message in display_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Show metadata for assistant messages
                    if message["role"] == "assistant" and "metadata" in message:
                        with st.expander("📊 Response Details", expanded=False):
                            metadata = message["metadata"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Mode:** {metadata.get('mode', 'N/A')}")
                                st.write(f"**Tokens:** {metadata.get('tokens', 'N/A')}")
                            with col2:
                                st.write(f"**Duration:** {metadata.get('duration', 'N/A')}s")
                                st.write(f"**Reranked:** {'✅' if metadata.get('reranked') else '❌'}")
                            with col3:
                                st.write(f"**Sources:** {metadata.get('sources', 0)}")
                                st.write(f"**Entities:** {metadata.get('entities', 0)}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    response = query_api(prompt)
                    duration = time.time() - start_time
                
                if "error" not in response:
                    assistant_response = response.get("response", "No response received")
                    st.markdown(assistant_response)
                    
                    # Prepare metadata
                    metadata = {
                        "mode": st.session_state.query_mode,
                        "duration": f"{duration:.2f}",
                        "reranked": st.session_state.enable_rerank,
                        "tokens": response.get("token_count", "N/A"),
                        "sources": len(response.get("sources", [])),
                        "entities": len(response.get("entities", []))
                    }
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": assistant_response,
                        "metadata": metadata
                    })
                else:
                    error_msg = f"Error: {response['error']}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Tab 2: Enhanced Knowledge Graph Visualizer
with tab2:
    st.header("🕸️ Knowledge Graph Visualizer")
    
    # KG controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kg_label = st.text_input("Entity/Label to explore", placeholder="Enter entity name...")
    with col2:
        kg_depth = st.slider("Depth", 1, 10, st.session_state.kg_max_depth)
    with col3:
        kg_nodes = st.slider("Max Nodes", 50, 2000, st.session_state.kg_max_nodes, step=50)
    with col4:
        layout_algo = st.selectbox(
            "Layout",
            ["spring_layout", "circular_layout", "kamada_kawai_layout", "random_layout"],
            index=0
        )
    
    if kg_label:
        with st.spinner("Loading knowledge graph..."):
            kg_data = get_knowledge_graph(kg_label, kg_depth, kg_nodes)
        
        if "error" not in kg_data and kg_data:
            # Build graph from API response
            G = nx.Graph()
            
            # Handle different API response formats
            if "nodes" in kg_data and "edges" in kg_data:
                # Standard format with nodes and edges
                for node in kg_data["nodes"]:
                    node_id = node.get("id", str(node))
                    G.add_node(node_id, **node)
                
                for edge in kg_data["edges"]:
                    source = edge.get("source", edge.get("from"))
                    target = edge.get("target", edge.get("to"))
                    if source and target:
                        G.add_edge(source, target, **edge)
            
            elif "entities" in kg_data or "relations" in kg_data:
                # LightRAG format with entities and relations
                entities = kg_data.get("entities", [])
                relations = kg_data.get("relations", [])
                
                # Add entities as nodes
                for entity in entities:
                    entity_name = entity.get("entity", entity.get("name", str(entity)))
                    G.add_node(entity_name, type="entity", **entity)
                
                # Add relations as edges
                for relation in relations:
                    source = relation.get("src_id", relation.get("source"))
                    target = relation.get("tgt_id", relation.get("target"))
                    if source and target:
                        G.add_edge(source, target, type="relation", **relation)
            
            # Create visualization if graph has nodes
            if G.nodes():
                # Apply selected layout algorithm
                try:
                    if layout_algo == "spring_layout":
                        pos = nx.spring_layout(G, k=1, iterations=50)
                    elif layout_algo == "circular_layout":
                        pos = nx.circular_layout(G)
                    elif layout_algo == "kamada_kawai_layout":
                        pos = nx.kamada_kawai_layout(G)
                    else:
                        pos = nx.random_layout(G)
                except:
                    # Fallback to spring layout
                    pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Extract coordinates
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                node_text = [str(node) for node in G.nodes()]
                node_info = [f"{node}<br>Degree: {G.degree(node)}" for node in G.nodes()]
                
                # Create node trace
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="middle center",
                    hoverinfo='text',
                    hovertext=node_info,
                    marker=dict(
                        size=KG_NODE_SIZE,
                        color=KG_NODE_COLOR,
                        line=dict(width=2, color='darkblue')
                    ),
                    name="Entities"
                )
                
                # Create edge traces
                edge_x = []
                edge_y = []
                edge_info = []
                
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
                    # Add edge information
                    edge_data = edge[2]
                    relation_type = edge_data.get("relation", edge_data.get("type", "connected"))
                    edge_info.append(f"{edge[0]} → {edge[1]}<br>Relation: {relation_type}")
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=KG_EDGE_WIDTH, color=KG_EDGE_COLOR),
                    hoverinfo='none',
                    mode='lines',
                    name="Relations"
                )
                
                # Create figure
                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Knowledge Graph for "{kg_label}" (Depth: {kg_depth}, Layout: {layout_algo})',
                        titlefont_size=16,
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text=f"Interactive Knowledge Graph - {len(G.nodes())} nodes, {len(G.edges())} edges",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color="gray", size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=KG_FIGURE_HEIGHT
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Graph statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Nodes", len(G.nodes()))
                with col2:
                    st.metric("Edges", len(G.edges()))
                with col3:
                    avg_degree = sum(dict(G.degree()).values())/len(G.nodes()) if G.nodes() else 0
                    st.metric("Avg Degree", f"{avg_degree:.1f}")
                with col4:
                    st.metric("Connected Components", nx.number_connected_components(G))
                
                # Detailed graph analysis
                with st.expander("📊 Detailed Graph Analysis", expanded=False):
                    if G.nodes():
                        # Top nodes by degree
                        degree_centrality = nx.degree_centrality(G)
                        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Top Nodes by Degree:**")
                            for node, centrality in top_nodes:
                                st.write(f"• {node}: {centrality:.3f}")
                        
                        with col2:
                            # Graph metrics
                            st.write("**Graph Metrics:**")
                            st.write(f"• Density: {nx.density(G):.3f}")
                            st.write(f"• Is Connected: {nx.is_connected(G)}")
                            if nx.is_connected(G):
                                st.write(f"• Diameter: {nx.diameter(G)}")
                                st.write(f"• Avg Path Length: {nx.average_shortest_path_length(G):.2f}")
                
                # Export options
                with st.expander("💾 Export Options", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📄 Export as JSON"):
                            graph_data = nx.node_link_data(G)
                            st.download_button(
                                "Download JSON",
                                json.dumps(graph_data, indent=2),
                                f"knowledge_graph_{kg_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                "application/json"
                            )
                    with col2:
                        if st.button("📊 Export as CSV"):
                            edges_df = pd.DataFrame(G.edges(data=True))
                            if not edges_df.empty:
                                st.download_button(
                                    "Download CSV",
                                    edges_df.to_csv(index=False),
                                    f"knowledge_graph_edges_{kg_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv"
                                )
                    with col3:
                        if st.button("🖼️ Export as HTML"):
                            html_str = fig.to_html()
                            st.download_button(
                                "Download HTML",
                                html_str,
                                f"knowledge_graph_{kg_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                "text/html"
                            )
            else:
                st.info("No graph data found. Try a different entity name or check if the knowledge base contains this entity.")
        else:
            st.warning("No knowledge graph data available or entity not found. Make sure documents are processed and the entity exists in the knowledge base.")
    else:
        st.info("Enter an entity name to explore the knowledge graph.")
        
        # Show available entities if possible
        with st.expander("💡 Tips for Knowledge Graph Exploration", expanded=True):
            st.write("""
            **How to use the Knowledge Graph Visualizer:**
            1. **Entity Name**: Enter any entity from your processed documents
            2. **Depth**: Controls how many relationship hops to include
            3. **Max Nodes**: Limits the visualization size for performance
            4. **Layout**: Choose different visualization arrangements
            
            **Tips:**
            - Start with depth 2-3 for most queries
            - Use higher node limits for comprehensive exploration
            - Try different layouts to find the clearest view
            - Export results for external analysis
            """)

# Tab 3: Enhanced Document Dashboard
with tab3:
    st.header("📋 Document Management Dashboard")
    
    # Document upload section
    st.subheader("📤 Upload Documents")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'md', 'json', 'csv', 'xlsx', 'pptx']
        )
    
    with col2:
        st.write("**Supported Formats:**")
        st.write("• PDF, DOCX, PPTX")
        st.write("• TXT, MD, JSON, CSV")
        st.write("• XLSX spreadsheets")
        
        if st.button("📤 Upload Files", disabled=not uploaded_files):
            with st.spinner("Uploading files..."):
                progress_bar = st.progress(0)
                results = []
                
                for i, file in enumerate(uploaded_files):
                    result = upload_file_to_api(file)
                    results.append({"file": file.name, "result": result})
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                progress_bar.empty()
                
                # Show results
                success_count = 0
                for result in results:
                    if "error" not in result["result"]:
                        st.success(f"✅ {result['file']} uploaded successfully")
                        success_count += 1
                    else:
                        st.error(f"❌ {result['file']} failed: {result['result']['error']}")
                
                st.info(f"Upload completed: {success_count}/{len(results)} files successful")
    
    # Text input section
    st.subheader("✏️ Add Text Directly")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area("Enter text content", height=100, placeholder="Paste or type your content here...")
        text_source = st.text_input("Source (optional)", value="Streamlit App", placeholder="Source description")
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("➕ Add Text", disabled=not text_input.strip()):
            with st.spinner("Adding text..."):
                result = add_text_to_api(text_input.strip(), text_source)
                if "error" not in result:
                    st.success("✅ Text added successfully")
                    st.session_state.text_added = True
                else:
                    st.error(f"❌ Failed: {result['error']}")
    
    # Document management actions
    st.subheader("🔧 Document Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Scan for New Files"):
            with st.spinner("Scanning..."):
                result = scan_documents()
                if "error" not in result:
                    st.success("📂 Scan initiated")
                else:
                    st.error(f"❌ Scan failed: {result['error']}")
    
    with col2:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    with col3:
        if st.button("🧹 Clear Cache"):
            with st.spinner("Clearing cache..."):
                result = clear_cache()
                if "error" not in result:
                    st.success("🧹 Cache cleared")
                else:
                    st.error(f"❌ Clear failed: {result['error']}")
    
    with col4:
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.get("confirm_clear"):
                with st.spinner("Clearing documents..."):
                    result = clear_documents()
                    if "error" not in result:
                        st.success("🗑️ Documents cleared")
                        st.session_state.confirm_clear = False
                    else:
                        st.error(f"❌ Clear failed: {result['error']}")
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm deletion")
    
    # Document status
    st.subheader("📊 Document Status")
    with st.spinner("Loading document status..."):
        doc_status = get_document_status()
    
    if "error" not in doc_status and "statuses" in doc_status:
        statuses = doc_status["statuses"]
        
        # Status overview
        status_counts = {
            "PENDING": len(statuses.get("PENDING", [])),
            "PROCESSING": len(statuses.get("PROCESSING", [])),
            "PROCESSED": len(statuses.get("PROCESSED", [])),
            "FAILED": len(statuses.get("FAILED", []))
        }
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Pending", status_counts["PENDING"])
        with col2:
            st.metric("⚙️ Processing", status_counts["PROCESSING"])
        with col3:
            st.metric("✅ Processed", status_counts["PROCESSED"])
        with col4:
            st.metric("❌ Failed", status_counts["FAILED"])
        
        # Status distribution chart
        if sum(status_counts.values()) > 0:
            fig = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Document Processing Status Distribution",
                color_discrete_map={
                    "PENDING": "#ffc107",
                    "PROCESSING": "#17a2b8", 
                    "PROCESSED": "#28a745",
                    "FAILED": "#dc3545"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed status table
        if any(statuses.values()):
            all_docs = []
            for status, docs in statuses.items():
                for doc in docs:
                    all_docs.append({
                        "File": doc.get("file_path", "Unknown"),
                        "Status": status,
                        "Size": f"{doc.get('content_length', 0):,} chars" if doc.get('content_length') else "Unknown",
                        "Chunks": doc.get("chunks_count", 0),
                        "Created": doc.get("created_at", "")[:19] if doc.get("created_at") else "",
                        "Updated": doc.get("updated_at", "")[:19] if doc.get("updated_at") else ""
                    })
            
            if all_docs:
                df = pd.DataFrame(all_docs)
                
                # Add filtering options
                col1, col2 = st.columns(2)
                with col1:
                    status_filter = st.multiselect(
                        "Filter by Status",
                        options=["PENDING", "PROCESSING", "PROCESSED", "FAILED"],
                        default=["PENDING", "PROCESSING", "PROCESSED", "FAILED"]
                    )
                with col2:
                    search_term = st.text_input("Search files", placeholder="Filter by filename...")
                
                # Apply filters
                if status_filter:
                    df = df[df["Status"].isin(status_filter)]
                
                if search_term:
                    df = df[df["File"].str.contains(search_term, case=False, na=False)]
                
                # Display paginated results
                if not df.empty:
                    total_pages = len(df) // UI_PAGE_SIZE + (1 if len(df) % UI_PAGE_SIZE > 0 else 0)
                    if total_pages > 1:
                        page = st.selectbox("Page", range(1, total_pages + 1), key="doc_page")
                        start_idx = (page - 1) * UI_PAGE_SIZE
                        end_idx = start_idx + UI_PAGE_SIZE
                        df_display = df.iloc[start_idx:end_idx]
                    else:
                        df_display = df
                    
                    st.dataframe(df_display, use_container_width=True)
                    st.write(f"Showing {len(df_display)} of {len(df)} documents")
                else:
                    st.info("No documents match the current filters.")
        else:
            st.info("No documents found in the system.")
    else:
        st.error("Failed to load document status")

# Tab 4: Enhanced System Monitoring
with tab4:
    st.header("📈 System Monitoring Dashboard")
    
    # Real-time metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Performance Metrics")
        if "error" not in health:
            perf = health.get("performance", {})
            
            if perf:
                # Create performance metrics chart
                metrics_data = {
                    "Metric": [
                        "Uptime (hours)",
                        "Total Requests", 
                        "Total Queries",
                        "Total Documents"
                    ],
                    "Value": [
                        perf.get('uptime_seconds', 0) / 3600,
                        perf.get('total_requests', 0),
                        perf.get('total_queries', 0),
                        perf.get('total_documents', 0)
                    ]
                }
                
                fig = px.bar(
                    metrics_data, 
                    x="Metric", 
                    y="Value", 
                    title="System Performance Overview",
                    color="Value",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(height=UI_CHART_HEIGHT)
                st.plotly_chart(fig, use_container_width=True)
                
                # Token usage if available
                token_usage = perf.get("token_usage", {})
                if token_usage:
                    st.subheader("🎯 Token Usage Analytics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Tokens", f"{token_usage.get('total_tokens', 0):,}")
                    with col2:
                        st.metric("Total Requests", f"{token_usage.get('total_requests', 0):,}")
                    with col3:
                        avg_tokens = token_usage.get('total_tokens', 0) / max(1, token_usage.get('total_requests', 1))
                        st.metric("Avg Tokens/Request", f"{avg_tokens:.1f}")
                    with col4:
                        cost_estimate = token_usage.get('total_tokens', 0) * 0.00002  # Rough estimate
                        st.metric("Est. Cost", f"${cost_estimate:.4f}")
            else:
                st.info("Performance metrics not available")
    
    with col2:
        st.subheader("⚙️ System Configuration")
        if "error" not in health:
            config = health.get("configuration", {})
            
            if config:
                st.write("**Server Configuration:**")
                server_config = config.get('server', {})
                st.write(f"• Host: `{server_config.get('host', 'N/A')}`")
                st.write(f"• Port: `{server_config.get('port', 'N/A')}`")
                st.write(f"• Workspace: `{server_config.get('workspace', 'N/A')}`")
                
                st.write("**LLM Configuration:**")
                llm_config = config.get('llm', {})
                st.write(f"• Model: `{llm_config.get('model', 'N/A')}`")
                st.write(f"• Max Async: `{llm_config.get('max_async', 'N/A')}`")
                st.write(f"• Temperature: `{llm_config.get('temperature', 'N/A')}`")
                
                st.write("**Embeddings:**")
                embed_config = config.get('embeddings', {})
                st.write(f"• Model: `{embed_config.get('model', 'N/A')}`")
                st.write(f"• Dimension: `{embed_config.get('dimension', 'N/A')}`")
                st.write(f"• Batch Size: `{embed_config.get('batch_size', 'N/A')}`")
            else:
                st.info("Configuration details not available")
        else:
            st.error("Unable to fetch system configuration")
    
    # API Calls Dashboard
    st.subheader("🌐 API Calls Analytics")
    
    if st.session_state.api_calls:
        # Recent API calls analysis
        recent_calls = st.session_state.api_calls[-200:]  # Last 200 calls
        
        # API call statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Calls", len(st.session_state.api_calls))
        with col2:
            success_rate = sum(1 for call in recent_calls if call["success"]) / len(recent_calls) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            avg_duration = sum(call["duration"] for call in recent_calls) / len(recent_calls)
            st.metric("Avg Duration", f"{avg_duration:.2f}s")
        with col4:
            endpoints = set(call["endpoint"] for call in recent_calls)
            st.metric("Unique Endpoints", len(endpoints))
        
        # API calls timeline
        if len(recent_calls) > 1:
            calls_df = pd.DataFrame([
                {
                    "Time": call["timestamp"],
                    "Endpoint": call["endpoint"],
                    "Method": call["method"],
                    "Status": call["status_code"],
                    "Duration": call["duration"],
                    "Success": "✅" if call["success"] else "❌"
                }
                for call in recent_calls
            ])
            
            # Timeline chart
            fig = px.scatter(
                calls_df, 
                x="Time", 
                y="Duration",
                color="Success",
                hover_data=["Endpoint", "Method", "Status"],
                title="API Calls Timeline (Last 200 calls)",
                color_discrete_map={"✅": "green", "❌": "red"}
            )
            fig.update_layout(height=UI_CHART_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)
            
            # Endpoint performance analysis
            endpoint_stats = calls_df.groupby("Endpoint").agg({
                "Duration": ["mean", "count"],
                "Success": lambda x: (x == "✅").sum() / len(x) * 100
            }).round(3)
            endpoint_stats.columns = ["Avg Duration (s)", "Call Count", "Success Rate (%)"]
            
            st.subheader("📊 Endpoint Performance")
            st.dataframe(endpoint_stats, use_container_width=True)
            
            # Recent calls table
            st.subheader("🕐 Recent API Calls")
            display_calls = calls_df.tail(UI_MESSAGES_PER_PAGE)
            st.dataframe(display_calls, use_container_width=True)
    else:
        st.info("No API calls recorded yet. Interact with the system to see API call statistics.")
    
    # Auto-refresh option
    if st.checkbox("🔄 Auto-refresh dashboard", value=False, key="auto_refresh"):
        if st.session_state.auto_refresh:
            time.sleep(UI_AUTO_REFRESH_INTERVAL)
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        {STREAMLIT_TITLE} - Built with ❤️ for production RAG systems<br>
        <small>Configuration loaded from environment • API: {API_BASE}</small>
    </div>
    """,
    unsafe_allow_html=True
) 