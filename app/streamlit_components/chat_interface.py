"""
Chat Interface Component for LightRAG Streamlit Frontend
Interactive chat interface with real-time token and API call tracking
"""

import streamlit as st
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

# Import API client and tracking at module level
from utils.api_client import get_api_client

# Import tracking function at module level to avoid repeated imports
try:
    from monitoring.enhanced_token_tracker import track_query_metrics
except ImportError:
    track_query_metrics = None


class ChatInterface:
    """Interactive chat interface component"""
    
    def __init__(self):
        self.api_client = get_api_client()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state for chat interface"""
        
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I'm your LightRAG assistant with Gemini integration. How can I help you today?",
                    "timestamp": datetime.now()
                }
            ]
        
        if "chat_config" not in st.session_state:
            st.session_state.chat_config = {
                "query_mode": "mix",
                "top_k": 40,
                "chunk_top_k": 10,
                "max_total_tokens": 30000,
                "response_type": "Multiple Paragraphs",
                "enable_rerank": True,
                "include_context": False,
                "include_prompt": False
            }
        
        if "query_history" not in st.session_state:
            st.session_state.query_history = []
    
    def render(self):
        """Render the complete chat interface"""
        
        st.title("💬 LightRAG Chat Interface")
        
        # Main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self.render_chat_area()
        
        with col2:
            self.render_sidebar_config()
    
    def render_chat_area(self):
        """Render the main chat area"""
        
        # Chat history
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_messages):
                self.render_message(message, i)
        
        # Chat input
        st.markdown("---")
        self.render_chat_input()
    
    def render_message(self, message: Dict[str, Any], index: int):
        """Render individual chat message with metadata"""
        
        role = message.get("role", "unknown")
        content = message.get("content", "")
        timestamp = message.get("timestamp")
        metadata = message.get("metadata", {})
        
        # Message styling
        if role == "user":
            st.markdown(f'<div class="chat-message user">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant">{content}</div>', unsafe_allow_html=True)
        
        # Show metadata in expander
        if metadata:
            with st.expander(f"📊 Query Details", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Performance:**")
                    if "response_time" in metadata:
                        st.write(f"⏱️ Response Time: {metadata['response_time']:.2f}s")
                    if "query_mode" in metadata:
                        st.write(f"🔍 Query Mode: {metadata['query_mode']}")
                    
                    # API Calls tracking
                    if "api_calls" in metadata:
                        st.write("**API Calls:**")
                        api_calls = metadata["api_calls"]
                        total_calls = api_calls.get("llm_calls", 0) + api_calls.get("embedding_calls", 0) + api_calls.get("rerank_calls", 0)
                        st.write(f"🔄 Total API Calls: {total_calls}")
                        if api_calls.get("llm_calls", 0) > 0:
                            st.write(f"🧠 LLM Calls: {api_calls['llm_calls']}")
                        if api_calls.get("embedding_calls", 0) > 0:
                            st.write(f"📊 Embedding Calls: {api_calls['embedding_calls']}")
                        if api_calls.get("rerank_calls", 0) > 0:
                            st.write(f"🔄 Rerank Calls: {api_calls['rerank_calls']}")
                
                with col2:
                    st.write("**Resources:**")
                    if "token_usage" in metadata:
                        tokens = metadata["token_usage"]
                        st.write(f"🪙 Tokens: {tokens.get('total_tokens', 0):,}")
                        st.write(f"📥 Input: {tokens.get('prompt_tokens', 0):,}")
                        st.write(f"📤 Output: {tokens.get('completion_tokens', 0):,}")
                    
                    if "context_chunks" in metadata:
                        st.write(f"📄 Context Chunks: {metadata['context_chunks']}")
                    if "entities_used" in metadata:
                        st.write(f"🏷️ Entities: {metadata['entities_used']}")
                    if "relationships_used" in metadata:
                        st.write(f"🔗 Relationships: {metadata['relationships_used']}")
        
        if timestamp:
            st.caption(f"🕐 {timestamp.strftime('%H:%M:%S')}")
    
    def render_chat_input(self):
        """Render chat input area"""
        
        # Example queries
        with st.expander("💡 Example Queries", expanded=False):
            example_queries = [
                "What are the main topics in the documents?",
                "Summarize the key findings",
                "What relationships exist between entities?",
                "Find information about specific concepts"
            ]
            
            st.write("Click any example to try it:")
            for query in example_queries:
                if st.button(f"📝 {query}", key=f"example_{hash(query)}"):
                    self.process_user_query(query)
        
        # Main input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Your Question:",
                placeholder="Type your question here...",
                height=100,
                help="Ask anything about your uploaded documents"
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                submit_button = st.form_submit_button("🚀 Ask", use_container_width=True)
            
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)
            
            with col3:
                context_only = st.checkbox("Return context only", value=False)
        
        # Handle form submissions
        if submit_button and user_input.strip():
            config = st.session_state.chat_config.copy()
            config["only_need_context"] = context_only
            self.process_user_query(user_input.strip(), config)
        
        if clear_button:
            st.session_state.chat_messages = [st.session_state.chat_messages[0]]  # Keep welcome message
            st.session_state.query_history = []
            st.rerun()
    
    def render_sidebar_config(self):
        """Render sidebar configuration panel"""
        
        st.subheader("⚙️ Query Configuration")
        
        # Query mode
        query_mode = st.selectbox(
            "Query Mode",
            ["mix", "local", "global", "hybrid", "naive"],
            index=["mix", "local", "global", "hybrid", "naive"].index(st.session_state.chat_config["query_mode"]),
            help="""
            - mix: Knowledge graph + vector retrieval (recommended)
            - local: Entity-focused retrieval
            - global: Relationship-focused retrieval  
            - hybrid: Combines local and global
            - naive: Simple vector search
            """
        )
        st.session_state.chat_config["query_mode"] = query_mode
        
        # Parameters
        st.markdown("**📊 Retrieval Parameters**")
        
        top_k = st.slider(
            "Top K Entities/Relations",
            min_value=1, max_value=100,
            value=st.session_state.chat_config["top_k"],
            help="Number of entities/relationships to retrieve"
        )
        st.session_state.chat_config["top_k"] = top_k
        
        chunk_top_k = st.slider(
            "Top K Chunks",
            min_value=1, max_value=50,
            value=st.session_state.chat_config["chunk_top_k"],
            help="Number of text chunks to retrieve"
        )
        st.session_state.chat_config["chunk_top_k"] = chunk_top_k
        
        max_tokens = st.slider(
            "Max Total Tokens",
            min_value=5000, max_value=50000, step=1000,
            value=st.session_state.chat_config["max_total_tokens"],
            help="Maximum tokens for the entire query context"
        )
        st.session_state.chat_config["max_total_tokens"] = max_tokens
        
        # Response configuration
        st.markdown("**📝 Response Options**")
        
        response_type = st.selectbox(
            "Response Type",
            ["Multiple Paragraphs", "Single Paragraph", "Single Sentence", "List of 3-7 Points", "Single Page", "Multi-Page Report"],
            index=["Multiple Paragraphs", "Single Paragraph", "Single Sentence", "List of 3-7 Points", "Single Page", "Multi-Page Report"].index(
                st.session_state.chat_config["response_type"]
            )
        )
        st.session_state.chat_config["response_type"] = response_type
        
        # Advanced options
        st.markdown("**🔧 Advanced Options**")
        
        enable_rerank = st.checkbox(
            "Enable Reranking",
            value=st.session_state.chat_config["enable_rerank"],
            help="Use LLM-based reranking for better results"
        )
        st.session_state.chat_config["enable_rerank"] = enable_rerank
        
        include_context = st.checkbox(
            "Include Retrieved Context",
            value=st.session_state.chat_config["include_context"],
            help="Show the retrieved context in the response"
        )
        st.session_state.chat_config["include_context"] = include_context
        
        include_prompt = st.checkbox(
            "Include Generated Prompt",
            value=st.session_state.chat_config["include_prompt"],
            help="Show the prompt sent to the LLM"
        )
        st.session_state.chat_config["include_prompt"] = include_prompt
        
        # Session statistics
        st.markdown("---")
        st.markdown("**📈 Session Statistics**")
        
        total_queries = len([msg for msg in st.session_state.chat_messages if msg["role"] == "user"])
        total_tokens = sum(
            msg.get("metadata", {}).get("token_usage", {}).get("total_tokens", 0)
            for msg in st.session_state.chat_messages
            if msg["role"] == "assistant"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Queries", total_queries)
            st.metric("Total Tokens", f"{total_tokens:,}")
        
        with col2:
            st.metric("Avg Tokens/Query", f"{total_tokens // max(total_queries, 1):,}")
            # Cost tracking removed as requested
        
        # Chat controls - Enhanced
        st.markdown("**💬 Chat Controls**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset Session", use_container_width=True):
                self.reset_session()
        
        with col2:
            if st.button("✨ New Chat", use_container_width=True):
                self.start_new_chat()
    
    def process_user_query(self, query: str, config: Optional[Dict] = None):
        """Process user query and get response from LightRAG"""
        
        # Add user message
        user_message = {
            "role": "user",
            "content": query,
            "timestamp": datetime.now()
        }
        st.session_state.chat_messages.append(user_message)
        
        # Process with API
        query_config = config.copy() if config is not None else st.session_state.chat_config.copy()
        
        # Show processing indicator
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            
            # Track API calls for this query
            api_calls_start = {
                "llm_calls": 0,
                "embedding_calls": 0,
                "rerank_calls": 0
            }
            
            # Get baseline API stats from Gemini integration
            llm_instance = None
            embedding_instance = None
            
            try:
                from gemini_integration.gemini_llm import get_global_llm_instance
                from gemini_integration.gemini_embeddings import get_global_embedding_instance
                
                llm_instance = get_global_llm_instance()
                embedding_instance = get_global_embedding_instance()
                
                if llm_instance:
                    llm_stats_before = llm_instance.get_session_stats()
                    api_calls_start["llm_calls"] = llm_stats_before.get("total_calls", 0)
                
                if embedding_instance:
                    embedding_stats_before = embedding_instance.get_session_stats()
                    api_calls_start["embedding_calls"] = embedding_stats_before.get("total_requests", 0)
            except ImportError:
                pass  # Continue without baseline if instances not available
            
            # Prepare conversation history
            conversation_history = []
            for msg in st.session_state.chat_messages[-6:]:  # Last 3 exchanges
                if msg["role"] in ["user", "assistant"]:
                    conversation_history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Make API request
            result = self.api_client.query(
                query=query,
                mode=query_config["query_mode"],
                top_k=query_config["top_k"],
                chunk_top_k=query_config["chunk_top_k"],
                max_total_tokens=query_config["max_total_tokens"],
                enable_rerank=query_config["enable_rerank"],
                response_type=query_config["response_type"],
                only_need_context=query_config.get("only_need_context", False),
                only_need_prompt=query_config.get("only_need_prompt", False),
                conversation_history=conversation_history[:-1]  # Exclude current query
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Calculate API calls for this query
            api_calls_used = {
                "llm_calls": 0,
                "embedding_calls": 0,
                "rerank_calls": 0
            }
            
            try:
                if llm_instance:
                    llm_stats_after = llm_instance.get_session_stats()
                    api_calls_used["llm_calls"] = llm_stats_after.get("total_calls", 0) - api_calls_start["llm_calls"]
                
                if embedding_instance:
                    embedding_stats_after = embedding_instance.get_session_stats()
                    api_calls_used["embedding_calls"] = embedding_stats_after.get("total_requests", 0) - api_calls_start["embedding_calls"]
                
                # Estimate rerank calls (if reranking was enabled)
                if query_config["enable_rerank"] and result.get("success", False):
                    # Approximate: 1 rerank call if reranking was enabled and query succeeded
                    api_calls_used["rerank_calls"] = 1
            except Exception:
                pass  # Continue with zeros if calculation fails
        
        # Process result
        if result.get("success", False):
            response_data = result.get("data", {})
            
            # Extract response content
            if query_config.get("only_need_context"):
                content = f"**Retrieved Context:**\n\n{response_data.get('context', 'No context available')}"
            elif query_config.get("only_need_prompt"):
                content = f"**Generated Prompt:**\n\n{response_data.get('prompt', 'No prompt available')}"
            else:
                content = response_data.get('response', 'No response generated')
            
            # Prepare metadata with API calls
            metadata = {
                "response_time": response_time,
                "query_mode": query_config["query_mode"],
                "token_usage": response_data.get("token_usage", {}),
                "context_chunks": len(response_data.get("context_chunks", [])),
                "entities_used": len(response_data.get("entities", [])),
                "relationships_used": len(response_data.get("relationships", [])),
                "api_calls": api_calls_used  # Per-query API call tracking
            }
            
            # Add context and prompt if requested
            if query_config.get("include_context") and "context" in response_data:
                content += f"\n\n**Retrieved Context:**\n{response_data['context']}"
            
            if query_config.get("include_prompt") and "prompt" in response_data:
                content += f"\n\n**Generated Prompt:**\n{response_data['prompt']}"
            
            # Track in enhanced token tracker
            if track_query_metrics:
                try:
                    track_query_metrics(
                        query_text=query,
                        query_mode=query_config["query_mode"],
                        response_time=response_time,
                        success=True,
                        token_usage=response_data.get("token_usage", {}),
                        cost_estimate=0.0,
                        llm_calls=api_calls_used["llm_calls"],
                        embedding_calls=api_calls_used["embedding_calls"],
                        rerank_calls=api_calls_used["rerank_calls"],
                        context_chunks=metadata["context_chunks"],
                        entities_used=metadata["entities_used"],
                        relationships_used=metadata["relationships_used"],
                        top_k=query_config["top_k"],
                        chunk_top_k=query_config["chunk_top_k"],
                        max_total_tokens=query_config["max_total_tokens"],
                        enable_rerank=query_config["enable_rerank"]
                    )
                except Exception as e:
                    print(f"Warning: Could not track query metrics: {e}")
            
        else:
            content = f"❌ **Error:** {result.get('error', 'Unknown error occurred')}"
            metadata = {
                "response_time": response_time,
                "error": result.get('error'),
                "query_mode": query_config["query_mode"],
                "api_calls": api_calls_used  # Track API calls even for errors
            }
            
            # Track error in enhanced token tracker
            if track_query_metrics:
                try:
                    track_query_metrics(
                        query_text=query,
                        query_mode=query_config["query_mode"],
                        response_time=response_time,
                        success=False,
                        token_usage={},
                        cost_estimate=0.0,
                        llm_calls=api_calls_used["llm_calls"],
                        embedding_calls=api_calls_used["embedding_calls"],
                        rerank_calls=api_calls_used["rerank_calls"],
                        error_message=result.get('error'),
                        error_type="api_error"
                    )
                except Exception as e:
                    print(f"Warning: Could not track error metrics: {e}")
        
        # Update session token usage
        if "token_usage" in metadata:
            token_usage = metadata["token_usage"]
            st.session_state.token_usage["total_tokens"] += token_usage.get("total_tokens", 0)
            
            # Update API call counts in session state
            if "api_calls" in metadata:
                api_calls = metadata["api_calls"]
                st.session_state.token_usage["api_calls"] += (
                    api_calls.get("llm_calls", 0) + 
                    api_calls.get("embedding_calls", 0) + 
                    api_calls.get("rerank_calls", 0)
                )
        
        # Add assistant response
        assistant_message = {
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata
        }
        st.session_state.chat_messages.append(assistant_message)
        
        # Update session state and rerun
        st.rerun()
    
    def start_new_chat(self):
        """Start a new chat session"""
        st.session_state.chat_messages = [{
            "role": "assistant", 
            "content": "Hello! I'm your LightRAG assistant with Gemini integration. How can I help you today?",
            "timestamp": datetime.now()
        }]
        st.session_state.query_history = []
        st.rerun()
    
    def reset_session(self):
        """Reset chat session"""
        st.session_state.chat_messages = [st.session_state.chat_messages[0]]  # Keep welcome message
        st.session_state.query_history = []
        st.rerun() 