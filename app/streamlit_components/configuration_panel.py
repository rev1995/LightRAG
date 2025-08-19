"""
Configuration Panel Component for LightRAG Streamlit Frontend
Manage system settings, environment variables, and model configurations
"""

import streamlit as st
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Import API client
from utils.api_client import get_api_client, test_api_connection


class ConfigurationPanel:
    """System configuration management component"""
    
    def __init__(self):
        self.api_client = get_api_client()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state for configuration"""
        
        if "config_changes" not in st.session_state:
            st.session_state.config_changes = {}
        
        if "config_saved" not in st.session_state:
            st.session_state.config_saved = False
    
    def render(self):
        """Render the complete configuration interface"""
        
        st.title("⚙️ System Configuration")
        
        # Configuration tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔑 API Keys", 
            "🧠 Model Settings", 
            "💾 Storage", 
            "⚡ Performance", 
            "🔧 Advanced"
        ])
        
        with tab1:
            self.render_api_keys_config()
        
        with tab2:
            self.render_model_config()
        
        with tab3:
            self.render_storage_config()
        
        with tab4:
            self.render_performance_config()
        
        with tab5:
            self.render_advanced_config()
        
        # Save configuration
        st.markdown("---")
        self.render_config_actions()
    
    def render_api_keys_config(self):
        """Render API keys configuration"""
        
        st.subheader("🔑 API Keys & Endpoints")
        
        # Gemini Configuration
        with st.expander("🤖 Gemini API Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                gemini_api_key = st.text_input(
                    "Gemini API Key",
                    value=os.getenv("GEMINI_API_KEY", ""),
                    type="password",
                    help="Your Google AI Studio API key for Gemini models"
                )
                
                if st.button("🧪 Test Gemini Connection"):
                    self.test_gemini_connection(gemini_api_key)
            
            with col2:
                st.info("""
                **Getting your Gemini API Key:**
                1. Go to [Google AI Studio](https://aistudio.google.com/)
                2. Sign in with your Google account
                3. Create a new API key
                4. Copy and paste it here
                """)
        
        # LightRAG Server Configuration
        with st.expander("🚀 LightRAG Server Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                lightrag_host = st.text_input(
                    "LightRAG Host",
                    value=os.getenv("HOST", "localhost"),
                    help="Hostname for LightRAG API server"
                )
                
                lightrag_port = st.number_input(
                    "LightRAG Port",
                    min_value=1000, max_value=9999,
                    value=int(os.getenv("PORT", "9621")),
                    help="Port number for LightRAG API server"
                )
                
                api_base_url = f"http://{lightrag_host}:{lightrag_port}"
                st.code(f"API Base URL: {api_base_url}")
            
            with col2:
                if st.button("🔍 Test API Connection"):
                    self.test_lightrag_connection(api_base_url)
                
                # Server status
                self.render_server_status()
        
        # Optional Authentication
        with st.expander("🔐 Authentication (Optional)", expanded=False):
            enable_auth = st.checkbox(
                "Enable Authentication",
                value=bool(os.getenv("AUTH_ACCOUNTS")),
                help="Enable user authentication for the LightRAG server"
            )
            
            if enable_auth:
                auth_accounts = st.text_area(
                    "Auth Accounts",
                    value=os.getenv("AUTH_ACCOUNTS", ""),
                    help="Format: username:password,username2:password2",
                    placeholder="admin:admin123,user1:pass456"
                )
                
                token_secret = st.text_input(
                    "Token Secret",
                    value=os.getenv("TOKEN_SECRET", ""),
                    type="password",
                    help="Secret key for JWT token generation"
                )
    
    def render_model_config(self):
        """Render model configuration"""
        
        st.subheader("🧠 Model Settings")
        
        # LLM Configuration
        with st.expander("🗣️ Language Model Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                llm_model = st.selectbox(
                    "LLM Model",
                    ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
                    index=0,
                    help="Choose the Gemini language model"
                )
                
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0, max_value=2.0, step=0.1,
                    value=float(os.getenv("TEMPERATURE", "0.0")),
                    help="Controls randomness in responses"
                )
                
                max_output_tokens = st.number_input(
                    "Max Output Tokens",
                    min_value=100, max_value=10000,
                    value=int(os.getenv("MAX_OUTPUT_TOKENS", "5000")),
                    help="Maximum tokens in model response"
                )
            
            with col2:
                timeout = st.number_input(
                    "Timeout (seconds)",
                    min_value=10, max_value=600,
                    value=int(os.getenv("TIMEOUT", "240")),
                    help="Request timeout for LLM calls"
                )
                
                enable_llm_cache = st.checkbox(
                    "Enable LLM Cache",
                    value=os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true",
                    help="Cache LLM responses to reduce API calls"
                )
        
        # Embedding Configuration
        with st.expander("📊 Embedding Model Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ["text-embedding-004", "textembedding-gecko"],
                    index=0,
                    help="Choose the Gemini embedding model"
                )
                
                embedding_dim = st.number_input(
                    "Embedding Dimensions",
                    min_value=256, max_value=2048, step=256,
                    value=int(os.getenv("EMBEDDING_DIM", "768")),
                    help="Embedding vector dimensions"
                )
            
            with col2:
                embedding_max_tokens = st.number_input(
                    "Max Embedding Tokens",
                    min_value=512, max_value=8192,
                    value=int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "8192")),
                    help="Maximum tokens per embedding request"
                )
                
                embedding_batch_size = st.number_input(
                    "Embedding Batch Size",
                    min_value=1, max_value=100,
                    value=int(os.getenv("EMBEDDING_BATCH_NUM", "10")),
                    help="Number of texts to embed in one batch"
                )
        
        # Reranking Configuration
        with st.expander("🔄 Reranking Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_rerank = st.checkbox(
                    "Enable Reranking",
                    value=os.getenv("ENABLE_RERANK", "true").lower() == "true",
                    help="Use LLM-based reranking for better results"
                )
                
                if enable_rerank:
                    rerank_model = st.selectbox(
                        "Rerank Model",
                        ["gemini-2.0-flash", "gemini-1.5-pro"],
                        index=0,
                        help="Model to use for reranking"
                    )
            
            with col2:
                if enable_rerank:
                    min_rerank_score = st.slider(
                        "Min Rerank Score",
                        min_value=0.0, max_value=1.0, step=0.1,
                        value=float(os.getenv("MIN_RERANK_SCORE", "0.0")),
                        help="Minimum score to keep documents after reranking"
                    )
    
    def render_storage_config(self):
        """Render storage configuration"""
        
        st.subheader("💾 Storage Configuration")
        
        # Storage Backend Selection
        with st.expander("🗄️ Storage Backends", expanded=True):
            storage_types = {
                "Key-Value Storage": {
                    "options": ["JsonKVStorage", "RedisKVStorage", "PGKVStorage", "MongoKVStorage"],
                    "env_var": "LIGHTRAG_KV_STORAGE",
                    "default": "JsonKVStorage"
                },
                "Vector Storage": {
                    "options": ["NanoVectorDBStorage", "FaissVectorDBStorage", "QdrantVectorDBStorage", "MilvusVectorDBStorage", "PGVectorStorage"],
                    "env_var": "LIGHTRAG_VECTOR_STORAGE", 
                    "default": "NanoVectorDBStorage"
                },
                "Graph Storage": {
                    "options": ["NetworkXStorage", "Neo4JStorage", "MemgraphStorage", "PGGraphStorage", "MongoGraphStorage"],
                    "env_var": "LIGHTRAG_GRAPH_STORAGE",
                    "default": "NetworkXStorage"
                },
                "Document Status Storage": {
                    "options": ["JsonDocStatusStorage", "RedisDocStatusStorage", "PGDocStatusStorage", "MongoDocStatusStorage"],
                    "env_var": "LIGHTRAG_DOC_STATUS_STORAGE",
                    "default": "JsonDocStatusStorage"
                }
            }
            
            for storage_name, config in storage_types.items():
                current_value = os.getenv(config["env_var"], config["default"])
                
                selected = st.selectbox(
                    storage_name,
                    config["options"],
                    index=config["options"].index(current_value) if current_value in config["options"] else 0,
                    help=f"Storage backend for {storage_name.lower()}"
                )
        
        # Storage Paths
        with st.expander("📁 Storage Paths", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                working_dir = st.text_input(
                    "Working Directory",
                    value=os.getenv("WORKING_DIR", "../storage/rag_storage"),
                    help="Directory for LightRAG data storage"
                )
                
                input_dir = st.text_input(
                    "Input Directory", 
                    value=os.getenv("INPUT_DIR", "../storage/documents"),
                    help="Directory for uploaded documents"
                )
            
            with col2:
                log_dir = st.text_input(
                    "Log Directory",
                    value=os.getenv("LOG_DIR", "../storage/logs"),
                    help="Directory for application logs"
                )
                
                # Create directories button
                if st.button("📁 Create Directories"):
                    self.create_storage_directories(working_dir, input_dir, log_dir)
        
        # Database Configurations
        self.render_database_configs()
    
    def render_performance_config(self):
        """Render performance configuration"""
        
        st.subheader("⚡ Performance Settings")
        
        # Query Configuration
        with st.expander("🔍 Query Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                top_k = st.slider(
                    "Top K Entities/Relations",
                    min_value=1, max_value=200,
                    value=int(os.getenv("TOP_K", "40")),
                    help="Number of entities/relationships to retrieve"
                )
                
                chunk_top_k = st.slider(
                    "Chunk Top K",
                    min_value=1, max_value=50,
                    value=int(os.getenv("CHUNK_TOP_K", "10")),
                    help="Number of text chunks to retrieve"
                )
                
                max_total_tokens = st.slider(
                    "Max Total Tokens",
                    min_value=5000, max_value=50000, step=1000,
                    value=int(os.getenv("MAX_TOTAL_TOKENS", "30000")),
                    help="Maximum tokens for entire query context"
                )
            
            with col2:
                max_entity_tokens = st.slider(
                    "Max Entity Tokens",
                    min_value=1000, max_value=20000, step=1000,
                    value=int(os.getenv("MAX_ENTITY_TOKENS", "10000")),
                    help="Maximum tokens for entity context"
                )
                
                max_relation_tokens = st.slider(
                    "Max Relation Tokens", 
                    min_value=1000, max_value=20000, step=1000,
                    value=int(os.getenv("MAX_RELATION_TOKENS", "10000")),
                    help="Maximum tokens for relationship context"
                )
                
                related_chunk_number = st.slider(
                    "Related Chunk Number",
                    min_value=1, max_value=20,
                    value=int(os.getenv("RELATED_CHUNK_NUMBER", "5")),
                    help="Number of related chunks per entity/relation"
                )
        
        # Concurrency Configuration
        with st.expander("⚡ Concurrency Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                max_async = st.slider(
                    "Max Async Requests",
                    min_value=1, max_value=20,
                    value=int(os.getenv("MAX_ASYNC", "4")),
                    help="Maximum concurrent LLM requests"
                )
                
                max_parallel_insert = st.slider(
                    "Max Parallel Insert",
                    min_value=1, max_value=10,
                    value=int(os.getenv("MAX_PARALLEL_INSERT", "2")),
                    help="Maximum parallel document processing"
                )
            
            with col2:
                embedding_max_async = st.slider(
                    "Max Embedding Async",
                    min_value=1, max_value=20,
                    value=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "8")),
                    help="Maximum concurrent embedding requests"
                )
        
        # Document Processing
        with st.expander("📄 Document Processing", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                chunk_size = st.slider(
                    "Chunk Size",
                    min_value=100, max_value=2000, step=100,
                    value=int(os.getenv("CHUNK_SIZE", "1200")),
                    help="Size of text chunks for processing"
                )
                
                chunk_overlap = st.slider(
                    "Chunk Overlap",
                    min_value=0, max_value=500, step=50,
                    value=int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),
                    help="Overlap between adjacent chunks"
                )
            
            with col2:
                max_gleaning = st.slider(
                    "Max Gleaning Attempts",
                    min_value=1, max_value=5,
                    value=int(os.getenv("MAX_GLEANING", "1")),
                    help="Maximum entity extraction attempts"
                )
                
                force_summary_merge = st.slider(
                    "Force Summary on Merge",
                    min_value=2, max_value=10,
                    value=int(os.getenv("FORCE_LLM_SUMMARY_ON_MERGE", "4")),
                    help="Entity count threshold for LLM re-summary"
                )
    
    def render_advanced_config(self):
        """Render advanced configuration options"""
        
        st.subheader("🔧 Advanced Settings")
        
        # Cache Management Section - NEW
        with st.expander("🗄️ Cache Management", expanded=False):
            st.write("**System Cache Controls**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🗑️ Clear LightRAG Cache", use_container_width=True):
                    self.clear_lightrag_cache()
            
            with col2:
                if st.button("💾 Clear Streamlit Cache", use_container_width=True):
                    self.clear_streamlit_cache()
            
            with col3:
                if st.button("📊 Clear Token History", use_container_width=True):
                    self.clear_token_history()
            
            st.markdown("---")
            
            # Cache Statistics
            cache_stats = self.get_cache_statistics()
            if cache_stats:
                st.write("**Cache Statistics:**")
                for cache_type, stats in cache_stats.items():
                    st.write(f"- **{cache_type}**: {stats}")
        
        # Workspace Configuration
        with st.expander("🏢 Workspace Settings", expanded=True):
            workspace = st.text_input(
                "Workspace Name",
                value=os.getenv("WORKSPACE", ""),
                help="Workspace name for data isolation (optional)"
            )
            
            if workspace:
                st.info(f"Data will be isolated under workspace: '{workspace}'")
        
        # Logging Configuration
        with st.expander("📝 Logging Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                log_level = st.selectbox(
                    "Log Level",
                    ["DEBUG", "INFO", "WARNING", "ERROR"],
                    index=["DEBUG", "INFO", "WARNING", "ERROR"].index(os.getenv("LOG_LEVEL", "INFO")),
                    help="Logging verbosity level"
                )
                
                verbose_debug = st.checkbox(
                    "Verbose Debug",
                    value=os.getenv("VERBOSE", "false").lower() == "true",
                    help="Enable detailed debug output"
                )
            
            with col2:
                log_max_bytes = st.number_input(
                    "Log Max Bytes",
                    min_value=1000000, max_value=100000000, step=1000000,
                    value=int(os.getenv("LOG_MAX_BYTES", "10485760")),
                    help="Maximum log file size in bytes"
                )
                
                log_backup_count = st.number_input(
                    "Log Backup Count",
                    min_value=1, max_value=20,
                    value=int(os.getenv("LOG_BACKUP_COUNT", "5")),
                    help="Number of log backup files to keep"
                )
        
        # Caching Configuration
        with st.expander("💾 Caching Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_llm_cache_extract = st.checkbox(
                    "Enable LLM Cache for Extraction",
                    value=os.getenv("ENABLE_LLM_CACHE_FOR_EXTRACT", "true").lower() == "true",
                    help="Cache LLM responses during entity extraction"
                )
                
                embedding_cache_enabled = st.checkbox(
                    "Enable Embedding Cache",
                    value=os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true",
                    help="Cache embedding computations"
                )
            
            with col2:
                if embedding_cache_enabled:
                    embedding_cache_threshold = st.slider(
                        "Embedding Cache Similarity Threshold",
                        min_value=0.0, max_value=1.0, step=0.05,
                        value=float(os.getenv("EMBEDDING_CACHE_SIMILARITY_THRESHOLD", "0.90")),
                        help="Similarity threshold for embedding cache hits"
                    )
        
        # Summary Language
        with st.expander("🌍 Language Settings", expanded=True):
            summary_language = st.selectbox(
                "Summary Language",
                ["English", "Chinese", "French", "German", "Spanish", "Japanese"],
                index=0,
                help="Language for generated summaries"
            )
    
    def render_database_configs(self):
        """Render database-specific configurations"""
        
        # Redis Configuration
        with st.expander("🔴 Redis Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                redis_uri = st.text_input(
                    "Redis URI",
                    value=os.getenv("REDIS_URI", "redis://localhost:6379"),
                    help="Redis connection URI"
                )
                
                redis_max_connections = st.number_input(
                    "Max Connections",
                    min_value=10, max_value=200,
                    value=int(os.getenv("REDIS_MAX_CONNECTIONS", "100")),
                    help="Maximum Redis connections"
                )
            
            with col2:
                redis_timeout = st.number_input(
                    "Socket Timeout",
                    min_value=5, max_value=120,
                    value=int(os.getenv("REDIS_SOCKET_TIMEOUT", "30")),
                    help="Redis socket timeout in seconds"
                )
        
        # Neo4j Configuration
        with st.expander("🟢 Neo4j Configuration", expanded=False):
            neo4j_uri = st.text_input(
                "Neo4j URI",
                value=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
                help="Neo4j database URI"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                neo4j_username = st.text_input(
                    "Neo4j Username",
                    value=os.getenv("NEO4J_USERNAME", "neo4j"),
                    help="Neo4j database username"
                )
            
            with col2:
                neo4j_password = st.text_input(
                    "Neo4j Password",
                    value=os.getenv("NEO4J_PASSWORD", ""),
                    type="password",
                    help="Neo4j database password"
                )
    
    def render_config_actions(self):
        """Render configuration save/load actions"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save Configuration", use_container_width=True):
                self.save_configuration()
        
        with col2:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                self.reset_to_defaults()
        
        with col3:
            if st.button("📥 Export Config", use_container_width=True):
                self.export_configuration()
        
        with col4:
            uploaded_config = st.file_uploader(
                "📤 Import Config",
                type=["json"],
                help="Upload a configuration file"
            )
            
            if uploaded_config:
                self.import_configuration(uploaded_config)
    
    def render_server_status(self):
        """Render server status information"""
        
        try:
            result = self.api_client.get_server_info()
            
            if result["success"]:
                info = result["data"]
                st.success("✅ Server Online")
                
                with st.expander("Server Info", expanded=False):
                    st.json(info)
            else:
                st.error("❌ Server Offline")
        
        except Exception:
            st.warning("⚠️ Server Status Unknown")
    
    def test_gemini_connection(self, api_key: str):
        """Test Gemini API connection"""
        
        if not api_key:
            st.error("Please provide a Gemini API key")
            return
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=api_key)
            
            # Test with a simple request
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content("Hello")
            
            if response.text:
                st.success("✅ Gemini API connection successful!")
            else:
                st.error("❌ Gemini API test failed")
        
        except Exception as e:
            st.error(f"❌ Gemini API connection failed: {str(e)}")
    
    def test_lightrag_connection(self, api_base_url: str):
        """Test LightRAG API connection"""
        
        result = test_api_connection(api_base_url)
        
        if result["connected"]:
            st.success("✅ LightRAG API connection successful!")
            
            if result["server_info"]:
                with st.expander("Server Details", expanded=False):
                    st.json(result["server_info"])
        else:
            st.error(f"❌ LightRAG API connection failed: {result.get('error', 'Unknown error')}")
    
    def create_storage_directories(self, working_dir: str, input_dir: str, log_dir: str):
        """Create storage directories"""
        
        try:
            directories = [working_dir, input_dir, log_dir]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            st.success(f"✅ Created directories: {', '.join(directories)}")
        
        except Exception as e:
            st.error(f"❌ Failed to create directories: {str(e)}")
    
    def save_configuration(self):
        """Save current configuration"""
        
        st.success("✅ Configuration saved! Restart the server to apply changes.")
        st.session_state.config_saved = True
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        
        if st.button("⚠️ Confirm Reset to Defaults"):
            st.info("🔄 Configuration reset to defaults")
            st.rerun()
        else:
            st.warning("⚠️ Click 'Confirm Reset to Defaults' to proceed")
    
    def export_configuration(self):
        """Export current configuration as JSON"""
        
        config_data = {
            "gemini_api_key": "REDACTED",
            "llm_model": os.getenv("LLM_MODEL", "gemini-2.0-flash"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-004"),
            "export_timestamp": st.session_state.get("export_timestamp", "")
        }
        
        json_data = json.dumps(config_data, indent=2)
        
        st.download_button(
            label="📥 Download Configuration",
            data=json_data,
            file_name="lightrag_config.json",
            mime="application/json"
        )
    
    def import_configuration(self, uploaded_file):
        """Import configuration from uploaded file"""
        
        try:
            config_data = json.load(uploaded_file)
            st.success("✅ Configuration imported successfully!")
            st.json(config_data)
        
        except Exception as e:
            st.error(f"❌ Failed to import configuration: {str(e)}") 

    def clear_lightrag_cache(self):
        """Clear LightRAG LLM response cache"""
        
        try:
            # Note: LightRAG cache clearing requires server restart or direct API access
            # For now, we show an informational message
            st.warning("⚠️ LightRAG cache clearing requires server restart or direct database access.")
            st.info("""
            **To clear LightRAG cache manually:**
            1. Stop the LightRAG server
            2. Delete files in `storage/rag_storage/` directory
            3. Restart the server
            
            **Or use the LightRAG Python API:**
            ```python
            await rag.aclear_cache()  # Clear all cache
            await rag.aclear_cache(modes=["local", "global"])  # Clear specific modes
            ```
            """)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    def clear_streamlit_cache(self):
        """Clear Streamlit frontend cache"""
        
        try:
            # Clear Streamlit's cache
            st.cache_data.clear()
            st.success("✅ Streamlit cache cleared successfully!")
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error clearing Streamlit cache: {str(e)}")

    def clear_token_history(self):
        """Clear token usage history"""
        
        try:
            from monitoring.enhanced_token_tracker import get_global_tracker
            
            # Confirm before deleting
            if st.button("⚠️ Confirm Delete All Token History", key="confirm_token_delete"):
                tracker = get_global_tracker()
                deleted_count = tracker.cleanup_old_data(days_to_keep=0)  # Delete all
                
                st.success(f"✅ Cleared {deleted_count} token history records!")
                st.rerun()
            else:
                st.warning("⚠️ This will permanently delete all token usage history.")
                st.info("Click 'Confirm Delete All Token History' to proceed.")
        
        except ImportError as e:
            st.error(f"❌ Token tracker not available: {e}")
        except Exception as e:
            st.error(f"❌ Error clearing token history: {str(e)}")

    def get_cache_statistics(self) -> Dict[str, str]:
        """Get cache usage statistics with proper error handling"""
        
        stats = {}
        
        try:
            # Streamlit cache info
            stats["Streamlit Cache"] = "In-memory (check browser dev tools)"
            
            # Use absolute paths relative to app directory with proper error handling
            try:
                app_dir = Path(__file__).parent.parent
                project_root = app_dir.parent
                
                # Token tracker database size
                db_path = project_root / "storage" / "token_usage.db"
                
                if db_path.exists():
                    size_mb = db_path.stat().st_size / (1024 * 1024)
                    stats["Token Database"] = f"{size_mb:.2f} MB"
                else:
                    stats["Token Database"] = "Not created yet"
                
                # LightRAG storage directory
                storage_path = project_root / "storage" / "rag_storage"
                if storage_path.exists():
                    try:
                        total_size = sum(f.stat().st_size for f in storage_path.rglob('*') if f.is_file())
                        size_mb = total_size / (1024 * 1024)
                        stats["LightRAG Storage"] = f"{size_mb:.2f} MB"
                    except (OSError, PermissionError):
                        stats["LightRAG Storage"] = "Access denied"
                else:
                    stats["LightRAG Storage"] = "Not created yet"
                    
            except Exception as path_error:
                stats["Path Error"] = f"Could not resolve paths: {path_error}"
            
            # Check if LightRAG server is running
            try:
                server_status = self.api_client.check_health()
                if server_status:
                    stats["LightRAG Server"] = "✅ Running"
                else:
                    stats["LightRAG Server"] = "❌ Not responding"
            except Exception:
                stats["LightRAG Server"] = "❌ Connection failed"
        
        except Exception as e:
            stats["General Error"] = str(e)
        
        return stats 