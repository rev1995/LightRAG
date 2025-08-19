"""
LightRAG API Client for Streamlit Frontend
Handles communication with the LightRAG server
"""

import requests
import json
import streamlit as st
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time

class LightRAGAPIClient:
    """Client for communicating with LightRAG API server"""
    
    def __init__(self, base_url: str = "http://localhost:9621"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        params: Optional[Dict] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Make HTTP request to LightRAG API"""
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=timeout)
            elif method.upper() == "POST":
                if files:
                    response = self.session.post(url, data=data, files=files, timeout=timeout)
                else:
                    response = self.session.post(url, json=data, timeout=timeout)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, json=data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            
            # Try to parse JSON response
            try:
                return {
                    "success": True,
                    "data": response.json(),
                    "status_code": response.status_code
                }
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "data": response.text,
                    "status_code": response.status_code
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }
    
    # Health and Status
    def check_health(self) -> bool:
        """Check if LightRAG server is healthy"""
        result = self._make_request("GET", "/health", timeout=5)
        return result["success"]
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information and status"""
        return self._make_request("GET", "/")
    
    # Document Management
    def upload_document(self, file_path: str, document_name: Optional[str] = None) -> Dict[str, Any]:
        """Upload a document to LightRAG"""
        
        file_path = Path(file_path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        
        with open(file_path, 'rb') as f:
            files = {
                'file': (document_name or file_path.name, f, 'application/octet-stream')
            }
            return self._make_request("POST", "/documents/upload", files=files, timeout=120)
    
    def upload_document_content(self, content: str, filename: str) -> Dict[str, Any]:
        """Upload document content as text"""
        data = {
            "content": content,
            "filename": filename
        }
        return self._make_request("POST", "/documents/upload_text", data=data)
    
    def list_documents(self) -> Dict[str, Any]:
        """List all uploaded documents"""
        return self._make_request("GET", "/documents")
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document"""
        return self._make_request("DELETE", f"/documents/{document_id}")
    
    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """Get document processing status"""
        return self._make_request("GET", f"/documents/{document_id}/status")
    
    # Query Operations
    def query(
        self, 
        query: str,
        mode: str = "mix",
        top_k: Optional[int] = None,
        chunk_top_k: Optional[int] = None,
        response_type: Optional[str] = None,
        only_need_context: bool = False,
        only_need_prompt: bool = False,
        enable_rerank: Optional[bool] = None,
        max_total_tokens: Optional[int] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Query the knowledge base"""
        
        data = {
            "query": query,
            "mode": mode,
            "only_need_context": only_need_context,
            "only_need_prompt": only_need_prompt
        }
        
        # Add optional parameters
        if top_k is not None:
            data["top_k"] = top_k
        if chunk_top_k is not None:
            data["chunk_top_k"] = chunk_top_k
        if response_type is not None:
            data["response_type"] = response_type
        if enable_rerank is not None:
            data["enable_rerank"] = enable_rerank
        if max_total_tokens is not None:
            data["max_total_tokens"] = max_total_tokens
        if conversation_history is not None:
            data["conversation_history"] = conversation_history
            
        return self._make_request("POST", "/query", data=data, timeout=120)
    
    def stream_query(
        self, 
        query: str,
        mode: str = "mix",
        **kwargs
    ):
        """Stream query responses (generator)"""
        # Note: This would need server-side streaming support
        # For now, return regular query
        return self.query(query, mode, **kwargs)
    
    # Graph Operations
    def get_graph_data(
        self, 
        max_nodes: int = 100,
        include_embeddings: bool = False
    ) -> Dict[str, Any]:
        """Get graph data for visualization"""
        params = {
            "max_nodes": max_nodes,
            "include_embeddings": include_embeddings
        }
        return self._make_request("GET", "/graph/export", params=params)
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return self._make_request("GET", "/graph/statistics")
    
    def export_graph(self, format: str = "graphml") -> Dict[str, Any]:
        """Export graph in specified format"""
        params = {"format": format}
        return self._make_request("GET", "/graph/export", params=params)
    
    # Analytics and Monitoring
    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        return self._make_request("GET", "/metrics/token_usage")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self._make_request("GET", "/metrics/performance")
    
    def get_query_history(self, limit: int = 50) -> Dict[str, Any]:
        """Get query history"""
        params = {"limit": limit}
        return self._make_request("GET", "/metrics/queries", params=params)


class StreamlitAPIClient(LightRAGAPIClient):
    """Enhanced API client with Streamlit integration"""
    
    def __init__(self, base_url: str = "http://localhost:9621"):
        super().__init__(base_url)
    
    @st.cache_data(ttl=30)
    def cached_get_graph_data(_self, max_nodes: int = 100) -> Dict[str, Any]:
        """Cached version of get_graph_data for better performance"""
        return _self.get_graph_data(max_nodes=max_nodes)
    
    @st.cache_data(ttl=60)
    def cached_get_documents(_self) -> Dict[str, Any]:
        """Cached version of list_documents"""
        return _self.list_documents()
    
    def query_with_progress(
        self, 
        query: str,
        progress_placeholder=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Query with progress indication"""
        
        if progress_placeholder:
            progress_placeholder.write("🔍 Processing query...")
            
        start_time = time.time()
        result = self.query(query, **kwargs)
        end_time = time.time()
        
        if progress_placeholder:
            if result["success"]:
                progress_placeholder.write(f"✅ Query completed in {end_time - start_time:.2f}s")
            else:
                progress_placeholder.write(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                
        return result
    
    def upload_with_progress(
        self, 
        file_path: str,
        progress_placeholder=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Upload document with progress indication"""
        
        if progress_placeholder:
            progress_placeholder.write("📄 Uploading document...")
            
        result = self.upload_document(file_path, **kwargs)
        
        if progress_placeholder:
            if result["success"]:
                progress_placeholder.write("✅ Document uploaded successfully")
            else:
                progress_placeholder.write(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
                
        return result


# Global client instance for session management
def get_api_client() -> StreamlitAPIClient:
    """Get or create API client instance"""
    
    if "api_client" not in st.session_state:
        base_url = st.session_state.get("config", {}).get(
            "lightrag_api_base", 
            "http://localhost:9621"
        )
        st.session_state.api_client = StreamlitAPIClient(base_url)
    
    return st.session_state.api_client


def test_api_connection(base_url: str) -> Dict[str, Any]:
    """Test API connection and return status"""
    
    client = LightRAGAPIClient(base_url)
    
    result = {
        "connected": False,
        "health": False,
        "server_info": None,
        "error": None
    }
    
    try:
        # Test basic connectivity
        health_result = client.check_health()
        result["health"] = health_result
        
        if health_result:
            # Get server info
            info_result = client.get_server_info()
            if info_result["success"]:
                result["server_info"] = info_result["data"]
                result["connected"] = True
            else:
                result["error"] = info_result.get("error", "Failed to get server info")
        else:
            result["error"] = "Health check failed"
            
    except Exception as e:
        result["error"] = str(e)
    
    return result 