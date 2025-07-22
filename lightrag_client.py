import requests
import json
import argparse
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
import sys

# Load environment variables
load_dotenv()

class LightRAGClient:
    """Client for interacting with the LightRAG API server"""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """Initialize the LightRAG client
        
        Args:
            base_url: Base URL of the API server
            api_key: API key for authentication
        """
        self.base_url = base_url or os.getenv("LIGHTRAG_API_URL", "http://localhost:9621")
        self.api_key = api_key or os.getenv("LIGHTRAG_API_KEY", "")
        
        # Set up headers
        self.headers = {}
        if self.api_key:
            self.headers["X-API-Key"] = self.api_key
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the API server"""
        response = requests.get(f"{self.base_url}/health", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def query(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """Query the RAG system
        
        Args:
            query_text: The query text
            **kwargs: Additional query parameters
            
        Returns:
            Dict[str, Any]: Query response
        """
        payload = {"query": query_text, **kwargs}
        response = requests.post(
            f"{self.base_url}/api/query", 
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def insert_text(self, text: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Insert text into the RAG system
        
        Args:
            text: Text content to insert
            doc_id: Optional document ID
            
        Returns:
            Dict[str, Any]: Insertion response
        """
        payload = {"text": text}
        if doc_id:
            payload["doc_id"] = doc_id
            
        response = requests.post(
            f"{self.base_url}/api/documents/insert/text",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def insert_file(self, file_path: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Insert a file into the RAG system
        
        Args:
            file_path: Path to the file
            doc_id: Optional document ID
            
        Returns:
            Dict[str, Any]: Insertion response
        """
        files = {"file": open(file_path, "rb")}
        data = {}
        if doc_id:
            data["doc_id"] = doc_id
            
        response = requests.post(
            f"{self.base_url}/api/documents/insert/file",
            headers=self.headers,
            files=files,
            data=data
        )
        response.raise_for_status()
        return response.json()
    
    def get_document_status(self, doc_id: str) -> Dict[str, Any]:
        """Get the status of a document
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dict[str, Any]: Document status
        """
        response = requests.get(
            f"{self.base_url}/api/documents/{doc_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def list_documents(self) -> Dict[str, Any]:
        """List all documents in the RAG system
        
        Returns:
            Dict[str, Any]: List of documents
        """
        response = requests.get(
            f"{self.base_url}/api/documents",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document from the RAG system
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dict[str, Any]: Deletion response
        """
        response = requests.delete(
            f"{self.base_url}/api/documents/{doc_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_knowledge_graph(self, limit: int = 1000) -> Dict[str, Any]:
        """Get the knowledge graph
        
        Args:
            limit: Maximum number of nodes/edges to return
            
        Returns:
            Dict[str, Any]: Knowledge graph
        """
        response = requests.get(
            f"{self.base_url}/api/graph?limit={limit}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def clear_cache(self, modes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Clear the cache
        
        Args:
            modes: Cache modes to clear
            
        Returns:
            Dict[str, Any]: Cache clearing response
        """
        payload = {}
        if modes:
            payload["modes"] = modes
            
        response = requests.post(
            f"{self.base_url}/api/cache/clear",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()


def main():
    parser = argparse.ArgumentParser(description="LightRAG API Client")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Health check command
    subparsers.add_parser("health", help="Check API server health")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument("--mode", help="Query mode (naive, local, global, hybrid, mix, bypass)")
    query_parser.add_argument("--top-k", type=int, help="Number of top entities/relations to retrieve")
    query_parser.add_argument("--response-type", help="Response format type")
    
    # Insert text command
    insert_text_parser = subparsers.add_parser("insert-text", help="Insert text into the RAG system")
    insert_text_parser.add_argument("text", help="Text content to insert")
    insert_text_parser.add_argument("--doc-id", help="Document ID")
    
    # Insert file command
    insert_file_parser = subparsers.add_parser("insert-file", help="Insert a file into the RAG system")
    insert_file_parser.add_argument("file_path", help="Path to the file")
    insert_file_parser.add_argument("--doc-id", help="Document ID")
    
    # Document status command
    doc_status_parser = subparsers.add_parser("doc-status", help="Get document status")
    doc_status_parser.add_argument("doc_id", help="Document ID")
    
    # List documents command
    subparsers.add_parser("list-docs", help="List all documents")
    
    # Delete document command
    delete_doc_parser = subparsers.add_parser("delete-doc", help="Delete a document")
    delete_doc_parser.add_argument("doc_id", help="Document ID")
    
    # Get knowledge graph command
    kg_parser = subparsers.add_parser("get-kg", help="Get knowledge graph")
    kg_parser.add_argument("--limit", type=int, default=1000, help="Maximum number of nodes/edges")
    
    # Clear cache command
    clear_cache_parser = subparsers.add_parser("clear-cache", help="Clear cache")
    clear_cache_parser.add_argument("--modes", nargs="+", help="Cache modes to clear")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create client
    client = LightRAGClient()
    
    try:
        # Execute command
        if args.command == "health":
            result = client.health_check()
        elif args.command == "query":
            kwargs = {}
            if args.mode:
                kwargs["mode"] = args.mode
            if args.top_k:
                kwargs["top_k"] = args.top_k
            if args.response_type:
                kwargs["response_type"] = args.response_type
            
            result = client.query(args.text, **kwargs)
        elif args.command == "insert-text":
            result = client.insert_text(args.text, doc_id=args.doc_id)
        elif args.command == "insert-file":
            result = client.insert_file(args.file_path, doc_id=args.doc_id)
        elif args.command == "doc-status":
            result = client.get_document_status(args.doc_id)
        elif args.command == "list-docs":
            result = client.list_documents()
        elif args.command == "delete-doc":
            result = client.delete_document(args.doc_id)
        elif args.command == "get-kg":
            result = client.get_knowledge_graph(limit=args.limit)
        elif args.command == "clear-cache":
            result = client.clear_cache(modes=args.modes)
        else:
            parser.print_help()
            sys.exit(1)
        
        # Print result
        print(json.dumps(result, indent=2))
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()