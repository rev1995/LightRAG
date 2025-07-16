#!/usr/bin/env python3
"""
LightRAG Backend Startup Script

This script starts the LightRAG API server with proper configuration.
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Start the LightRAG API server"""
    
    # Check if we're in the right directory
    if not Path("api_server.py").exists():
        print("❌ Error: api_server.py not found in current directory")
        print("Please run this script from the LightRAG directory")
        sys.exit(1)
    
    # Check for environment file
    env_file = ".env.backend"
    if not Path(env_file).exists():
        print(f"⚠️  Warning: {env_file} not found")
        print("Please create .env.backend with your configuration")
        print("You can copy from .env.backend.example")
    
    # Check for required environment variables
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable is required")
        print("Please set it in your .env.backend file")
        sys.exit(1)
    
    print("🚀 Starting LightRAG API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("=" * 50)
    
    # Start the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main() 