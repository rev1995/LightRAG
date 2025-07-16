#!/usr/bin/env python3
"""
LightRAG Full Application Startup Script
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def main():
    """Start the full LightRAG application"""
    print("🚀 LightRAG Full Application Startup")
    print("=" * 50)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is already running")
        else:
            print("❌ Backend is not responding properly")
    except:
        print("❌ Backend is not running")
        print("💡 Please start the backend first with: python start_backend.py")
        return
    
    # Check if frontend is running
    try:
        response = requests.get("http://localhost:5173", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is already running")
        else:
            print("❌ Frontend is not responding properly")
    except:
        print("❌ Frontend is not running")
        print("💡 Please start the frontend with: cd lightrag_webui && npm run dev")
        return
    
    print("\n🎉 Both services are running!")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Web UI: http://localhost:5173")
    print("🔍 Health Check: http://localhost:8000/health")
    print("\n💡 Press Ctrl+C to stop")

if __name__ == "__main__":
    main() 