#!/usr/bin/env python3
"""
LightRAG Frontend Startup Script

This script starts the LightRAG web UI with proper configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the LightRAG frontend"""
    
    # Check if we're in the right directory
    if not Path("lightrag_webui").exists():
        print("❌ Error: lightrag_webui directory not found")
        print("Please run this script from the LightRAG directory")
        sys.exit(1)
    
    # Change to webui directory
    webui_dir = Path("lightrag_webui")
    os.chdir(webui_dir)
    
    # Check for package.json
    if not Path("package.json").exists():
        print("❌ Error: package.json not found in lightrag_webui")
        print("Please ensure the frontend is properly set up")
        sys.exit(1)
    
    # Check for environment file
    env_file = ".env.local"
    if not Path(env_file).exists():
        print(f"⚠️  Warning: {env_file} not found")
        print("Creating default .env.local...")
        
        # Create default environment file
        with open(env_file, "w") as f:
            f.write("""# LightRAG Frontend Environment Configuration

# API Configuration
VITE_API_URL=http://localhost:8000
VITE_API_PROXY=http://localhost:8000
VITE_API_ENDPOINTS=http://localhost:8000
VITE_BACKEND_URL=http://localhost:8000
""")
        print("✅ Created .env.local with default configuration")
    
    print("🚀 Starting LightRAG Frontend...")
    print("🌐 Web UI: http://localhost:5173")
    print("📖 API Backend: http://localhost:8000")
    print("=" * 50)
    
    # Check if node_modules exists
    if not Path("node_modules").exists():
        print("📦 Installing dependencies...")
        try:
            subprocess.run(["npm", "install"], check=True)
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            sys.exit(1)
    
    # Start the development server
    try:
        print("🎯 Starting development server...")
        subprocess.run(["npm", "run", "dev"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start frontend: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped by user")

if __name__ == "__main__":
    main() 