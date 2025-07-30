#!/usr/bin/env python3
"""
LightRAG Gemini 2.0 Flash Server Startup Script
Comprehensive startup script with validation, dependency checking, and graceful management.
"""

import os
import sys
import subprocess
import argparse
import time
import signal
import atexit
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for pretty terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_colored(message: str, color: str = Colors.WHITE):
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.END}")


def print_banner():
    """Print startup banner"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LightRAG Gemini 2.0 Flash Server                         ║
║                  Production-Ready RAG Pipeline System                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}
"""
    print(banner)


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print_colored("❌ Python 3.8 or higher is required", Colors.RED)
        sys.exit(1)
    
    print_colored(f"✅ Python {sys.version.split()[0]} detected", Colors.GREEN)


def check_env_file():
    """Check if .env file exists and validate key variables"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print_colored("⚠️  .env file not found. Using environment variables only.", Colors.YELLOW)
        return False
    
    print_colored("✅ .env file found", Colors.GREEN)
    
    # Load and validate key environment variables
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env", override=False)
    
    required_vars = [
        "GEMINI_API_KEY",
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print_colored(f"❌ Missing required environment variables: {', '.join(missing_vars)}", Colors.RED)
        print_colored("Please set these variables in your .env file or environment", Colors.YELLOW)
        return False
    
    print_colored("✅ Key environment variables configured", Colors.GREEN)
    return True


def check_dependencies():
    """Check if required dependencies are installed"""
    print_colored("\n🔍 Checking dependencies...", Colors.BLUE)
    
    required_packages = [
        ("lightrag", "lightrag-hku"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("google.genai", "google-genai"),
        ("sentencepiece", "sentencepiece"),
        ("dotenv", "python-dotenv"),
        ("numpy", "numpy"),
        ("aiofiles", "aiofiles"),
    ]
    
    missing_packages = []
    
    for package, pip_name in required_packages:
        try:
            __import__(package)
            print_colored(f"  ✅ {package}", Colors.GREEN)
        except ImportError:
            print_colored(f"  ❌ {package}", Colors.RED)
            missing_packages.append(pip_name)
    
    if missing_packages:
        print_colored(f"\n❌ Missing packages: {', '.join(missing_packages)}", Colors.RED)
        print_colored("Install with: pip install " + " ".join(missing_packages), Colors.YELLOW)
        return False
    
    print_colored("✅ All required dependencies are installed", Colors.GREEN)
    return True


def create_directories():
    """Create necessary directories"""
    directories = [
        "./inputs",
        "./rag_storage", 
        "./logs",
        "./tokenizer_cache",
        "./embedding_cache",
        "./webui"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    print_colored("✅ Required directories created", Colors.GREEN)


def validate_gemini_config():
    """Validate Gemini configuration"""
    print_colored("\n🔑 Validating Gemini configuration...", Colors.BLUE)
    
    try:
        from gemini_llm import validate_gemini_config
        from gemini_embeddings import validate_gemini_embedding_config
        from gemma_tokenizer import validate_tokenizer_setup
        
        if not validate_gemini_config():
            print_colored("❌ Gemini LLM configuration validation failed", Colors.RED)
            return False
            
        if not validate_gemini_embedding_config():
            print_colored("❌ Gemini embedding configuration validation failed", Colors.RED)
            return False
            
        if not validate_tokenizer_setup():
            print_colored("❌ Gemma tokenizer validation failed", Colors.RED)
            return False
        
        print_colored("✅ Gemini configuration validated", Colors.GREEN)
        return True
        
    except Exception as e:
        print_colored(f"❌ Configuration validation error: {e}", Colors.RED)
        return False


def get_server_config():
    """Get server configuration from environment"""
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env", override=False)
    
    config = {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "9621")),
        "workers": int(os.getenv("WORKERS", "1")),
        "workspace": os.getenv("WORKSPACE", "gemini_production"),
        "llm_model": os.getenv("LLM_MODEL", "gemini-2.0-flash"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-004"),
        "debug": os.getenv("DEBUG_MODE", "false").lower() == "true",
        # SSL removed for simplicity
    }
    
    return config


def display_config(config: Dict):
    """Display server configuration"""
    print_colored("\n⚙️  Server Configuration:", Colors.BLUE)
    print_colored(f"  Host: {config['host']}", Colors.WHITE)
    print_colored(f"  Port: {config['port']}", Colors.WHITE)
    print_colored(f"  Workers: {config['workers']}", Colors.WHITE)
    print_colored(f"  Workspace: {config['workspace']}", Colors.WHITE)
    print_colored(f"  LLM Model: {config['llm_model']}", Colors.WHITE)
    print_colored(f"  Embedding Model: {config['embedding_model']}", Colors.WHITE)
    print_colored(f"  Debug Mode: {config['debug']}", Colors.WHITE)
    print_colored("  SSL: Disabled (removed for simplicity)", Colors.WHITE)


def check_port_availability(host: str, port: int):
    """Check if port is available"""
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
        return True
    except OSError:
        return False


def install_requirements():
    """Install requirements from requirements.txt"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print_colored("⚠️  requirements.txt not found, skipping dependency installation", Colors.YELLOW)
        return True
    
    print_colored("\n📦 Installing requirements...", Colors.BLUE)
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        print_colored("✅ Requirements installed successfully", Colors.GREEN)
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed to install requirements: {e}", Colors.RED)
        return False


def run_tests():
    """Run basic system tests"""
    print_colored("\n🧪 Running system tests...", Colors.BLUE)
    
    try:
        # Test imports
        from lightrag_gemini_server import app
        from gemini_llm import get_gemini_llm
        from gemini_embeddings import get_gemini_embeddings
        from gemma_tokenizer import get_gemma_tokenizer
        
        print_colored("  ✅ All modules import successfully", Colors.GREEN)
        
        # Test tokenizer
        tokenizer = get_gemma_tokenizer()
        test_tokens = tokenizer.count_tokens("This is a test sentence.")
        print_colored(f"  ✅ Tokenizer working ({test_tokens} tokens)", Colors.GREEN)
        
        print_colored("✅ System tests passed", Colors.GREEN)
        return True
        
    except Exception as e:
        print_colored(f"❌ System tests failed: {e}", Colors.RED)
        return False


def start_server(config: Dict, development: bool = False):
    """Start the server"""
    print_colored(f"\n🚀 Starting LightRAG Gemini server...", Colors.GREEN)
    
    # Build command
    cmd = [
        sys.executable, "lightrag_gemini_server.py"
    ]
    
    print_colored(f"🌐 Server will be available at:", Colors.CYAN)
    print_colored(f"   API Documentation: http://{config['host']}:{config['port']}/docs", Colors.CYAN)
    print_colored(f"   Health Check: http://{config['host']}:{config['port']}/health", Colors.CYAN)
    
    print_colored("\n⏱️  Starting server (this may take a moment for initial setup)...", Colors.YELLOW)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print_colored("\n👋 Server shutdown requested", Colors.YELLOW)
    except Exception as e:
        print_colored(f"\n❌ Server failed: {e}", Colors.RED)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LightRAG Gemini 2.0 Flash Server")
    parser.add_argument("--dev", action="store_true", help="Run in development mode with auto-reload")
    parser.add_argument("--install", action="store_true", help="Install requirements before starting")
    parser.add_argument("--test", action="store_true", help="Run tests only")
    parser.add_argument("--validate", action="store_true", help="Validate configuration only")
    parser.add_argument("--skip-checks", action="store_true", help="Skip validation checks")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Basic checks
    check_python_version()
    
    if args.install:
        if not install_requirements():
            sys.exit(1)
    
    if not args.skip_checks:
        # Environment check
        if not check_env_file():
            print_colored("\n⚠️  Continuing with potential configuration issues...", Colors.YELLOW)
        
        # Dependencies check
        if not check_dependencies():
            print_colored("\n💡 Try running with --install to install dependencies", Colors.BLUE)
            sys.exit(1)
        
        # Create directories
        create_directories()
        
        # Validate Gemini configuration
        if not validate_gemini_config():
            sys.exit(1)
    
    # Get configuration
    config = get_server_config()
    display_config(config)
    
    if args.validate:
        print_colored("\n✅ Configuration validation completed", Colors.GREEN)
        return
    
    # Run tests
    if args.test or not args.skip_checks:
        if not run_tests():
            if not args.test:
                print_colored("\n⚠️  Continuing despite test failures...", Colors.YELLOW)
            else:
                sys.exit(1)
    
    if args.test:
        print_colored("\n✅ All tests completed", Colors.GREEN)
        return
    
    # Check port availability
    if not check_port_availability(config["host"], config["port"]):
        print_colored(f"❌ Port {config['port']} is already in use", Colors.RED)
        print_colored("💡 Try changing the PORT in your .env file", Colors.BLUE)
        sys.exit(1)
    
    # Start server
    start_server(config, development=args.dev)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\n👋 Goodbye!", Colors.BLUE)
    except Exception as e:
        print_colored(f"\n💥 Unexpected error: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
        sys.exit(1) 