#!/bin/bash
# LightRAG Gemini Quick Start Script

echo "🚀 LightRAG Gemini Quick Start"
echo "================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Setting up environment configuration..."
    
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file from template"
        echo ""
        echo "⚠️  IMPORTANT: You need to edit the .env file and add your Gemini API key!"
        echo "   1. Get your API key from: https://aistudio.google.com/"
        echo "   2. Edit .env file and replace 'your_gemini_api_key_here' with your actual key"
        echo "   3. Run this script again after setting your API key"
        echo ""
        read -p "Press Enter to open .env file for editing (or Ctrl+C to exit)..."
        
        # Try to open the file with common editors
        if command -v code &> /dev/null; then
            code .env
        elif command -v nano &> /dev/null; then
            nano .env
        elif command -v vim &> /dev/null; then
            vim .env
        else
            echo "Please edit .env file manually with your preferred editor"
        fi
        exit 1
    else
        echo "❌ .env.example file not found!"
        echo "Please make sure you're in the app directory"
        exit 1
    fi
fi

# Check if GEMINI_API_KEY is set
if grep -q "your_gemini_api_key_here" .env; then
    echo "❌ Gemini API key not configured!"
    echo "Please edit .env file and replace 'your_gemini_api_key_here' with your actual API key"
    exit 1
fi

echo "✅ Environment configuration found"

# Check Python version
python_version=$(python3 --version 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+' | head -1)
if [ -z "$python_version" ]; then
    echo "❌ Python 3 not found! Please install Python 3.8+"
    exit 1
fi

major_version=$(echo $python_version | cut -d'.' -f1)
minor_version=$(echo $python_version | cut -d'.' -f2)

if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 8 ]); then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create storage directories
echo "📁 Creating storage directories..."
mkdir -p ../storage/rag_storage
mkdir -p ../storage/documents
mkdir -p ../storage/logs
mkdir -p ../storage/exports
echo "✅ Storage directories created"

# Install dependencies
echo "📦 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ Dependencies installed successfully"
    else
        echo "❌ Failed to install dependencies"
        exit 1
    fi
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Test LightRAG import
echo "🔧 Testing LightRAG setup..."
python setup_lightrag.py
if [ $? -eq 0 ]; then
    echo "✅ LightRAG setup verified"
else
    echo "❌ LightRAG setup failed"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "🚀 To start the application:"
echo ""
echo "Terminal 1 - LightRAG API Server:"
echo "  python main_server.py"
echo ""
echo "Terminal 2 - Streamlit Frontend:"
echo "  streamlit run streamlit_app.py"
echo ""
echo "📱 Access URLs:"
echo "  • Streamlit App: http://localhost:8501"
echo "  • API Documentation: http://localhost:9621/docs"
echo ""

# Ask if user wants to start automatically
read -p "Would you like to start the servers now? (y/N): " start_servers

if [[ $start_servers =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Starting LightRAG API Server..."
    
    # Start API server in background
    python main_server.py &
    api_pid=$!
    
    # Wait a moment for server to start
    sleep 5
    
    # Check if server is running
    if kill -0 $api_pid 2>/dev/null; then
        echo "✅ API Server started (PID: $api_pid)"
        echo ""
        echo "🌐 Starting Streamlit Frontend..."
        
        # Start Streamlit
        streamlit run streamlit_app.py
        
        # When Streamlit exits, cleanup
        echo ""
        echo "🛑 Stopping API Server..."
        kill $api_pid 2>/dev/null
        echo "✅ Cleanup completed"
    else
        echo "❌ Failed to start API server"
        exit 1
    fi
else
    echo ""
    echo "ℹ️  Run the commands above manually to start the servers"
fi 