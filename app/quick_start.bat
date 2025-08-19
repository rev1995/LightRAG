@echo off
REM LightRAG Gemini Quick Start Script for Windows

echo 🚀 LightRAG Gemini Quick Start
echo =================================

REM Check if .env file exists
if not exist ".env" (
    echo 📝 Setting up environment configuration...
    
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo ✅ Created .env file from template
        echo.
        echo ⚠️  IMPORTANT: You need to edit the .env file and add your Gemini API key!
        echo    1. Get your API key from: https://aistudio.google.com/
        echo    2. Edit .env file and replace 'your_gemini_api_key_here' with your actual key
        echo    3. Run this script again after setting your API key
        echo.
        pause
        
        REM Try to open the file with common editors
        if exist "%ProgramFiles%\Microsoft VS Code\Code.exe" (
            "%ProgramFiles%\Microsoft VS Code\Code.exe" .env
        ) else if exist "%ProgramFiles(x86)%\Microsoft VS Code\Code.exe" (
            "%ProgramFiles(x86)%\Microsoft VS Code\Code.exe" .env
        ) else (
            notepad .env
        )
        exit /b 1
    ) else (
        echo ❌ .env.example file not found!
        echo Please make sure you're in the app directory
        pause
        exit /b 1
    )
)

REM Check if GEMINI_API_KEY is set
findstr /C:"your_gemini_api_key_here" .env >nul
if %errorlevel% equ 0 (
    echo ❌ Gemini API key not configured!
    echo Please edit .env file and replace 'your_gemini_api_key_here' with your actual API key
    pause
    exit /b 1
)

echo ✅ Environment configuration found

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python found

REM Create storage directories
echo 📁 Creating storage directories...
if not exist "..\storage" mkdir "..\storage"
if not exist "..\storage\rag_storage" mkdir "..\storage\rag_storage"
if not exist "..\storage\documents" mkdir "..\storage\documents"
if not exist "..\storage\logs" mkdir "..\storage\logs"
if not exist "..\storage\exports" mkdir "..\storage\exports"
echo ✅ Storage directories created

REM Install dependencies
echo 📦 Installing dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed successfully
) else (
    echo ❌ requirements.txt not found!
    pause
    exit /b 1
)

REM Test LightRAG import
echo 🔧 Testing LightRAG setup...
python setup_lightrag.py
if %errorlevel% neq 0 (
    echo ❌ LightRAG setup failed
    pause
    exit /b 1
)
echo ✅ LightRAG setup verified

echo.
echo 🎉 Setup completed successfully!
echo.
echo 🚀 To start the application:
echo.
echo Terminal 1 - LightRAG API Server:
echo   python main_server.py
echo.
echo Terminal 2 - Streamlit Frontend:
echo   streamlit run streamlit_app.py
echo.
echo 📱 Access URLs:
echo   • Streamlit App: http://localhost:8501
echo   • API Documentation: http://localhost:9621/docs
echo.

REM Ask if user wants to start automatically
set /p start_servers="Would you like to start the servers now? (y/N): "

if /i "%start_servers%"=="y" (
    echo.
    echo 🚀 Starting LightRAG API Server...
    
    REM Start API server in new window
    start "LightRAG API Server" cmd /k "python main_server.py"
    
    REM Wait a moment for server to start
    timeout /t 5 /nobreak >nul
    
    echo ✅ API Server started in new window
    echo.
    echo 🌐 Starting Streamlit Frontend...
    
    REM Start Streamlit
    streamlit run streamlit_app.py
    
) else (
    echo.
    echo ℹ️  Run the commands above manually to start the servers
    pause
) 