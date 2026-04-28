@echo off
REM ============================================================
REM  PatentPilot AI — Launch Script (Windows)
REM ============================================================
REM  Uses Python 3.14 explicitly to avoid version conflicts.
REM  Disables Streamlit's file watcher to avoid torchvision spam.
REM ============================================================

echo.
echo   ============================================
echo     PatentPilot AI - Starting...
echo   ============================================
echo.

REM Check Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo   [WARNING] Ollama is not running!
    echo   Start it with: ollama serve
    echo   Then pull model: ollama pull llama3.2:3b
    echo.
)

echo   Starting Streamlit on http://localhost:8501
echo.

C:\Python314\python.exe -m streamlit run frontend/app.py ^
    --server.port 8501 ^
    --server.headless true ^
    --server.fileWatcherType none ^
    --server.runOnSave false
