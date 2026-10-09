@echo off
title Song to Piano and Guitar Cover AI Studio
cd /d "%~dp0"

echo =======================================================
echo   Song to Piano & Guitar Cover AI Studio
echo =======================================================
echo.

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe!
    echo Please run uv venv or python -m venv .venv first.
    pause
    exit /b 1
)

echo Starting Web Application...
.venv\Scripts\python.exe app.py
pause
