@echo off
setlocal
title Box Prediction App Setup

echo ==========================================
echo    Box Prediction App - Setup Tool
echo ==========================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ and try again.
    pause
    exit /b
)

echo [1/3] Creating Virtual Environment (venv)...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b
)

echo [2/3] Installing Dependencies (this may take a few minutes)...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b
)

echo.
echo [3/3] Setup Complete!
echo.
echo ==========================================
echo To start the app, run:
echo streamlit run streamlit_app.py
echo ==========================================
echo.

set /p choice="Would you like to start the app now? (y/n): "
if /i "%choice%"=="y" (
    echo Starting Streamlit...
    streamlit run streamlit_app.py
)

pause
