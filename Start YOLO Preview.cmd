@echo off
chcp 65001 >nul
setlocal
set "BLOCKADE_PYTHON=C:\Users\Dartsash\Desktop\Blockade AI\.venv\Scripts\python.exe"
if not exist "%BLOCKADE_PYTHON%" (
    echo Python was not found at "%BLOCKADE_PYTHON%"
    pause
    exit /b 1
)
"%BLOCKADE_PYTHON%" -I -u "%~dp0preview_yolo.py" %*
if errorlevel 1 pause
