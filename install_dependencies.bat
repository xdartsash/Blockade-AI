@echo off
setlocal
cd /d "%~dp0"
echo Installing Python dependencies for BLOCKADE AI...
echo This step needs internet access to download PyTorch and Ultralytics.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_local.ps1"
if errorlevel 1 goto failed
echo.
echo Installation finished. Follow START_RU.md to collect and label game frames.
pause
exit /b 0
:failed
echo.
echo Installation failed. Keep this window open and send a screenshot of the error.
pause
exit /b 1