@echo off
title PHI-RF-CLASSIFIER: Sovereign Web C4I Tactical Dashboard
cd /d "%~dp0"
echo ===============================================================================
echo   PHI-RF-CLASSIFIER: Sovereign AI SIGINT & C4I Tactical Dashboard
echo ===============================================================================
echo [*] Launching PyTorch 1D-CNN Real-Time Web Server & Opening Browser...
echo [*] URL: http://127.0.0.1:8080/
echo.
".\.venv\Scripts\python.exe" main.py --mode web
pause
