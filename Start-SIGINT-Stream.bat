@echo off
title PHI-RF-CLASSIFIER: Real-Time SIGINT & PHI-GUARD Interdiction Stream
cd /d "%~dp0"
echo [*] Launching PHI-RF-CLASSIFIER Real-Time Interdiction Stream...
".\.venv\Scripts\python.exe" main.py --mode stream
pause
