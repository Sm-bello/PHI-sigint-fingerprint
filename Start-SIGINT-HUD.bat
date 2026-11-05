@echo off
title PHI-RF-CLASSIFIER: HMGCC Rich C4I SIGINT Cockpit HUD
cd /d "%~dp0"
echo [*] Launching PHI-RF-CLASSIFIER Flicker-Free C4I Cockpit HUD...
".\.venv\Scripts\python.exe" main.py --mode hud
pause
