@echo off
cd /d "%~dp0"
python "%~dp0infer_force_realtime_2.py" %*
if errorlevel 1 pause
