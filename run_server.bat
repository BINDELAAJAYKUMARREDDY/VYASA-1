@echo off
echo ================================================
echo  VYASA-1 - Starting Server
echo ================================================
cd /d "%~dp0"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
