@echo off
REM Kullanilabilir kamera indekslerini tarar.
cd /d "%~dp0"
echo.
"detector\.venv\Scripts\python.exe" "detector\detect.py" cameras
echo.
pause
