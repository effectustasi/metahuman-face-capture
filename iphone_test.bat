@echo off
REM Live Link Face paketini dogrular - cift tiklayarak calistir.
REM Blender-de canli yakalama aciksa once ESC.
cd /d "%~dp0"
echo.
"detector\.venv\Scripts\python.exe" "scripts\llf_probe.py"
echo.
pause
