@echo off
REM Canli yuz yakalama - cift tiklayarak calistir.
REM Once Blender-de charface > Canli Yuz Yakalama acik olmali.
cd /d "%~dp0"
echo.
echo facecap - canli yuz yakalama
echo Kapatmak icin: onizleme penceresinde q
echo.
"detector\.venv\Scripts\python.exe" "detector\detect.py" live --udp 127.0.0.1:11111 --preview %*
echo.
echo ---- surec bitti ----
pause
